//! PrefillDispatch building blocks: the TriSolve kernel - the
//! intra-chunk decay-mask build, beta-scaled kkt matmul, and the
//! 63-iteration row-serial triangular inversion collapsed into ONE
//! nvrtc launch per layer-call (the classic chain spends ~325
//! launches there, the dominant share of the prefill dispatch tax).
//! Raw pointer args on the model's own stream, operands addressed at
//! their layout offsets (the offset-leak lock).
//!
//! Numerics: the same f32 math as gdn::chunk_rule's solve block with
//! reassociation-class deltas (serial in-thread dots where cublas
//! tiles; expf where the tensor exp ran) - the checkpoint-free lock
//! pins the spread. The cpu path and the stateless parity form never
//! dispatch here; only the CARRIED cuda chunk branch does.
use crate::*;

/// One block per (head, chunk) tile, 64 threads (one per row for the
/// build, one per column for the solve). Shared memory carries the
/// beta-scaled k tile, the 64x64 working matrix, the solve's row
/// snapshot, and the cumulative gates.
const KERNEL_SRC: &str = r#"
extern "C" __global__ void tri_solve_f32(
    float* a_out,
    const float* k,
    const float* beta_g,
    unsigned int dk
) {
    extern __shared__ float shared[];
    float* sk = shared;                        // [64, dk] beta-scaled k
    float* sa = shared + 64u * dk;             // [64, 64] working matrix
    float* srow = sa + 64u * 64u;              // [64] row snapshot
    float* sg = srow + 64u;                    // [64] cumulative gates
    unsigned long long b = blockIdx.x;
    unsigned int r = threadIdx.x;
    const float* kb = k + b * 64ull * dk;
    const float* bg = beta_g + b * 64ull * 2ull;
    float beta_r = bg[r * 2u];
    sg[r] = bg[r * 2u + 1u];
    for (unsigned int i = 0; i < dk; ++i) {
        sk[r * dk + i] = kb[r * dk + i] * beta_r;
    }
    __syncthreads();
    // A0[r][c] = -(k_beta[r] . k[c]) * exp(g[r] - g[c]) strictly
    // below the diagonal; zero on and above.
    for (unsigned int c = 0; c < 64u; ++c) {
        float value = 0.f;
        if (c < r) {
            float dot = 0.f;
            for (unsigned int i = 0; i < dk; ++i) {
                dot += sk[r * dk + i] * kb[c * dk + i];
            }
            value = -(dot * expf(sg[r] - sg[c]));
        }
        sa[r * 64u + c] = value;
    }
    __syncthreads();
    // Forward substitution, rows ascending (the torch loop): snapshot
    // row r_iter's prefix, then column-thread c accumulates
    // row[c] += sum_i prefix[i] * sa[i][c] over the updated block.
    for (unsigned int r_iter = 1; r_iter < 64u; ++r_iter) {
        if (r < r_iter) {
            srow[r] = sa[r_iter * 64u + r];
        }
        __syncthreads();
        if (r < r_iter) {
            float value = srow[r];
            for (unsigned int i = 0; i < r_iter; ++i) {
                value += srow[i] * sa[i * 64u + r];
            }
            sa[r_iter * 64u + r] = value;
        }
        __syncthreads();
    }
    // + I, then the row writes back.
    float* out_row = a_out + (b * 64ull + r) * 64ull;
    for (unsigned int c = 0; c < 64u; ++c) {
        float value = sa[r * 64u + c];
        if (c == r) {
            value += 1.0f;
        }
        out_row[c] = value;
    }
}

// StateAdvance v2: the inter-chunk serial recurrence with the chunk
// outputs fused (fla fwd_h + fwd_o folded; no HBM state scratch).
// OCCUPANCY SPLIT: grid (heads, 4 stripes) = 120 blocks, 48 threads
// each - one thread per v-column of its stripe; the shared staging
// (k tile, gates, decayed attn tile) is duplicated per stripe and
// the per-column math is IDENTICAL to the single-block form, so the
// split is value-transparent. Compile-time tiles (SA_DK/SA_DV) keep
// the state column in registers - runtime-bound register arrays
// spill to local memory and serialize every MAC (the measured v1
// failure).
#define SA_DK 96u
#define SA_DV 192u
#define SA_COLS 48u
#define SA_ROW (3u * SA_DK + SA_DV + 1u)

extern "C" __global__ void state_advance_f32(
    float* state,
    const float* bundle,
    float* out,
    unsigned int chunks
) {
    extern __shared__ float shared[];
    float* sk = shared;                        // [64, SA_DK] raw k
    float* sattn = shared + 64u * SA_DK;       // [64, 64] decayed local attn
    float* sg = sattn + 64u * 64u;             // [64] cumulative gates
    float* seg = sg + 64u;                     // [64] exp(g)
    float* ses = seg + 64u;                    // [64] exp(g_last - g)
    unsigned long long h = blockIdx.x;
    unsigned int stripe = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned int j = stripe * SA_COLS + tid;   // this thread's v-column
    const unsigned long long chunk_stride = 64ull * SA_ROW;
    const float* bundle_h = bundle + h * (unsigned long long)chunks * chunk_stride;
    float* out_h = out + h * (unsigned long long)chunks * 64ull * SA_DV;
    float* state_h = state + h * (unsigned long long)SA_DK * SA_DV;

    float s_reg[SA_DK];
    float vnew[64];
    #pragma unroll
    for (unsigned int i = 0; i < SA_DK; ++i) {
        s_reg[i] = state_h[(unsigned long long)i * SA_DV + j];
    }

    for (unsigned int chunk = 0; chunk < chunks; ++chunk) {
        const float* rows = bundle_h + chunk * chunk_stride;
        __syncthreads();
        // Stage k + gates (strided over the stripe's threads).
        for (unsigned int idx = tid; idx < 64u * SA_DK; idx += SA_COLS) {
            unsigned int r = idx / SA_DK;
            unsigned int i = idx - r * SA_DK;
            sk[r * SA_DK + i] = rows[r * SA_ROW + SA_DK + i];
        }
        for (unsigned int r = tid; r < 64u; r += SA_COLS) {
            sg[r] = rows[r * SA_ROW + 3u * SA_DK + SA_DV];
        }
        __syncthreads();
        for (unsigned int r = tid; r < 64u; r += SA_COLS) {
            seg[r] = expf(sg[r]);
            ses[r] = expf(sg[63] - sg[r]);
        }
        __syncthreads();
        // The decayed local attention tile: (q[r] . k[c]) *
        // exp(g[r] - g[c]) on and below the diagonal, zero above.
        for (unsigned int idx = tid; idx < 64u * 64u; idx += SA_COLS) {
            unsigned int r = idx / 64u;
            unsigned int c = idx - r * 64u;
            float value = 0.f;
            if (c <= r) {
                const float* q_row = rows + r * SA_ROW;
                float dot = 0.f;
                #pragma unroll
                for (unsigned int i = 0; i < SA_DK; ++i) {
                    dot += q_row[i] * sk[c * SA_DK + i];
                }
                value = dot * expf(sg[r] - sg[c]);
            }
            sattn[idx] = value;
        }
        __syncthreads();
        // v_new = u - w @ S over the carried column.
        for (unsigned int r = 0; r < 64u; ++r) {
            const float* row = rows + r * SA_ROW;
            const float* w_row = row + 2u * SA_DK;
            float acc = 0.f;
            #pragma unroll
            for (unsigned int i = 0; i < SA_DK; ++i) {
                acc += w_row[i] * s_reg[i];
            }
            vnew[r] = row[3u * SA_DK + j] - acc;
        }
        // out rows: exp(g_r) * (q[r] . S) + attn_local @ v_new.
        for (unsigned int r = 0; r < 64u; ++r) {
            const float* q_row = rows + r * SA_ROW;
            float inter = 0.f;
            #pragma unroll
            for (unsigned int i = 0; i < SA_DK; ++i) {
                inter += q_row[i] * s_reg[i];
            }
            float acc = inter * seg[r];
            const float* attn_row = sattn + r * 64u;
            for (unsigned int c = 0; c <= r; ++c) {
                acc += attn_row[c] * vnew[c];
            }
            out_h[(chunk * 64ull + r) * SA_DV + j] = acc;
        }
        // S = S * exp(g_last) + (k * exp(g_last - g))^T @ v_new -
        // r-major so each vnew row is read ONCE (a runtime-indexed
        // register array lives in local memory; the i-major form paid
        // 96 local reads per row and dominated the kernel).
        // Reassociation-class vs the i-major sum; the lock re-pins.
        float e_last = expf(sg[63]);
        #pragma unroll
        for (unsigned int i = 0; i < SA_DK; ++i) {
            s_reg[i] *= e_last;
        }
        for (unsigned int r = 0; r < 64u; ++r) {
            float scaled = ses[r] * vnew[r];
            const float* k_row = sk + r * SA_DK;
            #pragma unroll
            for (unsigned int i = 0; i < SA_DK; ++i) {
                s_reg[i] += k_row[i] * scaled;
            }
        }
    }
    #pragma unroll
    for (unsigned int i = 0; i < SA_DK; ++i) {
        state_h[(unsigned long long)i * SA_DV + j] = s_reg[i];
    }
}
"#;

/// Lazily nvrtc-compiled module, one per process (single-device box).
static MODULE: std::sync::OnceLock<std::sync::Arc<cudarc::driver::CudaModule>> =
    std::sync::OnceLock::new();

fn kernel(
    device: &candle_core::CudaDevice,
    name: &str,
) -> LibAlmostResult<cudarc::driver::CudaFunction> {
    let module = match MODULE.get() {
        Some(module) => module.clone(),
        None => {
            let ptx = match cudarc::nvrtc::compile_ptx(KERNEL_SRC) {
                Ok(ptx) => ptx,
                Err(error) => {
                    snafu::whatever!("nvrtc compile of tri_solve failed: {error}")
                }
            };
            let module = match device.cuda_stream().context().load_module(ptx) {
                Ok(module) => module,
                Err(error) => {
                    snafu::whatever!("loading the tri_solve module failed: {error}")
                }
            };
            MODULE.get_or_init(|| module).clone()
        }
    };
    match module.load_function(name) {
        Ok(function) => Ok(function),
        Err(error) => snafu::whatever!("loading kernel {name} failed: {error}"),
    }
}

/// `a.inplace_op3(&k, &beta_g, &TriSolve)`: a (blocks, 64, 64) f32 -
/// WRITTEN by the kernel (the inverted-and-identity-added intra-chunk
/// matrix); k (blocks, 64, dk) f32 raw keys; beta_g (blocks, 64, 2)
/// f32 rows beta | cumulative g. All contiguous, all addressed at
/// their layout offsets.
pub(crate) struct TriSolve;

impl candle_core::InplaceOp3 for TriSolve {
    fn name(&self) -> &'static str {
        "almost_tri_solve"
    }

    fn cpu_fwd(
        &self,
        _s1: &mut candle_core::CpuStorage,
        _l1: &candle_core::Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &candle_core::Layout,
        _s3: &candle_core::CpuStorage,
        _l3: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        candle_core::bail!("tri_solve is a cuda prefill building block")
    }

    fn cuda_fwd(
        &self,
        a: &mut candle_core::CudaStorage,
        a_layout: &candle_core::Layout,
        k: &candle_core::CudaStorage,
        k_layout: &candle_core::Layout,
        beta_g: &candle_core::CudaStorage,
        beta_g_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use cudarc::driver::{DevicePtr, PushKernelArg};

        let (blocks, chunk, chunk_2) = a_layout.shape().dims3()?;
        let (k_blocks, k_chunk, dk) = k_layout.shape().dims3()?;
        let bg_dims = beta_g_layout.shape().dims3()?;
        if chunk != 64 || chunk_2 != 64 || k_blocks != blocks || k_chunk != 64
            || bg_dims != (blocks, 64, 2)
        {
            candle_core::bail!(
                "tri_solve wants a (blocks, 64, 64), k (blocks, 64, dk), beta_g (blocks, 64, 2); \
                 got {:?}, {:?}, {bg_dims:?}",
                (blocks, chunk, chunk_2),
                (k_blocks, k_chunk, dk)
            );
        }
        if !a_layout.is_contiguous()
            || !k_layout.is_contiguous()
            || !beta_g_layout.is_contiguous()
        {
            candle_core::bail!("tri_solve wants contiguous operands");
        }

        let device = a.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, "tri_solve_f32") {
            Ok(function) => function,
            Err(error) => candle_core::bail!("{error}"),
        };
        let a_slice = a.as_cuda_slice::<f32>()?.slice(a_layout.start_offset()..);
        let (a_ptr, _a_guard) = a_slice.device_ptr(&stream);
        let k_slice = k.as_cuda_slice::<f32>()?.slice(k_layout.start_offset()..);
        let (k_ptr, _k_guard) = k_slice.device_ptr(&stream);
        let bg_slice = beta_g
            .as_cuda_slice::<f32>()?
            .slice(beta_g_layout.start_offset()..);
        let (bg_ptr, _bg_guard) = bg_slice.device_ptr(&stream);

        let dk = dk as u32;
        let shared_bytes = (64 * dk + 64 * 64 + 64 + 64) * 4;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        let mut builder = stream.launch_builder(&function);
        builder.arg(&a_ptr).arg(&k_ptr).arg(&bg_ptr).arg(&dk);
        if let Err(error) = unsafe { builder.launch(config) } {
            candle_core::bail!("tri_solve launch failed: {error}");
        }
        Ok(())
    }
}

/// `state.inplace_op3(&bundle, &out, &StateAdvance)`: state
/// (heads, dk, dv) f32 - the carried state, advanced IN PLACE across
/// every chunk (the kernel loops chunks serially inside); bundle
/// (heads, chunks, 64, 3*dk + dv + 1) f32 rows q | k | w | u | g;
/// out (heads, chunks, 64, dv) f32 - WRITTEN by the kernel (the
/// mixer outputs; candle tensors mutate through shared storage by
/// design). All contiguous, all addressed at their layout offsets.
/// Geometry-locked to dk 96 / dv 192 (compile-time tiles keep the
/// state column in registers); the grid splits each head's
/// v-columns across four 48-column stripes for occupancy (120
/// blocks; the shared staging is duplicated per stripe, the
/// per-column math is identical). The bundle pack STAYS this round:
/// the serial kernel needs q/k/w/u/g + state + out, InplaceOp3 caps
/// at three operands, and candle matmuls cannot land in slices -
/// measured pack cost ~2% of prefill wall.
pub(crate) struct StateAdvance;

impl candle_core::InplaceOp3 for StateAdvance {
    fn name(&self) -> &'static str {
        "almost_state_advance"
    }

    fn cpu_fwd(
        &self,
        _s1: &mut candle_core::CpuStorage,
        _l1: &candle_core::Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &candle_core::Layout,
        _s3: &candle_core::CpuStorage,
        _l3: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        candle_core::bail!("state_advance is a cuda prefill building block")
    }

    fn cuda_fwd(
        &self,
        state: &mut candle_core::CudaStorage,
        state_layout: &candle_core::Layout,
        bundle: &candle_core::CudaStorage,
        bundle_layout: &candle_core::Layout,
        out: &candle_core::CudaStorage,
        out_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use cudarc::driver::{DevicePtr, PushKernelArg};

        let (heads, dk, dv) = state_layout.shape().dims3()?;
        let (b_heads, chunks, b_chunk, row_width) = bundle_layout.shape().dims4()?;
        let out_dims = out_layout.shape().dims4()?;
        if b_heads != heads
            || b_chunk != 64
            || row_width != 3 * dk + dv + 1
            || out_dims != (heads, chunks, 64, dv)
        {
            candle_core::bail!(
                "state_advance wants state (h, dk, dv), bundle (h, n, 64, 3dk+dv+1), \
                 out (h, n, 64, dv); got {:?}, {:?}, {out_dims:?}",
                (heads, dk, dv),
                (b_heads, chunks, b_chunk, row_width)
            );
        }
        if dk != 96 || dv != 192 {
            candle_core::bail!(
                "state_advance is geometry-locked to dk == 96 and dv == 192 (got {dk}, {dv})"
            );
        }
        if !state_layout.is_contiguous()
            || !bundle_layout.is_contiguous()
            || !out_layout.is_contiguous()
        {
            candle_core::bail!("state_advance wants contiguous operands");
        }

        let device = state.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, "state_advance_f32") {
            Ok(function) => function,
            Err(error) => candle_core::bail!("{error}"),
        };
        let state_slice = state
            .as_cuda_slice::<f32>()?
            .slice(state_layout.start_offset()..);
        let (state_ptr, _state_guard) = state_slice.device_ptr(&stream);
        let bundle_slice = bundle
            .as_cuda_slice::<f32>()?
            .slice(bundle_layout.start_offset()..);
        let (bundle_ptr, _bundle_guard) = bundle_slice.device_ptr(&stream);
        let out_slice = out.as_cuda_slice::<f32>()?.slice(out_layout.start_offset()..);
        let (out_ptr, _out_guard) = out_slice.device_ptr(&stream);

        let chunks = chunks as u32;
        let shared_bytes: u32 = (64 * 96 + 64 * 64 + 3 * 64) * 4;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (heads as u32, 4, 1),
            block_dim: (48, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&state_ptr)
            .arg(&bundle_ptr)
            .arg(&out_ptr)
            .arg(&chunks);
        if let Err(error) = unsafe { builder.launch(config) } {
            candle_core::bail!("state_advance launch failed: {error}");
        }
        Ok(())
    }
}

/// The chunked gated-delta rule as the two-kernel family: TriSolve in
/// place of the classic mask-build + kkt + row-serial inversion, and
/// StateAdvance in place of the per-chunk sequential loop (outputs
/// fused; no decay-mask tensors at all). The same contract as
/// gdn::chunk_rule (pre-l2normed q/k, q pre-scaled, all f32,
/// initial-state carry; pad rows inert).
pub(crate) fn chunk_rule_fused(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    g: &candle_core::Tensor,
    beta: &candle_core::Tensor,
    state: &candle_core::Tensor,
) -> LibAlmostResult<(candle_core::Tensor, candle_core::Tensor)> {
    let (t, heads, head_k) = q.dims3()?;
    let head_v = v.dim(2)?;
    let device = q.device();
    let pad = (gdn::CHUNK - t % gdn::CHUNK) % gdn::CHUNK;
    let chunks = (t + pad) / gdn::CHUNK;

    let q = q.transpose(0, 1)?.pad_with_zeros(1, 0, pad)?;
    let k = k.transpose(0, 1)?.pad_with_zeros(1, 0, pad)?;
    let v = v.transpose(0, 1)?.pad_with_zeros(1, 0, pad)?;
    let g = g.transpose(0, 1)?.pad_with_zeros(1, 0, pad)?;
    let beta = beta.transpose(0, 1)?.pad_with_zeros(1, 0, pad)?;

    let v_beta = v.broadcast_mul(&beta.unsqueeze(2)?)?;
    let k_beta = k.broadcast_mul(&beta.unsqueeze(2)?)?;

    let q = q.reshape((heads, chunks, gdn::CHUNK, head_k))?;
    let k = k.reshape((heads, chunks, gdn::CHUNK, head_k))?;
    let v_beta = v_beta.reshape((heads, chunks, gdn::CHUNK, head_v))?;
    let k_beta = k_beta.reshape((heads, chunks, gdn::CHUNK, head_k))?;
    let g = g.reshape((heads, chunks, gdn::CHUNK))?.cumsum(2)?;

    // TriSolve: the whole intra-chunk build + inversion in one launch.
    let blocks = heads * chunks;
    let k_blocks = k.reshape((blocks, gdn::CHUNK, head_k))?.contiguous()?;
    let beta_column = beta.reshape((blocks, gdn::CHUNK, 1))?;
    let g_column = g.reshape((blocks, gdn::CHUNK, 1))?;
    // cat on a non-zero dim returns a transposed view (the era-one
    // contiguity lesson); the kernel wants packed rows.
    let beta_g = candle_core::Tensor::cat(&[&beta_column, &g_column], 2)?.contiguous()?;
    let attn = candle_core::Tensor::zeros(
        (blocks, gdn::CHUNK, gdn::CHUNK),
        candle_core::DType::F32,
        device,
    )?;
    attn.inplace_op3(&k_blocks, &beta_g, &TriSolve)?;
    let attn = attn.reshape((heads, chunks, gdn::CHUNK, gdn::CHUNK))?;

    let value = attn.matmul(&v_beta)?;
    let g_exp = g.exp()?.unsqueeze(3)?;
    let k_cumdecay = attn.matmul(&k_beta.broadcast_mul(&g_exp)?.contiguous()?)?;

    // StateAdvance: the serial inter-chunk recurrence + the chunk
    // outputs in one launch. cat on a non-zero dim returns a
    // transposed view (the era-one contiguity lesson); the kernel
    // wants packed rows.
    let bundle = candle_core::Tensor::cat(
        &[&q, &k, &k_cumdecay, &value, &g.unsqueeze(3)?],
        3,
    )?
    .contiguous()?;
    let carried = state.copy()?;
    let out = candle_core::Tensor::zeros(
        (heads, chunks, gdn::CHUNK, head_v),
        candle_core::DType::F32,
        device,
    )?;
    carried.inplace_op3(&bundle, &out, &StateAdvance)?;
    let out = out
        .reshape((heads, chunks * gdn::CHUNK, head_v))?
        .narrow(1, 0, t)?
        .transpose(0, 1)?
        .contiguous()?;
    Ok((out, carried))
}
