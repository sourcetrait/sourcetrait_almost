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

/// The chunked gated-delta rule with the TriSolve kernel in place of
/// the classic mask-build + kkt + row-serial inversion: the same
/// contract as gdn::chunk_rule (pre-l2normed q/k, q pre-scaled, all
/// f32, initial-state carry; pad rows inert) - the per-chunk state
/// advance stays the classic candle loop until the StateAdvance
/// slice lands.
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

    let lower_incl = gdn::lower_triangle(gdn::CHUNK, false, device)?;

    // The state loop still consumes the decay mask (StateAdvance
    // retires it); the solve no longer does.
    let diff = g.unsqueeze(3)?.broadcast_sub(&g.unsqueeze(2)?)?;
    let decay_mask = diff
        .broadcast_mul(&lower_incl)?
        .exp()?
        .broadcast_mul(&lower_incl)?;

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

    // Per-chunk sequential state advance - the classic loop verbatim
    // (the StateAdvance slice replaces it).
    let mut carried = state.clone();
    let mut outs = Vec::with_capacity(chunks);
    for chunk_idx in 0..chunks {
        let q_i = q.narrow(1, chunk_idx, 1)?.squeeze(1)?.contiguous()?;
        let k_i = k.narrow(1, chunk_idx, 1)?.squeeze(1)?.contiguous()?;
        let v_i = value.narrow(1, chunk_idx, 1)?.squeeze(1)?;
        let g_i = g.narrow(1, chunk_idx, 1)?.squeeze(1)?;
        let mask_i = decay_mask.narrow(1, chunk_idx, 1)?.squeeze(1)?;
        let attn_local = q_i
            .matmul(&k_i.transpose(1, 2)?.contiguous()?)?
            .mul(&mask_i)?
            .broadcast_mul(&lower_incl)?;
        let v_prime = k_cumdecay
            .narrow(1, chunk_idx, 1)?
            .squeeze(1)?
            .contiguous()?
            .matmul(&carried)?;
        let v_new = v_i.sub(&v_prime)?;
        let g_exp_i = g_i.exp()?.unsqueeze(2)?;
        let attn_inter = q_i.broadcast_mul(&g_exp_i)?.matmul(&carried)?;
        outs.push((attn_inter + attn_local.matmul(&v_new)?)?);
        let g_last = g_i.narrow(1, gdn::CHUNK - 1, 1)?;
        let state_decay = g_last.exp()?.unsqueeze(2)?;
        let k_scale = g_last.broadcast_sub(&g_i)?.exp()?.unsqueeze(2)?;
        carried = (carried.broadcast_mul(&state_decay)?
            + k_i
                .broadcast_mul(&k_scale)?
                .transpose(1, 2)?
                .contiguous()?
                .matmul(&v_new)?)?;
    }
    let out = candle_core::Tensor::cat(&outs, 1)?
        .narrow(1, 0, t)?
        .transpose(0, 1)?
        .contiguous()?;
    Ok((out, carried))
}
