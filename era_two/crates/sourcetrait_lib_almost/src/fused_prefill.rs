//! PrefillDispatch building blocks: the fused chunked-GDN prefill
//! family. PrepChunk collapses the per-chunk prep chain (the carried
//! causal conv + silu, the q/k l2 norms + scale, and the gating
//! scalars with the per-chunk decay cumsum) into ONE launch per
//! layer-call, writing q | k | g | beta into the StateAdvance bundle
//! DIRECTLY (PackTrim: no transpose/pad/cat pipeline exists at all);
//! TriSolve folds the intra-chunk decay-mask build, beta-scaled kkt
//! matmul, and the blocked-substitution triangular inversion into one
//! launch reading the same bundle; StateAdvance runs the serial
//! inter-chunk recurrence with the chunk outputs fused. Raw pointer
//! args on the model's own stream, operands addressed at their layout
//! offsets (the offset-leak lock).
//!
//! Numerics: the prep kernel runs the conv/silu/l2 chain in f32 where
//! the classic chain rounds through bf16 per op (the FusedDecodeConv
//! precedent - envelope-class, toward truth; the prep lock pins it);
//! the rule kernels run the same f32 math as gdn::chunk_rule with
//! reassociation-class deltas (the rule lock pins those). The cpu
//! path, the stateless parity form, and decode never dispatch here;
//! only the CARRIED cuda bf16 multi-token chunk branch does.
use crate::*;

/// Kernel geometry: one 64-row chunk tile per block. SA_ROW carries
/// q | k | w | u | g | beta (beta rides for TriSolve/PrepKbg;
/// StateAdvance ignores it). PC_KH/PC_VH are the GDN head dims the
/// family is geometry-locked to (dk 96 / dv 192).
const KERNEL_SRC: &str = r#"
#define SA_DK 96u
#define SA_DV 192u
#define SA_COLS 48u
#define SA_ROW (3u * SA_DK + SA_DV + 2u)
#define PC_TAPS 4u

__device__ __forceinline__ float bf16_to_f32(unsigned short bits) {
    unsigned int wide = ((unsigned int)bits) << 16;
    return __uint_as_float(wide);
}

// The intra-chunk solve: A0 = -(k_beta k^T . decay) strictly below
// the diagonal, then the blocked forward substitution, then + I. One
// block per (head, chunk) tile, 64 threads; k, beta, and the
// cumulative gates read from the bundle rows.
extern "C" __global__ void tri_solve_f32(
    float* a_out,
    const float* bundle
) {
    extern __shared__ float shared[];
    float* sk = shared;                        // [64, SA_DK] beta-scaled k
    float* sa = shared + 64u * SA_DK;          // [64, 64] working matrix
    float* srow = sa + 64u * 64u;              // [64] row snapshot
    float* sg = srow + 64u;                    // [64] cumulative gates
    unsigned long long b = blockIdx.x;
    unsigned int r = threadIdx.x;
    const float* rows = bundle + b * 64ull * SA_ROW;
    float beta_r = rows[r * SA_ROW + SA_ROW - 1u];
    sg[r] = rows[r * SA_ROW + 3u * SA_DK + SA_DV];
    for (unsigned int i = 0; i < SA_DK; ++i) {
        sk[r * SA_DK + i] = rows[r * SA_ROW + SA_DK + i] * beta_r;
    }
    __syncthreads();
    // A0[r][c] = -(k_beta[r] . k[c]) * exp(g[r] - g[c]) strictly
    // below the diagonal; zero on and above.
    for (unsigned int c = 0; c < 64u; ++c) {
        float value = 0.f;
        if (c < r) {
            float dot = 0.f;
            for (unsigned int i = 0; i < SA_DK; ++i) {
                dot += sk[r * SA_DK + i] * rows[c * SA_ROW + SA_DK + i];
            }
            value = -(dot * expf(sg[r] - sg[c]));
        }
        sa[r * 64u + c] = value;
    }
    __syncthreads();
    // Blocked forward substitution (BlockedSubstitution): the four
    // 16x16 diagonal blocks solve concurrently (independent regions,
    // the row-serial snapshot shape per block), then each block row's
    // off-diagonal tiles update by shared-memory block products -
    // S = A[bi][bj] + sum(k in [bj, bi)) A[bi][k] . M[k][bj], then
    // M[bi][bj] = S + M[bi][bi] . S (the in-block inverse applied
    // last). The same bilinear form as the row-serial loop with a
    // regrouped accumulation - reassociation-class; the lock re-pins.
    // The freed sk region stages the S tiles (dead after the A0
    // build); in-block upper triangles stay 0.f, so full 16-sums add
    // exact zeros.
    unsigned int db = r >> 4u;
    unsigned int dc = r & 15u;
    for (unsigned int s = 1; s < 16u; ++s) {
        if (dc < s) {
            srow[r] = sa[(db * 16u + s) * 64u + db * 16u + dc];
        }
        __syncthreads();
        if (dc < s) {
            float value = srow[r];
            for (unsigned int i = 0; i < s; ++i) {
                value += srow[db * 16u + i] * sa[(db * 16u + i) * 64u + db * 16u + dc];
            }
            sa[(db * 16u + s) * 64u + db * 16u + dc] = value;
        }
        __syncthreads();
    }
    float* stile = sk;
    for (unsigned int bi = 1; bi < 4u; ++bi) {
        for (unsigned int e = r; e < bi * 256u; e += 64u) {
            unsigned int bj = e >> 8u;
            unsigned int ti = e & 255u;
            unsigned int er = ti >> 4u;
            unsigned int ec = ti & 15u;
            unsigned int row_g = bi * 16u + er;
            float s_val = sa[row_g * 64u + bj * 16u + ec];
            for (unsigned int k = bj; k < bi; ++k) {
                for (unsigned int i = 0; i < 16u; ++i) {
                    s_val += sa[row_g * 64u + k * 16u + i]
                        * sa[(k * 16u + i) * 64u + bj * 16u + ec];
                }
            }
            stile[e] = s_val;
        }
        __syncthreads();
        for (unsigned int e = r; e < bi * 256u; e += 64u) {
            unsigned int bj = e >> 8u;
            unsigned int ti = e & 255u;
            unsigned int er = ti >> 4u;
            unsigned int ec = ti & 15u;
            unsigned int row_g = bi * 16u + er;
            float t_val = stile[e];
            for (unsigned int i = 0; i < 16u; ++i) {
                t_val += sa[row_g * 64u + bi * 16u + i]
                    * stile[bj * 256u + i * 16u + ec];
            }
            sa[row_g * 64u + bj * 16u + ec] = t_val;
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

// PrepChunk: the per-chunk prep chain in one launch - the carried
// causal conv + silu in f32 (the classic chain rounds through bf16
// per op), the q/k l2 norms + q scale, the gating scalars, and the
// per-chunk decay cumsum (thread 0 serial - the candle cumsum's
// association) - writing q | k | g | beta bundle-direct plus the
// v_beta gemm operand. dyn rows: 0..3 conv-weight taps (A_log/dt in
// the row-0/row-0 pad columns), 4..6 the carried tail (bf16), 7..
// the token rows (conv_in | a | b). Pad rows past t ride the scan as
// zeros (sg[63] keeps the last real cumsum) and write only g.
extern "C" __global__ void prep_chunk_f32(
    float* bundle,
    const unsigned short* dyn_rows,
    float* v_beta,
    unsigned int t,
    unsigned int heads,
    unsigned int chunks,
    float q_scale,
    float beta_scale
) {
    __shared__ float sg[64];
    unsigned int h = blockIdx.x;
    unsigned int chunk = blockIdx.y;
    unsigned int r = threadIdx.x;
    unsigned int tok = chunk * 64u + r;
    unsigned int cw = 2u * heads * SA_DK + heads * SA_DV;
    unsigned int dw = cw + 2u * heads;
    float g_raw = 0.f;
    float beta = 0.f;
    if (tok < t) {
        float a_v = bf16_to_f32(dyn_rows[(7u + tok) * dw + cw + h]);
        float b_v = bf16_to_f32(dyn_rows[(7u + tok) * dw + cw + heads + h]);
        float a_log = bf16_to_f32(dyn_rows[cw + h]);
        float dt = bf16_to_f32(dyn_rows[cw + heads + h]);
        float x = a_v + dt;
        float sp = x > 20.f ? x : logf(1.f + expf(x));
        g_raw = -(expf(a_log) * sp);
        beta = beta_scale / (1.f + expf(-b_v));
    }
    sg[r] = g_raw;
    __syncthreads();
    if (r == 0u) {
        for (unsigned int i = 1; i < 64u; ++i) {
            sg[i] += sg[i - 1u];
        }
    }
    __syncthreads();
    unsigned long long brow =
        ((unsigned long long)(h * chunks + chunk) * 64u + r) * SA_ROW;
    bundle[brow + 3u * SA_DK + SA_DV] = sg[r];
    if (tok >= t) { return; }
    bundle[brow + SA_ROW - 1u] = beta;
    unsigned int hist = 4u + tok;
    unsigned int q_base = h * SA_DK;
    unsigned int k_base = heads * SA_DK + h * SA_DK;
    unsigned int v_base = 2u * heads * SA_DK + h * SA_DV;
    // q/k: two passes (sumsq, then normalize + write) with the conv
    // recomputed - no 96-wide register arrays to spill.
    float q_sumsq = 0.f;
    float k_sumsq = 0.f;
    for (unsigned int i = 0; i < SA_DK; ++i) {
        float q_acc = 0.f;
        float k_acc = 0.f;
        for (unsigned int tap = 0; tap < PC_TAPS; ++tap) {
            q_acc += bf16_to_f32(dyn_rows[(hist + tap) * dw + q_base + i])
                * bf16_to_f32(dyn_rows[tap * dw + q_base + i]);
            k_acc += bf16_to_f32(dyn_rows[(hist + tap) * dw + k_base + i])
                * bf16_to_f32(dyn_rows[tap * dw + k_base + i]);
        }
        float q_s = q_acc / (1.f + expf(-q_acc));
        float k_s = k_acc / (1.f + expf(-k_acc));
        q_sumsq += q_s * q_s;
        k_sumsq += k_s * k_s;
    }
    float q_inv = 1.f / sqrtf(q_sumsq + 1e-6f);
    float k_inv = 1.f / sqrtf(k_sumsq + 1e-6f);
    for (unsigned int i = 0; i < SA_DK; ++i) {
        float q_acc = 0.f;
        float k_acc = 0.f;
        for (unsigned int tap = 0; tap < PC_TAPS; ++tap) {
            q_acc += bf16_to_f32(dyn_rows[(hist + tap) * dw + q_base + i])
                * bf16_to_f32(dyn_rows[tap * dw + q_base + i]);
            k_acc += bf16_to_f32(dyn_rows[(hist + tap) * dw + k_base + i])
                * bf16_to_f32(dyn_rows[tap * dw + k_base + i]);
        }
        float q_s = q_acc / (1.f + expf(-q_acc));
        float k_s = k_acc / (1.f + expf(-k_acc));
        bundle[brow + i] = q_s * q_inv * q_scale;
        bundle[brow + SA_DK + i] = k_s * k_inv;
    }
    unsigned long long vrow =
        ((unsigned long long)(h * chunks + chunk) * 64u + r) * SA_DV;
    for (unsigned int i = 0; i < SA_DV; ++i) {
        float acc = 0.f;
        for (unsigned int tap = 0; tap < PC_TAPS; ++tap) {
            acc += bf16_to_f32(dyn_rows[(hist + tap) * dw + v_base + i])
                * bf16_to_f32(dyn_rows[tap * dw + v_base + i]);
        }
        float v_s = acc / (1.f + expf(-acc));
        v_beta[vrow + i] = v_s * beta;
    }
}

// PrepKbg: the k_cumdecay gemm operand k * beta * exp(g_cum), read
// straight off the bundle ((k * beta) * exp - the classic mul order).
extern "C" __global__ void prep_kbg_f32(
    float* kbg,
    const float* bundle
) {
    unsigned long long row = (unsigned long long)blockIdx.x * 64u + threadIdx.x;
    const float* src = bundle + row * SA_ROW;
    float beta_v = src[SA_ROW - 1u];
    float g_exp = expf(src[3u * SA_DK + SA_DV]);
    float* dst = kbg + row * SA_DK;
    for (unsigned int i = 0; i < SA_DK; ++i) {
        dst[i] = (src[SA_DK + i] * beta_v) * g_exp;
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
        // v_new = u - w @ S over the carried column. Unrolled so every
        // vnew index is compile-time - the register-residency contract
        // shared by all three vnew loops (VnewUnroll).
        #pragma unroll
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
            // Full-width unroll: sattn's upper triangle is 0.f by
            // construction, so c > r adds exact zeros - and every vnew
            // index becomes compile-time -> registers (VnewUnroll; the
            // runtime c <= r bound kept vnew in local memory and its
            // triangular reads dominated the residual).
            const float* attn_row = sattn + r * 64u;
            #pragma unroll
            for (unsigned int c = 0; c < 64u; ++c) {
                acc += attn_row[c] * vnew[c];
            }
            out_h[(chunk * 64ull + r) * SA_DV + j] = acc;
        }
        // S = S * exp(g_last) + (k * exp(g_last - g))^T @ v_new -
        // r-major (each vnew row read once) and unrolled so the vnew
        // access stays compile-time-indexed (the VnewUnroll register
        // contract). Reassociation-class vs the i-major sum; the lock
        // re-pins.
        float e_last = expf(sg[63]);
        #pragma unroll
        for (unsigned int i = 0; i < SA_DK; ++i) {
            s_reg[i] *= e_last;
        }
        #pragma unroll
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

/// The bundle row width (q | k | w | u | g | beta) the kernels are
/// compiled against.
pub(crate) const BUNDLE_ROW: usize = 3 * 96 + 192 + 2;

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
                    snafu::whatever!("nvrtc compile of the prefill kernels failed: {error}")
                }
            };
            let module = match device.cuda_stream().context().load_module(ptx) {
                Ok(module) => module,
                Err(error) => {
                    snafu::whatever!("loading the prefill-kernel module failed: {error}")
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

/// `a.inplace_op2(&bundle, &TriSolve)`: a (blocks, 64, 64) f32 -
/// WRITTEN by the kernel (the inverted-and-identity-added intra-chunk
/// matrix); bundle (heads, chunks, 64, BUNDLE_ROW) f32 with blocks ==
/// heads * chunks (k, cumulative g, and beta read at their bundle
/// columns). All contiguous, all addressed at their layout offsets.
pub(crate) struct TriSolve;

impl candle_core::InplaceOp2 for TriSolve {
    fn name(&self) -> &'static str {
        "almost_tri_solve"
    }

    fn cpu_fwd(
        &self,
        _s1: &mut candle_core::CpuStorage,
        _l1: &candle_core::Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        candle_core::bail!("tri_solve is a cuda prefill building block")
    }

    fn cuda_fwd(
        &self,
        a: &mut candle_core::CudaStorage,
        a_layout: &candle_core::Layout,
        bundle: &candle_core::CudaStorage,
        bundle_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use cudarc::driver::{DevicePtr, PushKernelArg};

        let (blocks, chunk, chunk_2) = a_layout.shape().dims3()?;
        let (b_heads, b_chunks, b_chunk, row_width) = bundle_layout.shape().dims4()?;
        if chunk != 64
            || chunk_2 != 64
            || b_chunk != 64
            || row_width != BUNDLE_ROW
            || blocks != b_heads * b_chunks
        {
            candle_core::bail!(
                "tri_solve wants a (h*n, 64, 64) and bundle (h, n, 64, {BUNDLE_ROW}); \
                 got {:?}, {:?}",
                (blocks, chunk, chunk_2),
                (b_heads, b_chunks, b_chunk, row_width)
            );
        }
        if !a_layout.is_contiguous() || !bundle_layout.is_contiguous() {
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
        let bundle_slice = bundle
            .as_cuda_slice::<f32>()?
            .slice(bundle_layout.start_offset()..);
        let (bundle_ptr, _bundle_guard) = bundle_slice.device_ptr(&stream);

        let shared_bytes = ((64 * 96 + 64 * 64 + 64 + 64) * 4) as u32;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        let mut builder = stream.launch_builder(&function);
        builder.arg(&a_ptr).arg(&bundle_ptr);
        if let Err(error) = unsafe { builder.launch(config) } {
            candle_core::bail!("tri_solve launch failed: {error}");
        }
        Ok(())
    }
}

/// `bundle.inplace_op3(&dyn_rows, &v_beta, &PrepChunk { .. })`:
/// bundle (heads, chunks, 64, BUNDLE_ROW) f32 zeros - q | k | g |
/// beta WRITTEN by the kernel (w | u stay zero for the gemm
/// slice_sets); dyn_rows (tokens + 7, conv_width + 2 * heads) bf16 -
/// rows 0..3 the conv-weight taps (A_log | dt_bias in the row-0 pad
/// columns), 4..6 the carried tail, 7.. the token rows
/// (conv_in | a | b); v_beta (heads, chunks, 64, 192) f32 - WRITTEN
/// (candle tensors mutate through shared storage by design).
/// Geometry-locked to dk 96 / dv 192.
pub(crate) struct PrepChunk {
    pub(crate) tokens: usize,
    pub(crate) q_scale: f32,
    pub(crate) beta_scale: f32,
}

impl candle_core::InplaceOp3 for PrepChunk {
    fn name(&self) -> &'static str {
        "almost_prep_chunk"
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
        candle_core::bail!("prep_chunk is a cuda prefill building block")
    }

    fn cuda_fwd(
        &self,
        bundle: &mut candle_core::CudaStorage,
        bundle_layout: &candle_core::Layout,
        dyn_rows: &candle_core::CudaStorage,
        dyn_layout: &candle_core::Layout,
        v_beta: &candle_core::CudaStorage,
        v_beta_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use cudarc::driver::{DevicePtr, PushKernelArg};

        let (heads, chunks, chunk, row_width) = bundle_layout.shape().dims4()?;
        let (dyn_height, dyn_width) = dyn_layout.shape().dims2()?;
        let v_dims = v_beta_layout.shape().dims4()?;
        let conv_width = 2 * heads * 96 + heads * 192;
        if chunk != 64
            || row_width != BUNDLE_ROW
            || chunks != self.tokens.div_ceil(64)
            || dyn_height != self.tokens + 7
            || dyn_width != conv_width + 2 * heads
            || v_dims != (heads, chunks, 64, 192)
        {
            candle_core::bail!(
                "prep_chunk wants bundle (h, n, 64, {BUNDLE_ROW}), dyn (t+7, cw+2h), \
                 v_beta (h, n, 64, 192) for t {}; got {:?}, {:?}, {v_dims:?}",
                self.tokens,
                (heads, chunks, chunk, row_width),
                (dyn_height, dyn_width)
            );
        }
        if !bundle_layout.is_contiguous()
            || !dyn_layout.is_contiguous()
            || !v_beta_layout.is_contiguous()
        {
            candle_core::bail!("prep_chunk wants contiguous operands");
        }

        let device = bundle.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, "prep_chunk_f32") {
            Ok(function) => function,
            Err(error) => candle_core::bail!("{error}"),
        };
        let bundle_slice = bundle
            .as_cuda_slice::<f32>()?
            .slice(bundle_layout.start_offset()..);
        let (bundle_ptr, _bundle_guard) = bundle_slice.device_ptr(&stream);
        let dyn_slice = dyn_rows
            .as_cuda_slice::<half::bf16>()?
            .slice(dyn_layout.start_offset()..);
        let (dyn_ptr, _dyn_guard) = dyn_slice.device_ptr(&stream);
        let v_slice = v_beta
            .as_cuda_slice::<f32>()?
            .slice(v_beta_layout.start_offset()..);
        let (v_ptr, _v_guard) = v_slice.device_ptr(&stream);

        let tokens = self.tokens as u32;
        let heads_u = heads as u32;
        let chunks_u = chunks as u32;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (heads_u, chunks_u, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 64 * 4,
        };
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&bundle_ptr)
            .arg(&dyn_ptr)
            .arg(&v_ptr)
            .arg(&tokens)
            .arg(&heads_u)
            .arg(&chunks_u)
            .arg(&self.q_scale)
            .arg(&self.beta_scale);
        if let Err(error) = unsafe { builder.launch(config) } {
            candle_core::bail!("prep_chunk launch failed: {error}");
        }
        Ok(())
    }
}

/// `kbg.inplace_op2(&bundle, &PrepKbg)`: kbg (heads, chunks, 64, 96)
/// f32 - WRITTEN with (k * beta) * exp(g_cum) per row, read straight
/// off the bundle.
pub(crate) struct PrepKbg;

impl candle_core::InplaceOp2 for PrepKbg {
    fn name(&self) -> &'static str {
        "almost_prep_kbg"
    }

    fn cpu_fwd(
        &self,
        _s1: &mut candle_core::CpuStorage,
        _l1: &candle_core::Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        candle_core::bail!("prep_kbg is a cuda prefill building block")
    }

    fn cuda_fwd(
        &self,
        kbg: &mut candle_core::CudaStorage,
        kbg_layout: &candle_core::Layout,
        bundle: &candle_core::CudaStorage,
        bundle_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use cudarc::driver::{DevicePtr, PushKernelArg};

        let kbg_dims = kbg_layout.shape().dims4()?;
        let (heads, chunks, chunk, row_width) = bundle_layout.shape().dims4()?;
        if chunk != 64 || row_width != BUNDLE_ROW || kbg_dims != (heads, chunks, 64, 96) {
            candle_core::bail!(
                "prep_kbg wants kbg (h, n, 64, 96) and bundle (h, n, 64, {BUNDLE_ROW}); \
                 got {kbg_dims:?}, {:?}",
                (heads, chunks, chunk, row_width)
            );
        }
        if !kbg_layout.is_contiguous() || !bundle_layout.is_contiguous() {
            candle_core::bail!("prep_kbg wants contiguous operands");
        }

        let device = kbg.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, "prep_kbg_f32") {
            Ok(function) => function,
            Err(error) => candle_core::bail!("{error}"),
        };
        let kbg_slice = kbg.as_cuda_slice::<f32>()?.slice(kbg_layout.start_offset()..);
        let (kbg_ptr, _kbg_guard) = kbg_slice.device_ptr(&stream);
        let bundle_slice = bundle
            .as_cuda_slice::<f32>()?
            .slice(bundle_layout.start_offset()..);
        let (bundle_ptr, _bundle_guard) = bundle_slice.device_ptr(&stream);

        let config = cudarc::driver::LaunchConfig {
            grid_dim: ((heads * chunks) as u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&function);
        builder.arg(&kbg_ptr).arg(&bundle_ptr);
        if let Err(error) = unsafe { builder.launch(config) } {
            candle_core::bail!("prep_kbg launch failed: {error}");
        }
        Ok(())
    }
}

/// `state.inplace_op3(&bundle, &out, &StateAdvance)`: state
/// (heads, dk, dv) f32 - the carried state, advanced IN PLACE across
/// every chunk (the kernel loops chunks serially inside); bundle
/// (heads, chunks, 64, BUNDLE_ROW) f32 rows q | k | w | u | g | beta
/// (beta rides for TriSolve/PrepKbg; this kernel ignores it); out
/// (heads, chunks, 64, dv) f32 - WRITTEN by the kernel (the mixer
/// outputs; candle tensors mutate through shared storage by design).
/// All contiguous, all addressed at their layout offsets.
/// Geometry-locked to dk 96 / dv 192 (compile-time tiles keep the
/// state column in registers); the grid splits each head's
/// v-columns across four 48-column stripes for occupancy (120
/// blocks; the shared staging is duplicated per stripe, the
/// per-column math is identical).
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
            || row_width != 3 * dk + dv + 2
            || out_dims != (heads, chunks, 64, dv)
        {
            candle_core::bail!(
                "state_advance wants state (h, dk, dv), bundle (h, n, 64, 3dk+dv+2), \
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

/// The rule back half over a filled bundle: TriSolve, the two cublas
/// gemms, the PackTrim w/u slice_sets, and StateAdvance. Returns
/// (out [t, heads, dv] f32, final state).
fn rule_over_bundle(
    bundle: &candle_core::Tensor,
    v_beta: &candle_core::Tensor,
    state: &candle_core::Tensor,
    t: usize,
) -> LibAlmostResult<(candle_core::Tensor, candle_core::Tensor)> {
    let (heads, chunks, _, _) = bundle.dims4()?;
    let (_, dk, dv) = state.dims3()?;
    let device = bundle.device();
    let kbg = candle_core::Tensor::zeros(
        (heads, chunks, 64, dk),
        candle_core::DType::F32,
        device,
    )?;
    kbg.inplace_op2(bundle, &PrepKbg)?;
    let attn = candle_core::Tensor::zeros(
        (heads * chunks, 64, 64),
        candle_core::DType::F32,
        device,
    )?;
    attn.inplace_op2(bundle, &TriSolve)?;
    let attn = attn.reshape((heads, chunks, 64, 64))?;
    let value = attn.matmul(v_beta)?;
    let k_cumdecay = attn.matmul(&kbg)?;
    bundle.slice_set(&k_cumdecay, 3, 2 * dk)?;
    bundle.slice_set(&value, 3, 3 * dk)?;
    let carried = state.copy()?;
    let out = candle_core::Tensor::zeros(
        (heads, chunks, 64, dv),
        candle_core::DType::F32,
        device,
    )?;
    carried.inplace_op3(bundle, &out, &StateAdvance)?;
    let out = out
        .reshape((heads, chunks * 64, dv))?
        .narrow(1, 0, t)?
        .transpose(0, 1)?
        .contiguous()?;
    Ok((out, carried))
}

/// The PrepFusion pipeline's tensor operands: conv_in [t, 2*key +
/// value] bf16 (a view is fine); a_rows/b_rows [t, heads] bf16 (the
/// cublas gate projections); static_rows [4, cw + 2h] bf16 (the
/// layer's prestored conv-weight rows with A_log | dt_bias in the
/// row-0 pad); conv_tail [3, cw] f32; state [heads, dk, dv] f32.
pub(crate) struct PrepInputs<'a> {
    pub(crate) conv_in: &'a candle_core::Tensor,
    pub(crate) a_rows: &'a candle_core::Tensor,
    pub(crate) b_rows: &'a candle_core::Tensor,
    pub(crate) static_rows: &'a candle_core::Tensor,
    pub(crate) conv_tail: &'a candle_core::Tensor,
    pub(crate) state: &'a candle_core::Tensor,
}

/// The PrepFusion chunk pipeline: pack the dyn rows (weights + tail +
/// conv_in | a | b), run PrepChunk into a fresh bundle, then the rule
/// back half. Returns (out [t, heads, dv] f32, final state, new tail
/// [3, cw] f32 - the last 3 raw history rows, classic-exact through
/// the bf16 round-trip).
pub(crate) fn prep_chunk_rule(
    parts: PrepInputs<'_>,
    q_scale: f64,
    allow_neg_eigval: bool,
) -> LibAlmostResult<(candle_core::Tensor, candle_core::Tensor, candle_core::Tensor)> {
    let PrepInputs {
        conv_in,
        a_rows,
        b_rows,
        static_rows,
        conv_tail,
        state,
    } = parts;
    let (t, conv_width) = conv_in.dims2()?;
    let (heads, dk, dv) = state.dims3()?;
    snafu::ensure_whatever!(
        3 * dk + dv + 2 == BUNDLE_ROW,
        "prep_chunk_rule is geometry-locked to the {BUNDLE_ROW}-column bundle (dk {dk}, dv {dv})"
    );
    let chunks = t.div_ceil(64);
    let device = conv_in.device();

    let tail_bf = conv_tail.to_dtype(candle_core::DType::BF16)?;
    let tail_pad = candle_core::Tensor::zeros(
        (3, 2 * heads),
        candle_core::DType::BF16,
        device,
    )?;
    let tail_rows = candle_core::Tensor::cat(&[&tail_bf, &tail_pad], 1)?;
    let token_rows = candle_core::Tensor::cat(&[conv_in, a_rows, b_rows], 1)?;
    let dyn_rows = candle_core::Tensor::cat(&[static_rows, &tail_rows, &token_rows], 0)?;

    let bundle = candle_core::Tensor::zeros(
        (heads, chunks, 64, BUNDLE_ROW),
        candle_core::DType::F32,
        device,
    )?;
    let v_beta = candle_core::Tensor::zeros(
        (heads, chunks, 64, dv),
        candle_core::DType::F32,
        device,
    )?;
    bundle.inplace_op3(&dyn_rows, &v_beta, &PrepChunk {
        tokens: t,
        q_scale: q_scale as f32,
        beta_scale: if allow_neg_eigval { 2.0 } else { 1.0 },
    })?;

    let (out, carried) = rule_over_bundle(&bundle, &v_beta, state, t)?;
    // The new tail: the last kernel-1 rows of [tail; tokens] - dyn
    // rows [4 + t, 7 + t), conv columns only (classic-exact: the
    // same bf16 round-trip conv_with_tail's history takes).
    let new_tail = dyn_rows
        .narrow(0, 4 + t, 3)?
        .narrow(1, 0, conv_width)?
        .contiguous()?
        .to_dtype(candle_core::DType::F32)?;
    Ok((out, carried, new_tail))
}

/// The rule kernels driven from pre-prepped q/k/v/g/beta tensors (the
/// classic chunk_rule contract: q/k l2-normed, q pre-scaled, all f32,
/// initial-state carry) - the lock rig's harness: candle ops pack the
/// bundle the way PrepChunk writes it, then the back half runs.
#[cfg(test)]
pub(crate) fn rule_from_parts(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    g: &candle_core::Tensor,
    beta: &candle_core::Tensor,
    state: &candle_core::Tensor,
) -> LibAlmostResult<(candle_core::Tensor, candle_core::Tensor)> {
    let (t, heads, dk) = q.dims3()?;
    let dv = v.dim(2)?;
    let device = q.device();
    let pad = (64 - t % 64) % 64;
    let chunks = (t + pad) / 64;

    let head_major = |x: &candle_core::Tensor,
                      width: usize|
     -> LibAlmostResult<candle_core::Tensor> {
        Ok(x
            .transpose(0, 1)?
            .pad_with_zeros(1, 0, pad)?
            .reshape((heads, chunks, 64, width))?)
    };
    let q_r = head_major(q, dk)?;
    let k_r = head_major(k, dk)?;
    let v_r = head_major(v, dv)?;
    let g_r = g
        .transpose(0, 1)?
        .pad_with_zeros(1, 0, pad)?
        .reshape((heads, chunks, 64))?
        .cumsum(2)?;
    let beta_r = beta
        .transpose(0, 1)?
        .pad_with_zeros(1, 0, pad)?
        .reshape((heads, chunks, 64))?;

    let bundle = candle_core::Tensor::zeros(
        (heads, chunks, 64, BUNDLE_ROW),
        candle_core::DType::F32,
        device,
    )?;
    bundle.slice_set(&q_r.contiguous()?, 3, 0)?;
    bundle.slice_set(&k_r.contiguous()?, 3, dk)?;
    bundle.slice_set(&g_r.unsqueeze(3)?.contiguous()?, 3, 3 * dk + dv)?;
    bundle.slice_set(&beta_r.unsqueeze(3)?.contiguous()?, 3, 3 * dk + dv + 1)?;
    let v_beta = v_r.broadcast_mul(&beta_r.unsqueeze(3)?)?.contiguous()?;

    rule_over_bundle(&bundle, &v_beta, state, t)
}
