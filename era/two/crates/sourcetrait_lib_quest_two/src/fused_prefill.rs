//! The fused chunked-GDN prefill family: prep, solve, and the advance.
use crate::*;

/// The prefill-family kernel source, compiled once per tile variant.
const KERNEL_SRC: &str = r#"
#define SA_BLOCKS (SA_TILE / 16u)
#define SA_DK 96u
#define SA_DV 192u
#define SA_COLS 48u
#define SA_QC 0u
#define SA_KC 48u
#define SA_WC 96u
#define SA_UC 144u
#define SA_GC 240u
#define SA_BC 241u
#define SA_CELLS 242u
#define PC_TAPS 4u

__device__ __forceinline__ float bf16_to_f32(unsigned short bits) {
    unsigned int wide = ((unsigned int)bits) << 16;
    return __uint_as_float(wide);
}

__device__ __forceinline__ unsigned short f32_to_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    unsigned int rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
    return (unsigned short)(rounded >> 16);
}

// The bundle's bf16 pair packing: two values per f32 cell.
__device__ __forceinline__ void unpack_pair(float cell, float* even, float* odd) {
    unsigned int bits = __float_as_uint(cell);
    *even = bf16_to_f32((unsigned short)(bits & 0xFFFFu));
    *odd = bf16_to_f32((unsigned short)(bits >> 16u));
}

__device__ __forceinline__ float pack_pair(float even, float odd) {
    unsigned int bits = (unsigned int)f32_to_bf16(even)
        | (((unsigned int)f32_to_bf16(odd)) << 16u);
    return __uint_as_float(bits);
}

// The intra-chunk solve, by blocked forward substitution.
extern "C" __global__ void tri_solve_f32(
    float* a_out,
    const float* bundle
) {
    extern __shared__ float shared[];
    float* sk = shared;
    float* sa = shared + SA_TILE * SA_DK;
    float* srow = sa + SA_TILE * SA_TILE;
    float* sg = srow + SA_TILE;
    unsigned long long b = blockIdx.x;
    unsigned int r = threadIdx.x;
    const float* rows = bundle + b * (unsigned long long)SA_TILE * SA_CELLS;
    float beta_r = rows[r * SA_CELLS + SA_BC];
    sg[r] = rows[r * SA_CELLS + SA_GC];
    for (unsigned int i2 = 0; i2 < SA_COLS; ++i2) {
        float k_even;
        float k_odd;
        unpack_pair(rows[r * SA_CELLS + SA_KC + i2], &k_even, &k_odd);
        sk[r * SA_DK + 2u * i2] = k_even * beta_r;
        sk[r * SA_DK + 2u * i2 + 1u] = k_odd * beta_r;
    }
    __syncthreads();
    for (unsigned int c = 0; c < SA_TILE; ++c) {
        float value = 0.f;
        if (c < r) {
            float dot = 0.f;
            for (unsigned int i2 = 0; i2 < SA_COLS; ++i2) {
                float k_even;
                float k_odd;
                unpack_pair(rows[c * SA_CELLS + SA_KC + i2], &k_even, &k_odd);
                dot += sk[r * SA_DK + 2u * i2] * k_even
                    + sk[r * SA_DK + 2u * i2 + 1u] * k_odd;
            }
            value = -(dot * expf(sg[r] - sg[c]));
        }
        sa[r * SA_TILE + c] = value;
    }
    __syncthreads();
    unsigned int db = r >> 4u;
    unsigned int dc = r & 15u;
    for (unsigned int s = 1; s < 16u; ++s) {
        if (dc < s) {
            srow[r] = sa[(db * 16u + s) * SA_TILE + db * 16u + dc];
        }
        __syncthreads();
        if (dc < s) {
            float value = srow[r];
            for (unsigned int i = 0; i < s; ++i) {
                value += srow[db * 16u + i] * sa[(db * 16u + i) * SA_TILE + db * 16u + dc];
            }
            sa[(db * 16u + s) * SA_TILE + db * 16u + dc] = value;
        }
        __syncthreads();
    }
    float* stile = sk;
    for (unsigned int bi = 1; bi < SA_BLOCKS; ++bi) {
        for (unsigned int e = r; e < bi * 256u; e += SA_TILE) {
            unsigned int bj = e >> 8u;
            unsigned int ti = e & 255u;
            unsigned int er = ti >> 4u;
            unsigned int ec = ti & 15u;
            unsigned int row_g = bi * 16u + er;
            float s_val = sa[row_g * SA_TILE + bj * 16u + ec];
            for (unsigned int k = bj; k < bi; ++k) {
                for (unsigned int i = 0; i < 16u; ++i) {
                    s_val += sa[row_g * SA_TILE + k * 16u + i]
                        * sa[(k * 16u + i) * SA_TILE + bj * 16u + ec];
                }
            }
            stile[e] = s_val;
        }
        __syncthreads();
        for (unsigned int e = r; e < bi * 256u; e += SA_TILE) {
            unsigned int bj = e >> 8u;
            unsigned int ti = e & 255u;
            unsigned int er = ti >> 4u;
            unsigned int ec = ti & 15u;
            unsigned int row_g = bi * 16u + er;
            float t_val = stile[e];
            for (unsigned int i = 0; i < 16u; ++i) {
                t_val += sa[row_g * SA_TILE + bi * 16u + i]
                    * stile[bj * 256u + i * 16u + ec];
            }
            sa[row_g * SA_TILE + bj * 16u + ec] = t_val;
        }
        __syncthreads();
    }
    float* out_row = a_out
        + (b * (unsigned long long)SA_TILE + r) * (unsigned long long)SA_TILE;
    for (unsigned int c = 0; c < SA_TILE; ++c) {
        float value = sa[r * SA_TILE + c];
        if (c == r) {
            value += 1.0f;
        }
        out_row[c] = value;
    }
}

// The per-chunk prep chain in one launch, written bundle-direct.
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
    __shared__ float sg[SA_TILE];
    unsigned int h = blockIdx.x;
    unsigned int chunk = blockIdx.y;
    unsigned int r = threadIdx.x;
    unsigned int tok = chunk * SA_TILE + r;
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
        for (unsigned int i = 1; i < SA_TILE; ++i) {
            sg[i] += sg[i - 1u];
        }
    }
    __syncthreads();
    unsigned long long brow =
        ((unsigned long long)(h * chunks + chunk) * SA_TILE + r) * SA_CELLS;
    bundle[brow + SA_GC] = sg[r];
    if (tok >= t) { return; }
    bundle[brow + SA_BC] = beta;
    unsigned int hist = 4u + tok;
    unsigned int q_base = h * SA_DK;
    unsigned int k_base = heads * SA_DK + h * SA_DK;
    unsigned int v_base = 2u * heads * SA_DK + h * SA_DV;
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
    float q_pair[2];
    float k_pair[2];
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
        q_pair[i & 1u] = q_s * q_inv * q_scale;
        k_pair[i & 1u] = k_s * k_inv;
        if ((i & 1u) == 1u) {
            bundle[brow + SA_QC + i / 2u] = pack_pair(q_pair[0], q_pair[1]);
            bundle[brow + SA_KC + i / 2u] = pack_pair(k_pair[0], k_pair[1]);
        }
    }
    unsigned long long vrow =
        ((unsigned long long)(h * chunks + chunk) * SA_TILE + r) * SA_DV;
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

// The k-cumdecay gemm operand, read straight off the bundle.
extern "C" __global__ void prep_kbg_f32(
    float* kbg,
    const float* bundle
) {
    unsigned long long row = (unsigned long long)blockIdx.x * SA_TILE + threadIdx.x;
    const float* src = bundle + row * SA_CELLS;
    float beta_v = src[SA_BC];
    float g_exp = expf(src[SA_GC]);
    float* dst = kbg + row * SA_DK;
    for (unsigned int i2 = 0; i2 < SA_COLS; ++i2) {
        float k_even;
        float k_odd;
        unpack_pair(src[SA_KC + i2], &k_even, &k_odd);
        dst[2u * i2] = (k_even * beta_v) * g_exp;
        dst[2u * i2 + 1u] = (k_odd * beta_v) * g_exp;
    }
}

// Advance pass one: the serial inter-chunk recurrence alone.
extern "C" __global__ void state_stage_f32(
    float* state,
    const float* bundle,
    float* scratch,
    unsigned int chunks
) {
    extern __shared__ float shared[];
    float* sk = shared;
    float* sg = shared + SA_TILE * SA_DK;
    float* ses = sg + SA_TILE;
    unsigned long long h = blockIdx.x;
    unsigned int stripe = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned int j = stripe * SA_COLS + tid;
    const unsigned long long chunk_stride = (unsigned long long)SA_TILE * SA_CELLS;
    const float* bundle_h = bundle + h * (unsigned long long)chunks * chunk_stride;
    float* state_h = state + h * (unsigned long long)SA_DK * SA_DV;
    float* scratch_h = scratch
        + h * (unsigned long long)chunks * (unsigned long long)(SA_DK + SA_TILE) * SA_DV;

    float s_reg[SA_DK];
    float vnew[SA_TILE];
    #pragma unroll
    for (unsigned int i = 0; i < SA_DK; ++i) {
        s_reg[i] = state_h[(unsigned long long)i * SA_DV + j];
    }

    for (unsigned int chunk = 0; chunk < chunks; ++chunk) {
        const float* rows = bundle_h + chunk * chunk_stride;
        float* chunk_scratch = scratch_h
            + chunk * (unsigned long long)(SA_DK + SA_TILE) * SA_DV;
        __syncthreads();
        for (unsigned int idx = tid; idx < SA_TILE * SA_COLS; idx += SA_COLS) {
            unsigned int r = idx / SA_COLS;
            unsigned int i2 = idx - r * SA_COLS;
            float k_even;
            float k_odd;
            unpack_pair(rows[r * SA_CELLS + SA_KC + i2], &k_even, &k_odd);
            sk[r * SA_DK + 2u * i2] = k_even;
            sk[r * SA_DK + 2u * i2 + 1u] = k_odd;
        }
        for (unsigned int r = tid; r < SA_TILE; r += SA_COLS) {
            sg[r] = rows[r * SA_CELLS + SA_GC];
        }
        __syncthreads();
        for (unsigned int r = tid; r < SA_TILE; r += SA_COLS) {
            ses[r] = expf(sg[SA_TILE - 1u] - sg[r]);
        }
        #pragma unroll
        for (unsigned int i = 0; i < SA_DK; ++i) {
            chunk_scratch[(unsigned long long)i * SA_DV + j] = s_reg[i];
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int r = 0; r < SA_TILE; ++r) {
            const float* row = rows + r * SA_CELLS;
            float acc = 0.f;
            #pragma unroll
            for (unsigned int i2 = 0; i2 < SA_COLS; ++i2) {
                float w_even;
                float w_odd;
                unpack_pair(row[SA_WC + i2], &w_even, &w_odd);
                acc += w_even * s_reg[2u * i2] + w_odd * s_reg[2u * i2 + 1u];
            }
            float u_even;
            float u_odd;
            unpack_pair(row[SA_UC + j / 2u], &u_even, &u_odd);
            vnew[r] = ((j & 1u) == 0u ? u_even : u_odd) - acc;
            chunk_scratch[(unsigned long long)(SA_DK + r) * SA_DV + j] = vnew[r];
        }
        float e_last = expf(sg[SA_TILE - 1u]);
        #pragma unroll
        for (unsigned int i = 0; i < SA_DK; ++i) {
            s_reg[i] *= e_last;
        }
        #pragma unroll
        for (unsigned int r = 0; r < SA_TILE; ++r) {
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

// Advance pass two: the chunk-parallel output rows.
extern "C" __global__ void state_out_f32(
    float* out,
    const float* bundle,
    const float* scratch,
    unsigned int chunks
) {
    extern __shared__ float shared[];
    float* sk = shared;
    float* sattn = shared + SA_TILE * SA_DK;
    float* sg = sattn + SA_TILE * SA_TILE;
    float* seg = sg + SA_TILE;
    unsigned long long h = blockIdx.x;
    unsigned int chunk = blockIdx.y;
    unsigned int stripe = blockIdx.z;
    unsigned int tid = threadIdx.x;
    unsigned int j = stripe * SA_COLS + tid;
    const unsigned long long chunk_stride = (unsigned long long)SA_TILE * SA_CELLS;
    const float* rows = bundle
        + (h * (unsigned long long)chunks + chunk) * chunk_stride;
    const float* chunk_scratch = scratch
        + (h * (unsigned long long)chunks + chunk)
            * (unsigned long long)(SA_DK + SA_TILE) * SA_DV;
    float* out_rows = out
        + (h * (unsigned long long)chunks + chunk)
            * (unsigned long long)SA_TILE * SA_DV;

    for (unsigned int idx = tid; idx < SA_TILE * SA_COLS; idx += SA_COLS) {
        unsigned int r = idx / SA_COLS;
        unsigned int i2 = idx - r * SA_COLS;
        float k_even;
        float k_odd;
        unpack_pair(rows[r * SA_CELLS + SA_KC + i2], &k_even, &k_odd);
        sk[r * SA_DK + 2u * i2] = k_even;
        sk[r * SA_DK + 2u * i2 + 1u] = k_odd;
    }
    for (unsigned int r = tid; r < SA_TILE; r += SA_COLS) {
        sg[r] = rows[r * SA_CELLS + SA_GC];
    }
    __syncthreads();
    for (unsigned int r = tid; r < SA_TILE; r += SA_COLS) {
        seg[r] = expf(sg[r]);
    }
    for (unsigned int idx = tid; idx < SA_TILE * SA_TILE; idx += SA_COLS) {
        unsigned int r = idx / SA_TILE;
        unsigned int c = idx - r * SA_TILE;
        float value = 0.f;
        if (c <= r) {
            const float* q_row = rows + r * SA_CELLS;
            float dot = 0.f;
            #pragma unroll
            for (unsigned int i2 = 0; i2 < SA_COLS; ++i2) {
                float q_even;
                float q_odd;
                unpack_pair(q_row[SA_QC + i2], &q_even, &q_odd);
                dot += q_even * sk[c * SA_DK + 2u * i2]
                    + q_odd * sk[c * SA_DK + 2u * i2 + 1u];
            }
            value = dot * expf(sg[r] - sg[c]);
        }
        sattn[idx] = value;
    }
    __syncthreads();
    float s_reg[SA_DK];
    float vnew[SA_TILE];
    #pragma unroll
    for (unsigned int i = 0; i < SA_DK; ++i) {
        s_reg[i] = chunk_scratch[(unsigned long long)i * SA_DV + j];
    }
    #pragma unroll
    for (unsigned int r = 0; r < SA_TILE; ++r) {
        vnew[r] = chunk_scratch[(unsigned long long)(SA_DK + r) * SA_DV + j];
    }
    for (unsigned int r = 0; r < SA_TILE; ++r) {
        const float* q_row = rows + r * SA_CELLS;
        float inter = 0.f;
        #pragma unroll
        for (unsigned int i2 = 0; i2 < SA_COLS; ++i2) {
            float q_even;
            float q_odd;
            unpack_pair(q_row[SA_QC + i2], &q_even, &q_odd);
            inter += q_even * s_reg[2u * i2] + q_odd * s_reg[2u * i2 + 1u];
        }
        float acc = inter * seg[r];
        const float* attn_row = sattn + r * SA_TILE;
        #pragma unroll
        for (unsigned int c = 0; c < SA_TILE; ++c) {
            acc += attn_row[c] * vnew[c];
        }
        out_rows[(unsigned long long)r * SA_DV + j] = acc;
    }
}

// Pack a gemm's f32 rows into the bundle's bf16 pair cells.
extern "C" __global__ void pack_pairs_bf16(
    float* bundle,
    const float* src,
    unsigned int cell_offset,
    unsigned int width,
    unsigned int rows_total
) {
    unsigned int pairs = width / 2u;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows_total * pairs) { return; }
    unsigned int row = idx / pairs;
    unsigned int p = idx - row * pairs;
    bundle[(unsigned long long)row * SA_CELLS + cell_offset + p] = pack_pair(
        src[(unsigned long long)row * width + 2u * p],
        src[(unsigned long long)row * width + 2u * p + 1u]
    );
}
"#;

/// The bundle row width in f32 CELLS the kernels compile against.
pub(crate) const BUNDLE_CELLS: usize = 48 + 48 + 48 + 96 + 2;
/// Cell offsets of the packed columns, mirroring the kernel's own.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const CELL_Q: usize = 0;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const CELL_K: usize = 48;
pub(crate) const CELL_W: usize = 96;
pub(crate) const CELL_U: usize = 144;
pub(crate) const CELL_G: usize = 240;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const CELL_B: usize = 241;

/// The dispatch tile for a t-row span; 32 rows or fewer take tile 32.
pub(crate) fn tile_for(t: usize) -> usize {
    if t <= 32 { 32 } else { 64 }
}

/// Lazily compiled modules, one per tile variant, from one source.
static MODULE_TILE_64: std::sync::OnceLock<std::sync::Arc<cudarc::driver::CudaModule>> =
    std::sync::OnceLock::new();
static MODULE_TILE_32: std::sync::OnceLock<std::sync::Arc<cudarc::driver::CudaModule>> =
    std::sync::OnceLock::new();

fn kernel(
    device: &candle_core::CudaDevice,
    name: &str,
    tile: usize,
) -> LibQuestResult<cudarc::driver::CudaFunction> {
    let cell = match tile {
        64 => &MODULE_TILE_64,
        32 => &MODULE_TILE_32,
        other => snafu::whatever!("no tile-{other} prefill kernel module (32 and 64 exist)"),
    };
    let module = match cell.get() {
        Some(module) => module.clone(),
        None => {
            let source = format!("#define SA_TILE {tile}u\n{KERNEL_SRC}");
            let ptx = match cudarc::nvrtc::compile_ptx(source) {
                Ok(ptx) => ptx,
                Err(error) => {
                    snafu::whatever!(
                        "nvrtc compile of the tile-{tile} prefill kernels failed: {error}"
                    )
                }
            };
            let module = match device.cuda_stream().context().load_module(ptx) {
                Ok(module) => module,
                Err(error) => {
                    snafu::whatever!(
                        "loading the tile-{tile} prefill-kernel module failed: {error}"
                    )
                }
            };
            cell.get_or_init(|| module).clone()
        }
    };
    match module.load_function(name) {
        Ok(function) => Ok(function),
        Err(error) => snafu::whatever!("loading kernel {name} failed: {error}"),
    }
}

/// `a.inplace_op2(&bundle, &TriSolve { tile })`; `a` is the output.
pub(crate) struct TriSolve {
    pub(crate) tile: usize,
}

impl candle_core::InplaceOp2 for TriSolve {
    fn name(&self) -> &'static str {
        "quest_tri_solve"
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
        if chunk != self.tile
            || chunk_2 != self.tile
            || b_chunk != self.tile
            || row_width != BUNDLE_CELLS
            || blocks != b_heads * b_chunks
        {
            candle_core::bail!(
                "tri_solve wants a (h*n, {tile}, {tile}) and bundle (h, n, {tile}, \
                 {BUNDLE_CELLS}); got {:?}, {:?}",
                (blocks, chunk, chunk_2),
                (b_heads, b_chunks, b_chunk, row_width),
                tile = self.tile
            );
        }
        if !a_layout.is_contiguous() || !bundle_layout.is_contiguous() {
            candle_core::bail!("tri_solve wants contiguous operands");
        }

        let device = a.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, "tri_solve_f32", self.tile) {
            Ok(function) => function,
            Err(error) => candle_core::bail!("{error}"),
        };
        let a_slice = a.as_cuda_slice::<f32>()?.slice(a_layout.start_offset()..);
        let (a_ptr, _a_guard) = a_slice.device_ptr(&stream);
        let bundle_slice = bundle
            .as_cuda_slice::<f32>()?
            .slice(bundle_layout.start_offset()..);
        let (bundle_ptr, _bundle_guard) = bundle_slice.device_ptr(&stream);

        let tile = self.tile;
        let shared_bytes = ((tile * 96 + tile * tile + tile + tile) * 4) as u32;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (tile as u32, 1, 1),
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

/// `bundle.inplace_op3(&dyn_rows, &v_beta, &PrepChunk { .. })`.
pub(crate) struct PrepChunk {
    pub(crate) tokens: usize,
    pub(crate) q_scale: f32,
    pub(crate) beta_scale: f32,
    pub(crate) tile: usize,
}

impl candle_core::InplaceOp3 for PrepChunk {
    fn name(&self) -> &'static str {
        "quest_prep_chunk"
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
        if chunk != self.tile
            || row_width != BUNDLE_CELLS
            || chunks != self.tokens.div_ceil(self.tile)
            || dyn_height != self.tokens + 7
            || dyn_width != conv_width + 2 * heads
            || v_dims != (heads, chunks, self.tile, 192)
        {
            candle_core::bail!(
                "prep_chunk wants bundle (h, n, {tile}, {BUNDLE_CELLS}), dyn (t+7, cw+2h), \
                 v_beta (h, n, {tile}, 192) for t {}; got {:?}, {:?}, {v_dims:?}",
                self.tokens,
                (heads, chunks, chunk, row_width),
                (dyn_height, dyn_width),
                tile = self.tile
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
        let function = match kernel(&device, "prep_chunk_f32", self.tile) {
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
            block_dim: (self.tile as u32, 1, 1),
            shared_mem_bytes: (self.tile * 4) as u32,
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

/// `kbg.inplace_op2(&bundle, &PrepKbg { tile })`; kbg is the output.
pub(crate) struct PrepKbg {
    pub(crate) tile: usize,
}

impl candle_core::InplaceOp2 for PrepKbg {
    fn name(&self) -> &'static str {
        "quest_prep_kbg"
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
        if chunk != self.tile
            || row_width != BUNDLE_CELLS
            || kbg_dims != (heads, chunks, self.tile, 96)
        {
            candle_core::bail!(
                "prep_kbg wants kbg (h, n, {tile}, 96) and bundle (h, n, {tile}, \
                 {BUNDLE_CELLS}); got {kbg_dims:?}, {:?}",
                (heads, chunks, chunk, row_width),
                tile = self.tile
            );
        }
        if !kbg_layout.is_contiguous() || !bundle_layout.is_contiguous() {
            candle_core::bail!("prep_kbg wants contiguous operands");
        }

        let device = kbg.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, "prep_kbg_f32", self.tile) {
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
            block_dim: (self.tile as u32, 1, 1),
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

/// `state.inplace_op3(&bundle, &scratch, &StateStage { tile })`.
pub(crate) struct StateStage {
    pub(crate) tile: usize,
}

impl candle_core::InplaceOp3 for StateStage {
    fn name(&self) -> &'static str {
        "quest_state_stage"
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
        candle_core::bail!("state_stage is a cuda prefill building block")
    }

    fn cuda_fwd(
        &self,
        state: &mut candle_core::CudaStorage,
        state_layout: &candle_core::Layout,
        bundle: &candle_core::CudaStorage,
        bundle_layout: &candle_core::Layout,
        scratch: &candle_core::CudaStorage,
        scratch_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use cudarc::driver::{DevicePtr, PushKernelArg};

        let (heads, dk, dv) = state_layout.shape().dims3()?;
        let (b_heads, chunks, b_chunk, row_width) = bundle_layout.shape().dims4()?;
        let scratch_dims = scratch_layout.shape().dims4()?;
        if b_heads != heads
            || b_chunk != self.tile
            || row_width != BUNDLE_CELLS
            || scratch_dims != (heads, chunks, dk + self.tile, dv)
        {
            candle_core::bail!(
                "state_stage wants state (h, dk, dv), bundle (h, n, {tile}, \
                 {BUNDLE_CELLS}), scratch (h, n, dk+{tile}, dv); got {:?}, {:?}, \
                 {scratch_dims:?}",
                (heads, dk, dv),
                (b_heads, chunks, b_chunk, row_width),
                tile = self.tile
            );
        }
        if dk != 96 || dv != 192 {
            candle_core::bail!(
                "state_stage is geometry-locked to dk == 96 and dv == 192 (got {dk}, {dv})"
            );
        }
        if !state_layout.is_contiguous()
            || !bundle_layout.is_contiguous()
            || !scratch_layout.is_contiguous()
        {
            candle_core::bail!("state_stage wants contiguous operands");
        }

        let device = state.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, "state_stage_f32", self.tile) {
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
        let scratch_slice = scratch
            .as_cuda_slice::<f32>()?
            .slice(scratch_layout.start_offset()..);
        let (scratch_ptr, _scratch_guard) = scratch_slice.device_ptr(&stream);

        let chunks = chunks as u32;
        let shared_bytes = ((self.tile * 96 + 2 * self.tile) * 4) as u32;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (heads as u32, 4, 1),
            block_dim: (48, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&state_ptr)
            .arg(&bundle_ptr)
            .arg(&scratch_ptr)
            .arg(&chunks);
        if let Err(error) = unsafe { builder.launch(config) } {
            candle_core::bail!("state_stage launch failed: {error}");
        }
        Ok(())
    }
}

/// `out.inplace_op3(&bundle, &scratch, &StateOut { tile })`.
pub(crate) struct StateOut {
    pub(crate) tile: usize,
}

impl candle_core::InplaceOp3 for StateOut {
    fn name(&self) -> &'static str {
        "quest_state_out"
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
        candle_core::bail!("state_out is a cuda prefill building block")
    }

    fn cuda_fwd(
        &self,
        out: &mut candle_core::CudaStorage,
        out_layout: &candle_core::Layout,
        bundle: &candle_core::CudaStorage,
        bundle_layout: &candle_core::Layout,
        scratch: &candle_core::CudaStorage,
        scratch_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use cudarc::driver::{DevicePtr, PushKernelArg};

        let (heads, chunks, o_chunk, dv) = out_layout.shape().dims4()?;
        let bundle_dims = bundle_layout.shape().dims4()?;
        let scratch_dims = scratch_layout.shape().dims4()?;
        if o_chunk != self.tile
            || bundle_dims != (heads, chunks, self.tile, BUNDLE_CELLS)
            || scratch_dims != (heads, chunks, 96 + self.tile, dv)
        {
            candle_core::bail!(
                "state_out wants out (h, n, {tile}, dv), bundle (h, n, {tile}, \
                 {BUNDLE_CELLS}), scratch (h, n, 96+{tile}, dv); got {:?}, \
                 {bundle_dims:?}, {scratch_dims:?}",
                (heads, chunks, o_chunk, dv),
                tile = self.tile
            );
        }
        if dv != 192 {
            candle_core::bail!(
                "state_out is geometry-locked to dk == 96 and dv == 192 (got dv {dv})"
            );
        }
        if !out_layout.is_contiguous()
            || !bundle_layout.is_contiguous()
            || !scratch_layout.is_contiguous()
        {
            candle_core::bail!("state_out wants contiguous operands");
        }

        let device = out.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, "state_out_f32", self.tile) {
            Ok(function) => function,
            Err(error) => candle_core::bail!("{error}"),
        };
        let out_slice = out.as_cuda_slice::<f32>()?.slice(out_layout.start_offset()..);
        let (out_ptr, _out_guard) = out_slice.device_ptr(&stream);
        let bundle_slice = bundle
            .as_cuda_slice::<f32>()?
            .slice(bundle_layout.start_offset()..);
        let (bundle_ptr, _bundle_guard) = bundle_slice.device_ptr(&stream);
        let scratch_slice = scratch
            .as_cuda_slice::<f32>()?
            .slice(scratch_layout.start_offset()..);
        let (scratch_ptr, _scratch_guard) = scratch_slice.device_ptr(&stream);

        let chunks_u = chunks as u32;
        let shared_bytes =
            ((self.tile * 96 + self.tile * self.tile + 2 * self.tile) * 4) as u32;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (heads as u32, chunks_u, 4),
            block_dim: (48, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&out_ptr)
            .arg(&bundle_ptr)
            .arg(&scratch_ptr)
            .arg(&chunks_u);
        if let Err(error) = unsafe { builder.launch(config) } {
            candle_core::bail!("state_out launch failed: {error}");
        }
        Ok(())
    }
}

/// `bundle.inplace_op2(&src, &PackPairs { cell_offset })`, row-flat.
pub(crate) struct PackPairs {
    pub(crate) cell_offset: usize,
}

impl candle_core::InplaceOp2 for PackPairs {
    fn name(&self) -> &'static str {
        "quest_pack_pairs"
    }

    fn cpu_fwd(
        &self,
        _s1: &mut candle_core::CpuStorage,
        _l1: &candle_core::Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        candle_core::bail!("pack_pairs is a cuda prefill building block")
    }

    fn cuda_fwd(
        &self,
        bundle: &mut candle_core::CudaStorage,
        bundle_layout: &candle_core::Layout,
        src: &candle_core::CudaStorage,
        src_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use cudarc::driver::{DevicePtr, PushKernelArg};

        let (heads, chunks, chunk, row_width) = bundle_layout.shape().dims4()?;
        let src_dims = src_layout.shape().dims4()?;
        let width = src_dims.3;
        if row_width != BUNDLE_CELLS
            || src_dims != (heads, chunks, chunk, width)
            || width % 2 != 0
            || self.cell_offset + width / 2 > CELL_G
        {
            candle_core::bail!(
                "pack_pairs wants bundle (h, n, tile, {BUNDLE_CELLS}) and src (h, n, tile, \
                 even width) inside the pair cells; got {:?}, {src_dims:?} at cell {}",
                (heads, chunks, chunk, row_width),
                self.cell_offset
            );
        }
        if !bundle_layout.is_contiguous() || !src_layout.is_contiguous() {
            candle_core::bail!("pack_pairs wants contiguous operands");
        }

        let device = bundle.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, "pack_pairs_bf16", 64) {
            Ok(function) => function,
            Err(error) => candle_core::bail!("{error}"),
        };
        let bundle_slice = bundle
            .as_cuda_slice::<f32>()?
            .slice(bundle_layout.start_offset()..);
        let (bundle_ptr, _bundle_guard) = bundle_slice.device_ptr(&stream);
        let src_slice = src.as_cuda_slice::<f32>()?.slice(src_layout.start_offset()..);
        let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);

        let rows_total = (heads * chunks * chunk) as u32;
        let cell_offset = self.cell_offset as u32;
        let width = width as u32;
        let total = rows_total * (width / 2);
        let block = 256u32;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (total.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&bundle_ptr)
            .arg(&src_ptr)
            .arg(&cell_offset)
            .arg(&width)
            .arg(&rows_total);
        if let Err(error) = unsafe { builder.launch(config) } {
            candle_core::bail!("pack_pairs launch failed: {error}");
        }
        Ok(())
    }
}

/// The rule back half over a filled bundle: solve, gemms, advance.
fn rule_over_bundle(
    bundle: &candle_core::Tensor,
    v_beta: &candle_core::Tensor,
    state: &candle_core::Tensor,
    t: usize,
    pooled: Option<(
        candle_core::Tensor,
        candle_core::Tensor,
        candle_core::Tensor,
        candle_core::Tensor,
    )>,
) -> LibQuestResult<(candle_core::Tensor, candle_core::Tensor)> {
    let (heads, chunks, tile, _) = bundle.dims4()?;
    let (_, dk, dv) = state.dims3()?;
    let device = bundle.device();
    let (kbg, attn, scratch, out) = match pooled {
        Some(set) => set,
        None => (
            candle_core::Tensor::zeros(
                (heads, chunks, tile, dk),
                candle_core::DType::F32,
                device,
            )?,
            candle_core::Tensor::zeros(
                (heads * chunks, tile, tile),
                candle_core::DType::F32,
                device,
            )?,
            candle_core::Tensor::zeros(
                (heads, chunks, dk + tile, dv),
                candle_core::DType::F32,
                device,
            )?,
            candle_core::Tensor::zeros(
                (heads, chunks, tile, dv),
                candle_core::DType::F32,
                device,
            )?,
        ),
    };
    kbg.inplace_op2(bundle, &PrepKbg { tile })?;
    attn.inplace_op2(bundle, &TriSolve { tile })?;
    let attn = attn.reshape((heads, chunks, tile, tile))?;
    let value = attn.matmul(v_beta)?;
    let k_cumdecay = attn.matmul(&kbg)?;
    bundle.inplace_op2(&k_cumdecay, &PackPairs { cell_offset: CELL_W })?;
    bundle.inplace_op2(&value, &PackPairs { cell_offset: CELL_U })?;
    let carried = state.copy()?;
    carried.inplace_op3(bundle, &scratch, &StateStage { tile })?;
    out.inplace_op3(bundle, &scratch, &StateOut { tile })?;
    let out = out
        .reshape((heads, chunks * tile, dv))?
        .narrow(1, 0, t)?
        .transpose(0, 1)?
        .contiguous()?;
    Ok((out, carried))
}

/// The six per-layer-call prefill tensors for one standing shape.
pub(crate) struct PrefillScratch {
    shape: ScratchShape,
    bundle: candle_core::Tensor,
    v_beta: candle_core::Tensor,
    kbg: candle_core::Tensor,
    attn: candle_core::Tensor,
    scratch: candle_core::Tensor,
    out: candle_core::Tensor,
}

/// One standing pool geometry: the full-span shape it serves.
#[derive(Clone, Copy, PartialEq, Eq)]
struct ScratchShape {
    heads: usize,
    chunks: usize,
    tile: usize,
    dk: usize,
    dv: usize,
}

/// The checked-out set, as clones sharing the pool's storage.
type PooledSet = (
    candle_core::Tensor,
    candle_core::Tensor,
    candle_core::Tensor,
    candle_core::Tensor,
    candle_core::Tensor,
    candle_core::Tensor,
);

pub(crate) type SharedPrefillScratch =
    std::rc::Rc<std::cell::RefCell<Option<PrefillScratch>>>;

impl PrefillScratch {
    fn allocate(shape: ScratchShape, device: &candle_core::Device) -> LibQuestResult<Self> {
        let ScratchShape { heads, chunks, tile, dk, dv } = shape;
        let zeros = |dims: (usize, usize, usize, usize)| {
            candle_core::Tensor::zeros(dims, candle_core::DType::F32, device)
        };
        Ok(Self {
            shape,
            bundle: zeros((heads, chunks, tile, BUNDLE_CELLS))?,
            v_beta: zeros((heads, chunks, tile, dv))?,
            kbg: zeros((heads, chunks, tile, dk))?,
            attn: candle_core::Tensor::zeros(
                (heads * chunks, tile, tile),
                candle_core::DType::F32,
                device,
            )?,
            scratch: zeros((heads, chunks, dk + tile, dv))?,
            out: zeros((heads, chunks, tile, dv))?,
        })
    }

    /// The pooled set for a FULL span; None for anything else.
    fn checkout(
        pool: Option<&SharedPrefillScratch>,
        shape: ScratchShape,
        t: usize,
        device: &candle_core::Device,
    ) -> LibQuestResult<Option<PooledSet>> {
        let Some(shared) = pool else {
            return Ok(None);
        };
        if t != shape.chunks * shape.tile {
            return Ok(None);
        }
        let mut slot = shared.borrow_mut();
        if slot.as_ref().is_none_or(|held| held.shape != shape) {
            *slot = Some(Self::allocate(shape, device)?);
        }
        let held = slot.as_ref().expect("just ensured");
        Ok(Some((
            held.bundle.clone(),
            held.v_beta.clone(),
            held.kbg.clone(),
            held.attn.clone(),
            held.scratch.clone(),
            held.out.clone(),
        )))
    }
}

/// The prep pipeline's tensor operands, gathered into one argument.
pub(crate) struct PrepInputs<'a> {
    pub(crate) conv_in: &'a candle_core::Tensor,
    pub(crate) a_rows: &'a candle_core::Tensor,
    pub(crate) b_rows: &'a candle_core::Tensor,
    pub(crate) static_rows: &'a candle_core::Tensor,
    pub(crate) conv_tail: &'a candle_core::Tensor,
    pub(crate) state: &'a candle_core::Tensor,
}

/// The chunk pipeline: pack the dyn rows, prep, then the back half.
pub(crate) fn prep_chunk_rule(
    parts: PrepInputs<'_>,
    q_scale: f64,
    allow_neg_eigval: bool,
    pool: Option<&SharedPrefillScratch>,
) -> LibQuestResult<(candle_core::Tensor, candle_core::Tensor, candle_core::Tensor)> {
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
        dk == 96 && dv == 192,
        "prep_chunk_rule is geometry-locked to the {BUNDLE_CELLS}-cell bundle (dk {dk}, dv {dv})"
    );
    let tile = tile_for(t);
    let chunks = t.div_ceil(tile);
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

    let pooled_set = PrefillScratch::checkout(
        pool,
        ScratchShape { heads, chunks, tile, dk, dv },
        t,
        device,
    )?;
    let (bundle, v_beta, pooled_rest) = match pooled_set {
        Some((bundle, v_beta, kbg, attn, scratch, out)) => {
            (bundle, v_beta, Some((kbg, attn, scratch, out)))
        }
        None => (
            candle_core::Tensor::zeros(
                (heads, chunks, tile, BUNDLE_CELLS),
                candle_core::DType::F32,
                device,
            )?,
            candle_core::Tensor::zeros(
                (heads, chunks, tile, dv),
                candle_core::DType::F32,
                device,
            )?,
            None,
        ),
    };
    bundle.inplace_op3(&dyn_rows, &v_beta, &PrepChunk {
        tokens: t,
        q_scale: q_scale as f32,
        beta_scale: if allow_neg_eigval { 2.0 } else { 1.0 },
        tile,
    })?;

    let (out, carried) = rule_over_bundle(&bundle, &v_beta, state, t, pooled_rest)?;
    let new_tail = dyn_rows
        .narrow(0, 4 + t, 3)?
        .narrow(1, 0, conv_width)?
        .contiguous()?
        .to_dtype(candle_core::DType::F32)?;
    Ok((out, carried, new_tail))
}

/// The rule kernels driven from pre-prepped tensors: the lock harness.
#[cfg(test)]
pub(crate) fn rule_from_parts(
    q: &candle_core::Tensor,
    k: &candle_core::Tensor,
    v: &candle_core::Tensor,
    g: &candle_core::Tensor,
    beta: &candle_core::Tensor,
    state: &candle_core::Tensor,
) -> LibQuestResult<(candle_core::Tensor, candle_core::Tensor)> {
    let (t, heads, dk) = q.dims3()?;
    let dv = v.dim(2)?;
    let device = q.device();
    let tile = tile_for(t);
    let pad = (tile - t % tile) % tile;
    let chunks = (t + pad) / tile;

    let head_major = |x: &candle_core::Tensor,
                      width: usize|
     -> LibQuestResult<candle_core::Tensor> {
        Ok(x
            .transpose(0, 1)?
            .pad_with_zeros(1, 0, pad)?
            .reshape((heads, chunks, tile, width))?)
    };
    let q_r = head_major(q, dk)?;
    let k_r = head_major(k, dk)?;
    let v_r = head_major(v, dv)?;
    let g_r = g
        .transpose(0, 1)?
        .pad_with_zeros(1, 0, pad)?
        .reshape((heads, chunks, tile))?
        .cumsum(2)?;
    let beta_r = beta
        .transpose(0, 1)?
        .pad_with_zeros(1, 0, pad)?
        .reshape((heads, chunks, tile))?;

    let bundle = candle_core::Tensor::zeros(
        (heads, chunks, tile, BUNDLE_CELLS),
        candle_core::DType::F32,
        device,
    )?;
    bundle.inplace_op2(&q_r.contiguous()?, &PackPairs { cell_offset: CELL_Q })?;
    bundle.inplace_op2(&k_r.contiguous()?, &PackPairs { cell_offset: CELL_K })?;
    bundle.slice_set(&g_r.unsqueeze(3)?.contiguous()?, 3, CELL_G)?;
    bundle.slice_set(&beta_r.unsqueeze(3)?.contiguous()?, 3, CELL_B)?;
    let v_beta = v_r.broadcast_mul(&beta_r.unsqueeze(3)?)?.contiguous()?;

    rule_over_bundle(&bundle, &v_beta, state, t, None)
}
