//! PrefillDispatch building blocks: the fused chunked-GDN prefill
//! family. PrepChunk collapses the per-chunk prep chain (the carried
//! causal conv + silu, the q/k l2 norms + scale, and the gating
//! scalars with the per-chunk decay cumsum) into ONE launch per
//! layer-call, writing q | k | g | beta into the StateAdvance bundle
//! DIRECTLY (PackTrim: no transpose/pad/cat pipeline exists at all);
//! TriSolve folds the intra-chunk decay-mask build, beta-scaled kkt
//! matmul, and the blocked-substitution triangular inversion into one
//! launch reading the same bundle; the StateAdvancePipeline splits
//! the inter-chunk recurrence in two - StateStage runs the SERIAL
//! state math alone, materializing each chunk's initial state and
//! vnew rows to a scratch, and StateOut computes every chunk's
//! output rows chunk-PARALLEL from that scratch. BundleBf16: the
//! bundle's q | k | w | u columns ride bf16 pairs packed two per
//! f32 cell (halving the rule family's bandwidth-bound bundle
//! traffic) while g | beta and the recurrence state stay exact f32
//! (the vllm accumulation-order lesson); PackPairs packs the gemm
//! outputs into the w/u cells. SmallT: the module compiles once per
//! tile via a prepended SA_TILE define - 64 the prefill default, 32
//! the small-span variant (a <= 17-row speculation verify/readvance
//! span computes a quarter of the tile-64 pad waste; final prefill
//! chunks at or under 32 rows ride it too). Raw pointer args on the
//! model's own stream, operands addressed at their layout offsets
//! (the offset-leak lock).
//!
//! Numerics: the prep kernel runs the conv/silu/l2 chain in f32 where
//! the classic chain rounds through bf16 per op (the FusedDecodeConv
//! precedent - envelope-class, toward truth; the prep lock pins it);
//! the rule kernels run the same f32 math as gdn::chunk_rule with
//! reassociation-class deltas (the rule lock pins those). The cpu
//! path, the stateless parity form, and decode never dispatch here;
//! only the CARRIED cuda bf16 multi-token chunk branch does.
use crate::*;

/// Kernel geometry: one SA_TILE-row chunk tile per block (SmallT:
/// SA_TILE is prepended per compile - 64 or 32; SA_BLOCKS is the
/// blocked-substitution 16-row block count). A bundle row is
/// SA_CELLS f32 cells: q | k | w | u as bf16 pairs (BundleBf16) at
/// SA_QC/SA_KC/SA_WC/SA_UC, then exact-f32 g | beta at SA_GC/SA_BC
/// (beta rides for TriSolve/PrepKbg; the StateAdvancePipeline
/// ignores it). The family is geometry-locked to dk 96 / dv 192.
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

// BundleBf16: the q | k | w | u columns ride bf16 PAIRS packed two
// per f32 cell (one 4-byte load carries two elements - the rule
// family's bundle traffic halves), while g | beta stay EXACT f32
// cells (the gating scalars compound through the recurrent state -
// the vllm accumulation-order lesson). Even column = low half.
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

// The intra-chunk solve: A0 = -(k_beta k^T . decay) strictly below
// the diagonal, then the blocked forward substitution, then + I. One
// block per (head, chunk) tile, SA_TILE threads; k, beta, and the
// cumulative gates read from the bundle rows.
extern "C" __global__ void tri_solve_f32(
    float* a_out,
    const float* bundle
) {
    extern __shared__ float shared[];
    float* sk = shared;                        // [SA_TILE, SA_DK] beta-scaled k
    float* sa = shared + SA_TILE * SA_DK;      // [SA_TILE, SA_TILE] working matrix
    float* srow = sa + SA_TILE * SA_TILE;      // [SA_TILE] row snapshot
    float* sg = srow + SA_TILE;                // [SA_TILE] cumulative gates
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
    // A0[r][c] = -(k_beta[r] . k[c]) * exp(g[r] - g[c]) strictly
    // below the diagonal; zero on and above.
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
    // Blocked forward substitution (BlockedSubstitution): the
    // SA_BLOCKS 16x16 diagonal blocks solve concurrently (independent regions,
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
    // + I, then the row writes back.
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

// PrepKbg: the k_cumdecay gemm operand k * beta * exp(g_cum), read
// straight off the bundle ((k * beta) * exp - the classic mul order).
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

// StateAdvancePipeline pass 1 (StateStage): the serial inter-chunk
// recurrence ALONE - per chunk, write the chunk's INITIAL state
// (scratch rows 0..SA_DK) and its vnew rows (scratch rows
// SA_DK..SA_DK+64) then advance the carried state; no attention
// tile, no output rows (pass 2 computes them chunk-parallel). Grid
// (heads, 4 stripes) = 120 blocks, 48 threads - one thread per
// v-column of its stripe; the vnew and update math is the fused
// single-pass form verbatim (same per-element accumulation orders;
// the VnewUnroll register contract carries).
extern "C" __global__ void state_stage_f32(
    float* state,
    const float* bundle,
    float* scratch,
    unsigned int chunks
) {
    extern __shared__ float shared[];
    float* sk = shared;                        // [SA_TILE, SA_DK] raw k
    float* sg = shared + SA_TILE * SA_DK;      // [SA_TILE] cumulative gates
    float* ses = sg + SA_TILE;                 // [SA_TILE] exp(g_last - g)
    unsigned long long h = blockIdx.x;
    unsigned int stripe = blockIdx.y;
    unsigned int tid = threadIdx.x;
    unsigned int j = stripe * SA_COLS + tid;   // this thread's v-column
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
        // Stage k + gates (strided over the stripe's threads; k
        // unpacks from its pair cells).
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
        // The chunk's INITIAL state column, register-direct.
        #pragma unroll
        for (unsigned int i = 0; i < SA_DK; ++i) {
            chunk_scratch[(unsigned long long)i * SA_DV + j] = s_reg[i];
        }
        __syncthreads();
        // v_new = u - w @ S over the carried column, written to the
        // scratch's vnew rows for the output pass (w and u unpack
        // from their pair cells; per-element order is unchanged).
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
        // S = S * exp(g_last) + (k * exp(g_last - g))^T @ v_new -
        // the landed r-major order.
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

// StateAdvancePipeline pass 2 (StateOut): the chunk-PARALLEL output
// pass - grid (heads, chunks, 4 stripes): every tile reads its
// chunk's initial state and vnew rows from the scratch, builds the
// decayed local attention against its own bundle rows, and writes
// its output rows independently - the serial chain is gone from the
// heavy math. Same per-element accumulation orders as the fused
// single-pass form (attn tile, inter, full-width out loop).
extern "C" __global__ void state_out_f32(
    float* out,
    const float* bundle,
    const float* scratch,
    unsigned int chunks
) {
    extern __shared__ float shared[];
    float* sk = shared;                        // [SA_TILE, SA_DK] raw k
    float* sattn = shared + SA_TILE * SA_DK;   // [SA_TILE, SA_TILE] decayed local attn
    float* sg = sattn + SA_TILE * SA_TILE;     // [SA_TILE] cumulative gates
    float* seg = sg + SA_TILE;                 // [SA_TILE] exp(g)
    unsigned long long h = blockIdx.x;
    unsigned int chunk = blockIdx.y;
    unsigned int stripe = blockIdx.z;
    unsigned int tid = threadIdx.x;
    unsigned int j = stripe * SA_COLS + tid;   // this thread's v-column
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
    // The decayed local attention tile: (q[r] . k[c]) *
    // exp(g[r] - g[c]) on and below the diagonal, zero above.
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
    // out rows: exp(g_r) * (q[r] . S) + attn_local @ v_new
    // (full-width unroll; the upper triangle adds exact zeros).
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

// BundleBf16: pack a gemm's f32 output rows into the bundle's bf16
// pair cells at a fixed cell offset (the w and u columns; also the
// lock harness's q/k packing). One thread per pair; rounding is the
// same round-to-nearest-even every bf16 store in the family uses.
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

/// The bundle row width in f32 CELLS the kernels are compiled
/// against (BundleBf16): q | k | w | u ride bf16 pairs packed two
/// per cell (48 + 48 + 48 + 96 cells), g | beta stay exact f32
/// cells - halving the rule family's bundle traffic while the
/// gating scalars and the recurrence state keep full precision.
pub(crate) const BUNDLE_CELLS: usize = 48 + 48 + 48 + 96 + 2;
/// Pair-cell offsets of the packed columns (the kernel-side
/// SA_QC/SA_KC/SA_WC/SA_UC/SA_GC/SA_BC mirror; Q/K/B are
/// harness-side - the lock rig packs q/k and slice_sets beta).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const CELL_Q: usize = 0;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const CELL_K: usize = 48;
pub(crate) const CELL_W: usize = 96;
pub(crate) const CELL_U: usize = 144;
pub(crate) const CELL_G: usize = 240;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const CELL_B: usize = 241;

/// The dispatch tile for a t-row span (SmallT): spans at or under 32
/// rows ride the tile-32 module - a <= 17-row speculation verify or
/// readvance span computes a quarter of the tile-64 pad waste -
/// everything larger keeps the tile-64 default, so 512-token prefill
/// chunks and 33-63-row final chunks are untouched.
pub(crate) fn tile_for(t: usize) -> usize {
    if t <= 32 { 32 } else { 64 }
}

/// Lazily nvrtc-compiled modules, one per tile variant (SmallT;
/// single-device box). Both variants compile from the SAME source
/// with SA_TILE prepended, so tile-64 codegen is byte-for-byte the
/// pre-SmallT module.
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

/// `a.inplace_op2(&bundle, &TriSolve { tile })`: a (blocks, tile,
/// tile) f32 - WRITTEN by the kernel (the inverted-and-identity-added
/// intra-chunk matrix); bundle (heads, chunks, tile, BUNDLE_CELLS)
/// with blocks == heads * chunks (k unpacked from its pair cells;
/// cumulative g and beta at their f32 cells). The tile field names
/// the compiled module (SmallT: 32 or 64). All contiguous, all
/// addressed at their layout offsets.
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

/// `bundle.inplace_op3(&dyn_rows, &v_beta, &PrepChunk { .. })`:
/// bundle (heads, chunks, tile, BUNDLE_CELLS) f32 zeros - q | k
/// (packed pairs) and g | beta WRITTEN by the kernel (the w | u
/// cells stay zero for the gemm PackPairs);
/// dyn_rows (tokens + 7, conv_width + 2 * heads) bf16 -
/// rows 0..3 the conv-weight taps (A_log | dt_bias in the row-0 pad
/// columns), 4..6 the carried tail, 7.. the token rows
/// (conv_in | a | b); v_beta (heads, chunks, tile, 192) f32 - WRITTEN
/// (candle tensors mutate through shared storage by design).
/// Geometry-locked to dk 96 / dv 192; tile names the module (SmallT).
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

/// `kbg.inplace_op2(&bundle, &PrepKbg { tile })`: kbg (heads, chunks,
/// tile, 96) f32 - WRITTEN with (k * beta) * exp(g_cum) per row, read
/// straight off the bundle.
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

/// `state.inplace_op3(&bundle, &scratch, &StateStage { tile })`:
/// pass 1 of the StateAdvancePipeline - state (heads, dk, dv) f32,
/// the carried state, advanced IN PLACE across every chunk (the
/// serial recurrence lives here alone); bundle (heads, chunks, tile,
/// BUNDLE_CELLS) rows q | k | w | u | g | beta (q and beta unused
/// in this pass); scratch (heads, chunks, dk + tile, dv) f32 -
/// WRITTEN with each chunk's INITIAL state (rows 0..dk) and vnew
/// rows (rows dk..dk+tile) for the parallel output pass (candle
/// tensors mutate through shared storage by design). All contiguous,
/// all addressed at their layout offsets. Geometry-locked to dk 96 /
/// dv 192; grid (heads, 4 stripes) x 48 threads - the landed
/// register contract.
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

/// `out.inplace_op3(&bundle, &scratch, &StateOut { tile })`: pass 2
/// of the StateAdvancePipeline - out (heads, chunks, tile, dv) f32
/// WRITTEN chunk-PARALLEL (grid (heads, chunks, 4 stripes) x 48
/// threads): every tile reads its chunk's initial state and vnew
/// rows from the scratch, builds the decayed local attention against
/// its own bundle rows, and writes its output rows independently -
/// no serial dependency remains in the heavy math. Same per-element
/// accumulation orders as the fused single-pass form.
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

/// `bundle.inplace_op2(&src, &PackPairs { cell_offset })`: pack an
/// f32 tensor's rows into the bundle's bf16 pair cells at a fixed
/// cell offset - the w/u gemm outputs, and the lock harness's q/k.
/// dst (h, n, tile, BUNDLE_CELLS) f32; src (h, n, tile, width) f32
/// with width even and the packed span inside the pair-cell region.
/// Tile-agnostic (row-flat kernel; the chunk dims just have to
/// agree).
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
        // The pack kernel is row-flat, so either module serves it;
        // the tile-64 module always exists once anything prefills.
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

/// The rule back half over a filled bundle: TriSolve, the two cublas
/// gemms, the BundleBf16 w/u pair packing, and the
/// StateAdvancePipeline (StateStage serial, StateOut chunk-parallel
/// over the scratch). The tile rides the bundle's chunk dim (SmallT:
/// the caller sized it via tile_for); `pooled` carries the reused
/// kbg/attn/scratch/out set on a full span
/// (PrefillScratchReuse - every cell rewrites below, TriSolve and
/// the writers covering their whole outputs, so pooled and fresh
/// zeros are value-identical). Returns (out [t, heads, dv] f32,
/// final state).
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

/// PrefillScratchReuse: the six per-layer-call prefill tensors for
/// ONE standing chunk shape, allocated once and shared by every GDN
/// layer (they run sequentially; Rc single-thread by the same
/// contract as the graph cache). Only FULL spans reuse
/// (t == chunks * tile): there every cell of every tensor is
/// rewritten each pass - PrepChunk covers q/k/g/beta and v_beta,
/// PackPairs the w/u cells, TriSolve/PrepKbg/StateStage/StateOut
/// their whole outputs - so reuse is value-identical to fresh zeros
/// by construction, and no re-zeroing exists at all. Ragged and
/// small spans keep per-call allocation (the pad-row discipline
/// stays the allocator's). A shape change (a different chunks/tile)
/// reallocates the set; prefill tensors are never graph-captured,
/// so the replaced set frees legally.
pub(crate) struct PrefillScratch {
    shape: ScratchShape,
    bundle: candle_core::Tensor,
    v_beta: candle_core::Tensor,
    kbg: candle_core::Tensor,
    attn: candle_core::Tensor,
    scratch: candle_core::Tensor,
    out: candle_core::Tensor,
}

/// One standing pool geometry (the full-span shape a checkout
/// serves).
#[derive(Clone, Copy, PartialEq, Eq)]
struct ScratchShape {
    heads: usize,
    chunks: usize,
    tile: usize,
    dk: usize,
    dv: usize,
}

/// The checked-out pooled set: bundle, v_beta, kbg, attn, scratch,
/// out (clones sharing the pool's storage).
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

    /// The pooled set for a FULL span, (re)allocated on a shape
    /// change; None for any span the pool does not serve. The clones
    /// share storage (the kernels write in place), so no RefCell
    /// borrow outlives this call.
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
/// initial-state carry) - the lock rig's harness: PackPairs and
/// slice_set fill the bundle the way PrepChunk writes it (q/k as bf16
/// pairs, g/beta exact f32), then the back half runs. The tile
/// derives from t exactly like the live paths (SmallT).
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
