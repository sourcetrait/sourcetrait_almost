//! The fused gated-delta decode kernels, and the norms around them.
use crate::*;

/// The decode-family kernel source, compiled once per process.
const KERNEL_SRC: &str = r#"
// One block per head, one thread per value column; state in place.
extern "C" __global__ void gdn_fused_step_f32(
    float* state,
    const float* packed,
    float* y,
    unsigned int heads,
    unsigned int dk,
    unsigned int dv
) {
    unsigned int h = blockIdx.x;
    unsigned int j = threadIdx.x;
    if (h >= heads || j >= dv) { return; }
    const float* q = packed + (unsigned long long)h * (2 * dk + dv + 2);
    const float* k = q + dk;
    const float* v = k + dk;
    float decay = v[dv];
    float beta = v[dv + 1];
    float* s = state + ((unsigned long long)h * dk) * dv + j;
    float kv = 0.f;
    for (unsigned int i = 0; i < dk; ++i) {
        kv += s[(unsigned long long)i * dv] * k[i];
    }
    kv *= decay;
    float delta = (v[j] - kv) * beta;
    float acc = 0.f;
    for (unsigned int i = 0; i < dk; ++i) {
        float value = s[(unsigned long long)i * dv] * decay + k[i] * delta;
        s[(unsigned long long)i * dv] = value;
        acc += value * q[i];
    }
    y[(unsigned long long)h * dv + j] = acc;
}

__device__ __forceinline__ float bf16_to_f32(unsigned short bits) {
    unsigned int wide = ((unsigned int)bits) << 16;
    return __uint_as_float(wide);
}

__device__ __forceinline__ unsigned short f32_to_bf16(float value) {
    unsigned int bits = __float_as_uint(value);
    unsigned int rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
    return (unsigned short)(rounded >> 16);
}

// RMSNorm over a row in f32, with one rounding at the store.
extern "C" __global__ void rms_norm_fused_bf16(
    unsigned short* out,
    const unsigned short* x,
    const unsigned short* weight,
    float eps,
    unsigned int columns
) {
    extern __shared__ float partial[];
    unsigned long long row = blockIdx.x;
    const unsigned short* x_row = x + row * columns;
    unsigned short* out_row = out + row * columns;
    float sumsq = 0.f;
    for (unsigned int i = threadIdx.x; i < columns; i += blockDim.x) {
        float value = bf16_to_f32(x_row[i]);
        sumsq += value * value;
    }
    partial[threadIdx.x] = sumsq;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float inv = rsqrtf(partial[0] / (float)columns + eps);
    for (unsigned int i = threadIdx.x; i < columns; i += blockDim.x) {
        float normed = bf16_to_f32(x_row[i]) * inv;
        out_row[i] = f32_to_bf16(normed * bf16_to_f32(weight[i]));
    }
}

// The four-tap causal conv decode step; the tail rotates in place.
extern "C" __global__ void conv_step_fused_bf16(
    unsigned short* out,
    float* tail,
    const unsigned short* packed,
    unsigned int channels
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) { return; }
    float h0 = tail[c];
    float h1 = tail[(unsigned long long)channels + c];
    float h2 = tail[2ull * channels + c];
    float h3 = bf16_to_f32(packed[c]);
    float w0 = bf16_to_f32(packed[(unsigned long long)channels + c]);
    float w1 = bf16_to_f32(packed[2ull * channels + c]);
    float w2 = bf16_to_f32(packed[3ull * channels + c]);
    float w3 = bf16_to_f32(packed[4ull * channels + c]);
    float acc = h0 * w0 + h1 * w1 + h2 * w2 + h3 * w3;
    out[c] = f32_to_bf16(acc / (1.f + expf(-acc)));
    tail[c] = h1;
    tail[(unsigned long long)channels + c] = h2;
    tail[2ull * channels + c] = h3;
}

// The gated form over per-head rows, y and gate packed together.
extern "C" __global__ void rms_norm_gated_fused_bf16(
    unsigned short* out,
    const float* packed,
    const unsigned short* weight,
    float eps,
    unsigned int dv
) {
    extern __shared__ float partial[];
    unsigned long long head = blockIdx.x;
    const float* y_row = packed + head * (2ull * dv);
    const float* gate_row = y_row + dv;
    unsigned short* out_row = out + head * dv;
    float sumsq = 0.f;
    for (unsigned int i = threadIdx.x; i < dv; i += blockDim.x) {
        sumsq += y_row[i] * y_row[i];
    }
    partial[threadIdx.x] = sumsq;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float inv = rsqrtf(partial[0] / (float)dv + eps);
    for (unsigned int i = threadIdx.x; i < dv; i += blockDim.x) {
        float weighted = y_row[i] * inv * bf16_to_f32(weight[i]);
        float gate = gate_row[i];
        float silu = gate / (1.f + expf(-gate));
        out_row[i] = f32_to_bf16(weighted * silu);
    }
}

#define HP_DK 96u
#define HP_DV 192u
#define HP_ROW (2u * HP_DK + HP_DV + 2u)
#define HP_THREADS 128u
// The whole single-token head-prep chain in one launch.
extern "C" __global__ void head_prep_f32(
    float* packed,
    const unsigned short* conv_out,
    const unsigned short* dyn_row,
    unsigned int heads,
    float q_scale,
    float beta_scale
) {
    __shared__ float red[HP_THREADS];
    __shared__ float inv_q;
    __shared__ float inv_k;
    unsigned int h = blockIdx.x;
    unsigned int t = threadIdx.x;
    if (h >= heads) { return; }
    const unsigned short* q_in = conv_out + h * HP_DK;
    const unsigned short* k_in = conv_out + heads * HP_DK + h * HP_DK;
    const unsigned short* v_in = conv_out + 2u * heads * HP_DK + h * HP_DV;
    float* out_row = packed + (unsigned long long)h * HP_ROW;
    float q_val = t < HP_DK ? bf16_to_f32(q_in[t]) : 0.f;
    red[t] = q_val * q_val;
    __syncthreads();
    for (unsigned int stride = HP_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (t < stride) { red[t] += red[t + stride]; }
        __syncthreads();
    }
    if (t == 0u) { inv_q = 1.f / sqrtf(red[0] + 1e-6f); }
    __syncthreads();
    float k_val = t < HP_DK ? bf16_to_f32(k_in[t]) : 0.f;
    red[t] = k_val * k_val;
    __syncthreads();
    for (unsigned int stride = HP_THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (t < stride) { red[t] += red[t + stride]; }
        __syncthreads();
    }
    if (t == 0u) { inv_k = 1.f / sqrtf(red[0] + 1e-6f); }
    __syncthreads();
    if (t < HP_DK) {
        out_row[t] = q_val * inv_q * q_scale;
        out_row[HP_DK + t] = k_val * inv_k;
    }
    for (unsigned int i = t; i < HP_DV; i += HP_THREADS) {
        out_row[2u * HP_DK + i] = bf16_to_f32(v_in[i]);
    }
    if (t == 0u) {
        float a_v = bf16_to_f32(dyn_row[h]);
        float b_v = bf16_to_f32(dyn_row[heads + h]);
        float a_log = bf16_to_f32(dyn_row[2u * heads + h]);
        float dt = bf16_to_f32(dyn_row[3u * heads + h]);
        float x = a_v + dt;
        float sp = x > 20.f ? x : logf(1.f + expf(x));
        out_row[HP_ROW - 2u] = expf(-(expf(a_log) * sp));
        out_row[HP_ROW - 1u] = beta_scale / (1.f + expf(-b_v));
    }
}
"#;

/// Lazily nvrtc-compiled module, one per process (single-device box).
static MODULE: std::sync::OnceLock<std::sync::Arc<cudarc::driver::CudaModule>> =
    std::sync::OnceLock::new();

fn kernel(
    device: &candle_core::CudaDevice,
    name: &str,
) -> LibQuestResult<cudarc::driver::CudaFunction> {
    let module = match MODULE.get() {
        Some(module) => module.clone(),
        None => {
            let ptx = match cudarc::nvrtc::compile_ptx(KERNEL_SRC) {
                Ok(ptx) => ptx,
                Err(error) => {
                    snafu::whatever!("nvrtc compile of the fused kernels failed: {error}")
                }
            };
            let module = match device.cuda_stream().context().load_module(ptx) {
                Ok(module) => module,
                Err(error) => {
                    snafu::whatever!("loading the fused-kernel module failed: {error}")
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

/// The norm kernels' block width; a power of two for the reduction.
const NORM_BLOCK: u32 = 256;

/// `state.inplace_op3(&packed, &y, &GdnFusedStep)`; y is an output.
pub(crate) struct GdnFusedStep;

impl candle_core::InplaceOp3 for GdnFusedStep {
    fn name(&self) -> &'static str {
        "quest_gdn_fused_step"
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
        candle_core::bail!("gdn_fused_step is a cuda decode building block")
    }

    fn cuda_fwd(
        &self,
        state: &mut candle_core::CudaStorage,
        state_layout: &candle_core::Layout,
        packed: &candle_core::CudaStorage,
        packed_layout: &candle_core::Layout,
        y: &candle_core::CudaStorage,
        y_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use cudarc::driver::{DevicePtr, PushKernelArg};

        let (heads, dk, dv) = state_layout.shape().dims3()?;
        let packed_dims = packed_layout.shape().dims2()?;
        let y_dims = y_layout.shape().dims2()?;
        if packed_dims != (heads, 2 * dk + dv + 2) || y_dims != (heads, dv) {
            candle_core::bail!(
                "gdn_fused_step wants state (h, dk, dv), packed (h, 2dk+dv+2), y (h, dv); \
                 got {:?}, {packed_dims:?}, {y_dims:?}",
                (heads, dk, dv)
            );
        }
        if !state_layout.is_contiguous()
            || !packed_layout.is_contiguous()
            || !y_layout.is_contiguous()
        {
            candle_core::bail!("gdn_fused_step wants contiguous operands");
        }
        if dv > 1024 {
            candle_core::bail!("gdn_fused_step blocks one thread per value column (dv <= 1024)");
        }

        let device = state.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, "gdn_fused_step_f32") {
            Ok(function) => function,
            Err(error) => candle_core::bail!("{error}"),
        };

        let state_slice = state.as_cuda_slice::<f32>()?.slice(state_layout.start_offset()..);
        let (state_ptr, _state_guard) = state_slice.device_ptr(&stream);
        let packed_slice = packed
            .as_cuda_slice::<f32>()?
            .slice(packed_layout.start_offset()..);
        let (packed_ptr, _packed_guard) = packed_slice.device_ptr(&stream);
        let y_slice = y.as_cuda_slice::<f32>()?.slice(y_layout.start_offset()..);
        let (y_ptr, _y_guard) = y_slice.device_ptr(&stream);

        let (heads, dk, dv) = (heads as u32, dk as u32, dv as u32);
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (heads, 1, 1),
            block_dim: (dv, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&state_ptr)
            .arg(&packed_ptr)
            .arg(&y_ptr)
            .arg(&heads)
            .arg(&dk)
            .arg(&dv);
        if let Err(error) = unsafe { builder.launch(config) } {
            candle_core::bail!("gdn_fused_step launch failed: {error}");
        }
        Ok(())
    }
}

/// `out.inplace_op3(&x, &weight, &RmsNormFused { eps })`, all bf16.
pub(crate) struct RmsNormFused {
    pub(crate) eps: f32,
}

impl candle_core::InplaceOp3 for RmsNormFused {
    fn name(&self) -> &'static str {
        "quest_rms_norm_fused"
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
        candle_core::bail!("rms_norm_fused is a cuda decode building block")
    }

    fn cuda_fwd(
        &self,
        out: &mut candle_core::CudaStorage,
        out_layout: &candle_core::Layout,
        x: &candle_core::CudaStorage,
        x_layout: &candle_core::Layout,
        weight: &candle_core::CudaStorage,
        weight_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use cudarc::driver::{DevicePtr, PushKernelArg};

        let (rows, columns) = out_layout.shape().dims2()?;
        if x_layout.shape().dims2()? != (rows, columns)
            || weight_layout.shape().dims1()? != columns
        {
            candle_core::bail!(
                "rms_norm_fused wants out/x (rows, columns) and weight (columns,)"
            );
        }
        if !out_layout.is_contiguous()
            || !x_layout.is_contiguous()
            || !weight_layout.is_contiguous()
        {
            candle_core::bail!("rms_norm_fused wants contiguous operands");
        }

        let device = out.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, "rms_norm_fused_bf16") {
            Ok(function) => function,
            Err(error) => candle_core::bail!("{error}"),
        };
        let out_slice = out
            .as_cuda_slice::<half::bf16>()?
            .slice(out_layout.start_offset()..);
        let (out_ptr, _out_guard) = out_slice.device_ptr(&stream);
        let x_slice = x
            .as_cuda_slice::<half::bf16>()?
            .slice(x_layout.start_offset()..);
        let (x_ptr, _x_guard) = x_slice.device_ptr(&stream);
        let weight_slice = weight
            .as_cuda_slice::<half::bf16>()?
            .slice(weight_layout.start_offset()..);
        let (weight_ptr, _weight_guard) = weight_slice.device_ptr(&stream);

        let columns = columns as u32;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (NORM_BLOCK, 1, 1),
            shared_mem_bytes: NORM_BLOCK * 4,
        };
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&out_ptr)
            .arg(&x_ptr)
            .arg(&weight_ptr)
            .arg(&self.eps)
            .arg(&columns);
        if let Err(error) = unsafe { builder.launch(config) } {
            candle_core::bail!("rms_norm_fused launch failed: {error}");
        }
        Ok(())
    }
}

/// `out.inplace_op3(&tail, &packed, &ConvStepFused)`; the tail rotates.
pub(crate) struct ConvStepFused;

impl candle_core::InplaceOp3 for ConvStepFused {
    fn name(&self) -> &'static str {
        "quest_conv_step_fused"
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
        candle_core::bail!("conv_step_fused is a cuda decode building block")
    }

    fn cuda_fwd(
        &self,
        out: &mut candle_core::CudaStorage,
        out_layout: &candle_core::Layout,
        tail: &candle_core::CudaStorage,
        tail_layout: &candle_core::Layout,
        packed: &candle_core::CudaStorage,
        packed_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use cudarc::driver::{DevicePtr, PushKernelArg};

        let (rows, channels) = out_layout.shape().dims2()?;
        if rows != 1
            || tail_layout.shape().dims2()? != (3, channels)
            || packed_layout.shape().dims2()? != (5, channels)
        {
            candle_core::bail!(
                "conv_step_fused wants out (1, C), tail (3, C), packed (5, C)"
            );
        }
        if !out_layout.is_contiguous()
            || !tail_layout.is_contiguous()
            || !packed_layout.is_contiguous()
        {
            candle_core::bail!("conv_step_fused wants contiguous operands");
        }

        let device = out.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, "conv_step_fused_bf16") {
            Ok(function) => function,
            Err(error) => candle_core::bail!("{error}"),
        };
        let out_slice = out
            .as_cuda_slice::<half::bf16>()?
            .slice(out_layout.start_offset()..);
        let (out_ptr, _out_guard) = out_slice.device_ptr(&stream);
        let tail_slice = tail.as_cuda_slice::<f32>()?.slice(tail_layout.start_offset()..);
        let (tail_ptr, _tail_guard) = tail_slice.device_ptr(&stream);
        let packed_slice = packed
            .as_cuda_slice::<half::bf16>()?
            .slice(packed_layout.start_offset()..);
        let (packed_ptr, _packed_guard) = packed_slice.device_ptr(&stream);

        let channels = channels as u32;
        let block = 256u32;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (channels.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&out_ptr)
            .arg(&tail_ptr)
            .arg(&packed_ptr)
            .arg(&channels);
        if let Err(error) = unsafe { builder.launch(config) } {
            candle_core::bail!("conv_step_fused launch failed: {error}");
        }
        Ok(())
    }
}

/// `out.inplace_op3(&packed, &weight, &RmsNormGatedFused { eps })`.
pub(crate) struct RmsNormGatedFused {
    pub(crate) eps: f32,
}

impl candle_core::InplaceOp3 for RmsNormGatedFused {
    fn name(&self) -> &'static str {
        "quest_rms_norm_gated_fused"
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
        candle_core::bail!("rms_norm_gated_fused is a cuda decode building block")
    }

    fn cuda_fwd(
        &self,
        out: &mut candle_core::CudaStorage,
        out_layout: &candle_core::Layout,
        packed: &candle_core::CudaStorage,
        packed_layout: &candle_core::Layout,
        weight: &candle_core::CudaStorage,
        weight_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use cudarc::driver::{DevicePtr, PushKernelArg};

        let (heads, dv) = out_layout.shape().dims2()?;
        if packed_layout.shape().dims2()? != (heads, 2 * dv)
            || weight_layout.shape().dims1()? != dv
        {
            candle_core::bail!(
                "rms_norm_gated_fused wants out (h, dv), packed (h, 2dv), weight (dv,)"
            );
        }
        if !out_layout.is_contiguous()
            || !packed_layout.is_contiguous()
            || !weight_layout.is_contiguous()
        {
            candle_core::bail!("rms_norm_gated_fused wants contiguous operands");
        }

        let device = out.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, "rms_norm_gated_fused_bf16") {
            Ok(function) => function,
            Err(error) => candle_core::bail!("{error}"),
        };
        let out_slice = out
            .as_cuda_slice::<half::bf16>()?
            .slice(out_layout.start_offset()..);
        let (out_ptr, _out_guard) = out_slice.device_ptr(&stream);
        let packed_slice = packed
            .as_cuda_slice::<f32>()?
            .slice(packed_layout.start_offset()..);
        let (packed_ptr, _packed_guard) = packed_slice.device_ptr(&stream);
        let weight_slice = weight
            .as_cuda_slice::<half::bf16>()?
            .slice(weight_layout.start_offset()..);
        let (weight_ptr, _weight_guard) = weight_slice.device_ptr(&stream);

        let dv = dv as u32;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (heads as u32, 1, 1),
            block_dim: (NORM_BLOCK, 1, 1),
            shared_mem_bytes: NORM_BLOCK * 4,
        };
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&out_ptr)
            .arg(&packed_ptr)
            .arg(&weight_ptr)
            .arg(&self.eps)
            .arg(&dv);
        if let Err(error) = unsafe { builder.launch(config) } {
            candle_core::bail!("rms_norm_gated_fused launch failed: {error}");
        }
        Ok(())
    }
}

/// `packed.inplace_op3(&conv_out, &dyn_row, &HeadPrepFused { .. })`.
pub(crate) struct HeadPrepFused {
    pub(crate) q_scale: f32,
    pub(crate) beta_scale: f32,
}

impl candle_core::InplaceOp3 for HeadPrepFused {
    fn name(&self) -> &'static str {
        "quest_head_prep_fused"
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
        candle_core::bail!("head_prep is a cuda decode building block")
    }

    fn cuda_fwd(
        &self,
        packed: &mut candle_core::CudaStorage,
        packed_layout: &candle_core::Layout,
        conv_out: &candle_core::CudaStorage,
        conv_layout: &candle_core::Layout,
        dyn_row: &candle_core::CudaStorage,
        dyn_layout: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use cudarc::driver::{DevicePtr, PushKernelArg};

        let (heads, row) = packed_layout.shape().dims2()?;
        let conv_dims = conv_layout.shape().dims2()?;
        let dyn_dims = dyn_layout.shape().dims2()?;
        if row != 2 * 96 + 192 + 2
            || conv_dims != (1, heads * (2 * 96 + 192))
            || dyn_dims != (1, 4 * heads)
        {
            candle_core::bail!(
                "head_prep wants packed (h, 386), conv_out (1, h*384), dyn (1, 4h); \
                 got {:?}, {conv_dims:?}, {dyn_dims:?}",
                (heads, row)
            );
        }
        if !packed_layout.is_contiguous()
            || !conv_layout.is_contiguous()
            || !dyn_layout.is_contiguous()
        {
            candle_core::bail!("head_prep wants contiguous operands");
        }

        let device = packed.device.clone();
        let stream = device.cuda_stream();
        let function = match kernel(&device, "head_prep_f32") {
            Ok(function) => function,
            Err(error) => candle_core::bail!("{error}"),
        };
        let packed_slice = packed
            .as_cuda_slice::<f32>()?
            .slice(packed_layout.start_offset()..);
        let (packed_ptr, _packed_guard) = packed_slice.device_ptr(&stream);
        let conv_slice = conv_out
            .as_cuda_slice::<half::bf16>()?
            .slice(conv_layout.start_offset()..);
        let (conv_ptr, _conv_guard) = conv_slice.device_ptr(&stream);
        let dyn_slice = dyn_row
            .as_cuda_slice::<half::bf16>()?
            .slice(dyn_layout.start_offset()..);
        let (dyn_ptr, _dyn_guard) = dyn_slice.device_ptr(&stream);

        let heads = heads as u32;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (heads, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&function);
        builder
            .arg(&packed_ptr)
            .arg(&conv_ptr)
            .arg(&dyn_ptr)
            .arg(&heads)
            .arg(&self.q_scale)
            .arg(&self.beta_scale);
        if let Err(error) = unsafe { builder.launch(config) } {
            candle_core::bail!("head_prep launch failed: {error}");
        }
        Ok(())
    }
}
