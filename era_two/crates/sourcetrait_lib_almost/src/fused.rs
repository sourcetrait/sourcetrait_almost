//! GdnChainFusion building blocks: the fused gated-delta decode step -
//! the per-token recurrence (decay, kv memory, delta, state update,
//! y readout) as ONE raw nvrtc kernel over the f32 [heads, dk, dv]
//! state IN PLACE, replacing the ~7-kernel candle chain per GDN layer
//! per step. Raw pointer args on the model's own stream, operands
//! addressed at their layout offsets (the offset-leak lock) -
//! capture-legal exactly like slot_write.
//!
//! Numerics: the same per-element math as gdn::recurrent_step with two
//! deliberate reassociations - the decay fold (decay * sum(state * k)
//! for sum((state * decay) * k)) and serial in-thread reductions over
//! dk (candle sums tree-wise) - so cuda readings move within the bf16
//! envelope class; the cpu path never dispatches here and stays
//! bit-stable by construction.
use crate::*;

/// One block per head, one thread per value column: pass 1 reduces the
/// kv memory for column j, pass 2 updates the state column in place
/// and reduces the y readout. packed rows are q | k | v | decay | beta
/// (2*dk + dv + 2 columns).
const KERNEL_SRC: &str = r#"
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
"#;

/// Lazily nvrtc-compiled module, one per process (single-device box).
static MODULE: std::sync::OnceLock<std::sync::Arc<cudarc::driver::CudaModule>> =
    std::sync::OnceLock::new();

fn kernel(
    device: &candle_core::CudaDevice,
) -> LibAlmostResult<cudarc::driver::CudaFunction> {
    let module = match MODULE.get() {
        Some(module) => module.clone(),
        None => {
            let ptx = match cudarc::nvrtc::compile_ptx(KERNEL_SRC) {
                Ok(ptx) => ptx,
                Err(error) => {
                    snafu::whatever!("nvrtc compile of gdn_fused_step failed: {error}")
                }
            };
            let module = match device.cuda_stream().context().load_module(ptx) {
                Ok(module) => module,
                Err(error) => {
                    snafu::whatever!("loading the gdn_fused_step module failed: {error}")
                }
            };
            MODULE.get_or_init(|| module).clone()
        }
    };
    match module.load_function("gdn_fused_step_f32") {
        Ok(function) => Ok(function),
        Err(error) => snafu::whatever!("loading gdn_fused_step_f32 failed: {error}"),
    }
}

/// `state.inplace_op3(&packed, &y, &GdnFusedStep)`: state
/// (heads, dk, dv) f32 updated IN PLACE; packed
/// (heads, 2*dk + dv + 2) f32 rows q | k | v | decay | beta; y
/// (heads, dv) f32 - WRITTEN by the kernel (the third operand is an
/// output; candle tensors mutate through shared storage by design,
/// same as slice_set). All f32, all contiguous, all addressed at
/// their layout offsets.
pub(crate) struct GdnFusedStep;

impl candle_core::InplaceOp3 for GdnFusedStep {
    fn name(&self) -> &'static str {
        "almost_gdn_fused_step"
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
        let function = match kernel(&device) {
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
