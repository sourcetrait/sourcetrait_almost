use crate::*;

/// Precomputed cos/sin for one RoPE parameterization over the full position
/// range. Tables are computed in f32 then cast once to the model dtype
/// (computing positions in bf16 degrades past ~256) - D2.
#[derive(Debug, Clone)]
pub(crate) struct RopeTables {
    cos: candle_core::Tensor,
    sin: candle_core::Tensor,
    max_position: usize,
}

impl RopeTables {
    /// Vanilla RoPE (Olmo 3 sliding-attention layers).
    pub(crate) fn vanilla(
        head_dim: usize,
        theta: f64,
        max_position: usize,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> AlmostResult<Self> {
        Self::from_inv_freq(default_inv_freq(head_dim, theta), 1.0, max_position, dtype, device)
    }

    /// Table for full-attention layers: YaRN when the config carries a yarn
    /// rope_scaling block, vanilla when absent or explicitly default.
    pub(crate) fn for_full_layers(
        cfg: &Olmo3Config,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> AlmostResult<Self> {
        match &cfg.rope_scaling {
            None => Self::vanilla(cfg.head_dim(), cfg.rope_theta, cfg.max_position_embeddings, dtype, device),
            Some(scaling) if scaling.rope_type == "default" => {
                Self::vanilla(cfg.head_dim(), cfg.rope_theta, cfg.max_position_embeddings, dtype, device)
            }
            Some(scaling) if scaling.rope_type == "yarn" => {
                let params = yarn_params(cfg.head_dim(), cfg.rope_theta, cfg.max_position_embeddings, scaling)?;
                Self::from_inv_freq(
                    params.inv_freq,
                    params.attention_factor,
                    cfg.max_position_embeddings,
                    dtype,
                    device,
                )
            }
            Some(scaling) => snafu::whatever!("unsupported rope_type {}", scaling.rope_type),
        }
    }

    fn from_inv_freq(
        inv_freq: Vec<f32>,
        attention_scaling: f64,
        max_position: usize,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> AlmostResult<Self> {
        let half_dim = inv_freq.len();
        let inv_freq = candle_core::Tensor::from_vec(inv_freq, (1, half_dim), device)?;
        let positions = candle_core::Tensor::arange(0u32, max_position as u32, device)?
            .to_dtype(candle_core::DType::F32)?
            .reshape((max_position, 1))?;
        let freqs = positions.matmul(&inv_freq)?;
        let cos = (freqs.cos()? * attention_scaling)?.to_dtype(dtype)?;
        let sin = (freqs.sin()? * attention_scaling)?.to_dtype(dtype)?;
        Ok(Self { cos, sin, max_position })
    }

    /// Rotate q/k in place at absolute positions offset..offset+seq_len.
    pub(crate) fn apply(
        &self,
        q: &candle_core::Tensor,
        k: &candle_core::Tensor,
        offset: usize,
    ) -> AlmostResult<(candle_core::Tensor, candle_core::Tensor)> {
        let (_b_sz, _h, seq_len, _head_dim) = q.dims4()?;
        snafu::ensure_whatever!(
            offset + seq_len <= self.max_position,
            "position {} exceeds the rope table range {}",
            offset + seq_len,
            self.max_position
        );
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q = r::candle::rope(&q.contiguous()?, &cos, &sin)?;
        let k = r::candle::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

/// 1 / theta^(2i/dim) for i in 0..dim/2.
pub(crate) fn default_inv_freq(head_dim: usize, theta: f64) -> Vec<f32> {
    (0..head_dim)
        .step_by(2)
        .map(|i| (1f64 / theta.powf(i as f64 / head_dim as f64)) as f32)
        .collect()
}

pub(crate) struct YarnParams {
    pub(crate) inv_freq: Vec<f32>,
    pub(crate) attention_factor: f64,
}

/// transformers _compute_yarn_parameters: blend interpolated (freq/factor)
/// and extrapolated (freq) inverse frequencies across a linear ramp between
/// the beta_fast/beta_slow correction dims; attention_factor falls back to
/// the mscale formula 0.1*ln(factor)+1 and multiplies cos/sin downstream.
/// A stored factor wins; recomputing max/original is the absence fallback.
pub(crate) fn yarn_params(
    head_dim: usize,
    theta: f64,
    max_position_embeddings: usize,
    scaling: &RopeScaling,
) -> AlmostResult<YarnParams> {
    let Some(original_max) = scaling.original_max_position_embeddings else {
        snafu::whatever!("yarn rope_scaling requires original_max_position_embeddings");
    };
    let factor = scaling
        .factor
        .unwrap_or(max_position_embeddings as f64 / original_max as f64);
    let attention_factor = scaling.attention_factor.unwrap_or_else(|| {
        if factor <= 1.0 {
            1.0
        } else {
            0.1 * factor.ln() + 1.0
        }
    });
    let beta_fast = scaling.beta_fast.unwrap_or(32.0);
    let beta_slow = scaling.beta_slow.unwrap_or(1.0);

    let dim = head_dim as f64;
    fn correction_dim(rotations: f64, dim: f64, theta: f64, original_max: f64) -> f64 {
        dim * (original_max / (rotations * 2.0 * std::f64::consts::PI)).ln() / (2.0 * theta.ln())
    }
    let low = correction_dim(beta_fast, dim, theta, original_max as f64)
        .floor()
        .max(0.0);
    let high = correction_dim(beta_slow, dim, theta, original_max as f64)
        .ceil()
        .min(dim - 1.0);
    let ramp_span = if high == low { 0.001 } else { high - low };

    let inv_freq = (0..head_dim)
        .step_by(2)
        .enumerate()
        .map(|(freq_idx, i)| {
            let pos_freq = theta.powf(i as f64 / dim);
            let interpolation = 1.0 / (factor * pos_freq);
            let extrapolation = 1.0 / pos_freq;
            let ramp = ((freq_idx as f64 - low) / ramp_span).clamp(0.0, 1.0);
            let extrapolation_factor = 1.0 - ramp;
            (interpolation * (1.0 - extrapolation_factor) + extrapolation * extrapolation_factor) as f32
        })
        .collect();

    Ok(YarnParams {
        inv_freq,
        attention_factor,
    })
}
