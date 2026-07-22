use crate::*;

/// Inverse frequencies for vanilla RoPE: 1 / theta^(2i/dim), f64 math.
pub(crate) fn default_inv_freq(head_dim: usize, theta: f64) -> Vec<f64> {
    (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / theta.powf(i as f64 / head_dim as f64))
        .collect()
}

/// YaRN inverse frequencies + attention factor, ported independently from
/// transformers' _compute_yarn_parameters: interpolated (freq/factor) and
/// extrapolated (freq) blends across a linear ramp between the correction
/// dims found for beta_fast/beta_slow, computed against the ORIGINAL
/// pretraining length; the attention factor (explicit, else 0.1*ln(f)+1)
/// scales cos and sin downstream.
pub(crate) fn yarn_inv_freq(
    head_dim: usize,
    theta: f64,
    max_position_embeddings: usize,
    scaling: &HeatRopeScaling,
) -> HeatResult<(Vec<f64>, f64)> {
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
    let span = if high == low { 0.001 } else { high - low };

    let inv_freq = (0..head_dim)
        .step_by(2)
        .enumerate()
        .map(|(freq_index, i)| {
            let pos_freq = theta.powf(i as f64 / dim);
            let ramp = ((freq_index as f64 - low) / span).clamp(0.0, 1.0);
            let extrapolation_share = 1.0 - ramp;
            (1.0 / (factor * pos_freq)) * (1.0 - extrapolation_share)
                + (1.0 / pos_freq) * extrapolation_share
        })
        .collect();
    Ok((inv_freq, attention_factor))
}

/// cos/sin tables in the HF layout - row p is cat(freqs, freqs) evaluated at
/// position p, times the attention factor - flattened row-major (n, head_dim).
pub(crate) fn rope_tables(
    inv_freq: &[f64],
    attention_factor: f64,
    positions: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let width = half * 2;
    let mut cos = vec![0f32; positions * width];
    let mut sin = vec![0f32; positions * width];
    for position in 0..positions {
        for (i, freq) in inv_freq.iter().enumerate() {
            let angle = position as f64 * freq;
            let cos_value = (angle.cos() * attention_factor) as f32;
            let sin_value = (angle.sin() * attention_factor) as f32;
            cos[position * width + i] = cos_value;
            cos[position * width + half + i] = cos_value;
            sin[position * width + i] = sin_value;
            sin[position * width + half + i] = sin_value;
        }
    }
    (cos, sin)
}
