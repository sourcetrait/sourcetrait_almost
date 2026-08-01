//! The training objectives, as loss blocks over the top hidden state.
#![allow(dead_code)]
use crate::*;

use burn::tensor::{
    ElementConversion,
    Int,
    backend::AutodiffBackend,
};

type FloatTensor<B, const D: usize> = burn::tensor::Tensor<B, D>;

/// The spread below which a group counts as zero-variance.
const ADVANTAGE_EPS: f32 = 1e-6;

/// One supervised position's loss: (target index, negative log-prob).
pub type TokenLoss = (usize, f32);

/// Add the head delta's contribution onto a logits chunk, by its two
/// column groups; narrow-and-cat so the whole path stays autodiff.
pub fn apply_head_delta<AD: AutodiffBackend>(
    logits: FloatTensor<AD, 2>,
    hidden_rows: &FloatTensor<AD, 2>,
    delta: &lora::HeadDelta<AD>,
) -> FloatTensor<AD, 2> {
    let vocab = logits.dims()[1];
    let config_at = delta.config_index;
    let run_at = delta.run_start;
    let config_add = hidden_rows.clone().matmul(delta.config_col.clone());
    let run_add = hidden_rows.clone().matmul(delta.run_cols.clone());
    let mut parts: Vec<FloatTensor<AD, 2>> = Vec::with_capacity(5);
    if config_at > 0 {
        parts.push(logits.clone().narrow(1, 0, config_at));
    }
    parts.push(logits.clone().narrow(1, config_at, 1) + config_add);
    let gap = run_at - (config_at + 1);
    if gap > 0 {
        parts.push(logits.clone().narrow(1, config_at + 1, gap));
    }
    parts.push(logits.clone().narrow(1, run_at, 6) + run_add);
    let tail = vocab - (run_at + 6);
    if tail > 0 {
        parts.push(logits.clone().narrow(1, run_at + 6, tail));
    }
    burn::tensor::Tensor::cat(parts, 1)
}

/// The masked positions' summed target log-probability, head-chunked.
/// `capture` collects each supervised position's TokenLoss as it goes.
#[allow(clippy::too_many_arguments)]
fn masked_logprob_sum<AD: AutodiffBackend>(
    hidden: FloatTensor<AD, 2>,
    lm_head_transposed: &FloatTensor<AD, 2>,
    head_delta: Option<&lora::HeadDelta<AD>>,
    targets: &[u32],
    mask: &[u8],
    chunk: usize,
    device: &AD::Device,
    mut capture: Option<&mut Vec<TokenLoss>>,
) -> LibQuestResult<FloatTensor<AD, 1>> {
    let n = targets.len();
    snafu::ensure_whatever!(
        mask.len() == n,
        "loss mask carries {} positions for {n} targets",
        mask.len()
    );
    let chunk = chunk.max(1);
    let mut total: Option<FloatTensor<AD, 1>> = None;
    let mut start = 0usize;
    while start < n {
        let end = (start + chunk).min(n);
        let rows = end - start;
        let hidden_rows = hidden.clone().narrow(0, start, rows);
        let logits = hidden_rows.clone().matmul(lm_head_transposed.clone());
        let logits = match head_delta {
            Some(delta) => apply_head_delta::<AD>(logits, &hidden_rows, delta),
            None => logits,
        };
        let log_probs = burn::tensor::activation::log_softmax(logits, 1);
        let indices: Vec<i64> = targets[start..end].iter().map(|t| *t as i64).collect();
        let index_tensor = burn::tensor::Tensor::<AD, 2, Int>::from_data(
            burn::tensor::TensorData::new(indices, [rows, 1]),
            device,
        );
        let picked = log_probs.gather(1, index_tensor);
        if let Some(capture) = capture.as_deref_mut() {
            let values = match picked.clone().into_data().convert::<f32>().to_vec::<f32>() {
                Ok(values) => values,
                Err(error) => snafu::whatever!("token-loss extraction failed: {error:?}"),
            };
            for (offset, value) in values.iter().enumerate() {
                if mask[start + offset] != 0 {
                    capture.push((start + offset, -value));
                }
            }
        }
        let weights: Vec<f32> = mask[start..end].iter().map(|m| *m as f32).collect();
        let weight_tensor = FloatTensor::<AD, 2>::from_data(
            burn::tensor::TensorData::new(weights, [rows, 1]),
            device,
        );
        let chunk_sum = (picked * weight_tensor).sum();
        total = Some(match total {
            Some(accumulated) => accumulated + chunk_sum,
            None => chunk_sum,
        });
        start = end;
    }
    match total {
        Some(total) => Ok(total),
        None => snafu::whatever!("empty target sequence"),
    }
}

/// How many positions a mask actually supervises.
pub fn supervised_count(mask: &[u8]) -> usize {
    mask.iter().filter(|slot| **slot != 0).count()
}

/// The supervised objective: mean cross-entropy over masked positions.
/// `capture` receives each supervised position's TokenLoss when given.
#[allow(clippy::too_many_arguments)]
pub fn masked_cross_entropy<AD: AutodiffBackend>(
    hidden: FloatTensor<AD, 2>,
    lm_head_transposed: &FloatTensor<AD, 2>,
    head_delta: Option<&lora::HeadDelta<AD>>,
    targets: &[u32],
    mask: &[u8],
    chunk: usize,
    device: &AD::Device,
    capture: Option<&mut Vec<TokenLoss>>,
) -> LibQuestResult<FloatTensor<AD, 1>> {
    let supervised = supervised_count(mask);
    snafu::ensure_whatever!(
        supervised > 0,
        "the loss mask supervises no position, so this example trains nothing"
    );
    let summed = masked_logprob_sum::<AD>(
        hidden,
        lm_head_transposed,
        head_delta,
        targets,
        mask,
        chunk,
        device,
        capture,
    )?;
    Ok(summed.neg().div_scalar(supervised as f64))
}

/// A sequence's total masked log-probability; not length-normalized.
pub fn sequence_logprob<AD: AutodiffBackend>(
    hidden: FloatTensor<AD, 2>,
    lm_head_transposed: &FloatTensor<AD, 2>,
    head_delta: Option<&lora::HeadDelta<AD>>,
    targets: &[u32],
    mask: &[u8],
    chunk: usize,
    device: &AD::Device,
) -> LibQuestResult<FloatTensor<AD, 1>> {
    snafu::ensure_whatever!(
        supervised_count(mask) > 0,
        "the sequence masks no position, so it carries no log-probability"
    );
    masked_logprob_sum::<AD>(
        hidden,
        lm_head_transposed,
        head_delta,
        targets,
        mask,
        chunk,
        device,
        None,
    )
}

/// softplus as max(x, 0) + ln(1 + exp(-|x|)): the backward is safe.
fn softplus_stable<AD: AutodiffBackend>(x: FloatTensor<AD, 1>) -> FloatTensor<AD, 1> {
    let positive_part = burn::tensor::activation::relu(x.clone());
    let negative_magnitude = x.abs().neg();
    positive_part + negative_magnitude.exp().add_scalar(1.0).log()
}

/// The DPO loss for one pair, as softplus of the negated log-ratio.
pub fn dpo_loss<AD: AutodiffBackend>(
    chosen_policy: FloatTensor<AD, 1>,
    rejected_policy: FloatTensor<AD, 1>,
    chosen_reference: f32,
    rejected_reference: f32,
    beta: f64,
) -> FloatTensor<AD, 1> {
    let reference_ratio = (chosen_reference - rejected_reference) as f64;
    let logits = (chosen_policy - rejected_policy)
        .sub_scalar(reference_ratio)
        .mul_scalar(beta);
    softplus_stable::<AD>(logits.neg())
}

/// Group-relative advantages: rewards centred and scaled per group.
pub fn group_advantages(rewards: &[f32]) -> Vec<f32> {
    let count = rewards.len();
    if count == 0 {
        return Vec::new();
    }
    let mean = rewards.iter().sum::<f32>() / count as f32;
    let variance =
        rewards.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / count as f32;
    let deviation = variance.sqrt();
    if deviation <= ADVANTAGE_EPS {
        return vec![0.0; count];
    }
    rewards.iter().map(|r| (r - mean) / deviation).collect()
}

/// The policy-gradient term for one rollout: -advantage * logprob.
pub fn policy_gradient_loss<AD: AutodiffBackend>(
    sequence_logprob: FloatTensor<AD, 1>,
    advantage: f32,
) -> FloatTensor<AD, 1> {
    sequence_logprob.mul_scalar(-(advantage as f64))
}

/// Read a rank-1 scalar off the graph.
pub fn scalar_of<AD: AutodiffBackend>(value: &FloatTensor<AD, 1>) -> f32 {
    value.clone().into_scalar().elem()
}
