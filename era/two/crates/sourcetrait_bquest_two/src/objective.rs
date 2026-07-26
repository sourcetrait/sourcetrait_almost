//! The training objectives, as loss blocks over the top hidden state.
//!
//! Every one of these takes the final-normed hidden state and returns
//! a scalar, so the per-layer gradient chain beneath them is
//! untouched: the chain consumes a seed gradient at the top hidden
//! state and knows nothing about which objective produced it. That is
//! why supervised, preference and reinforcement training share one
//! trainer.
//!
//! ## DEV
//! Continued pretraining averaged over EVERY position. The supervised
//! form is the same computation weighted by a mask and divided by the
//! masked count instead of the row count - the whole difference
//! between "continue this document" and "produce this reply".
//!
//! Preference and reinforcement both want a SUM rather than a mean:
//! the DPO log-ratio and the policy-gradient term are defined over a
//! sequence's total log-probability, and dividing by length would
//! silently make long replies cheaper to prefer.
//! ##
// The stage loops are the consumers; the allow retires with them.
#![allow(dead_code)]
use crate::*;

use burn::tensor::{
    ElementConversion,
    Int,
    backend::AutodiffBackend,
};

type FloatTensor<B, const D: usize> = burn::tensor::Tensor<B, D>;

/// Guard against a zero-variance group (every rollout scoring alike
/// carries no preference signal, so its advantages are zero).
const ADVANTAGE_EPS: f32 = 1e-6;

/// Per-position log-probabilities of the target tokens, accumulated
/// head-chunk by head-chunk so the (n, vocab) logits never
/// materialize as one tensor. Positions whose mask is zero contribute
/// nothing.
fn masked_logprob_sum<AD: AutodiffBackend>(
    hidden: FloatTensor<AD, 2>,
    lm_head_transposed: &FloatTensor<AD, 2>,
    targets: &[u32],
    mask: &[u8],
    chunk: usize,
    device: &AD::Device,
) -> BquestResult<FloatTensor<AD, 1>> {
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
        let logits = hidden_rows.matmul(lm_head_transposed.clone());
        let log_probs = burn::tensor::activation::log_softmax(logits, 1);
        let indices: Vec<i64> = targets[start..end].iter().map(|t| *t as i64).collect();
        let index_tensor = burn::tensor::Tensor::<AD, 2, Int>::from_data(
            burn::tensor::TensorData::new(indices, [rows, 1]),
            device,
        );
        let picked = log_probs.gather(1, index_tensor);
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
pub(crate) fn supervised_count(mask: &[u8]) -> usize {
    mask.iter().filter(|slot| **slot != 0).count()
}

/// Mean next-token cross-entropy over the MASKED positions alone -
/// the supervised objective. An example with nothing masked is an
/// error rather than a zero-loss step, because it would look like a
/// clean step while training nothing.
pub(crate) fn masked_cross_entropy<AD: AutodiffBackend>(
    hidden: FloatTensor<AD, 2>,
    lm_head_transposed: &FloatTensor<AD, 2>,
    targets: &[u32],
    mask: &[u8],
    chunk: usize,
    device: &AD::Device,
) -> BquestResult<FloatTensor<AD, 1>> {
    let supervised = supervised_count(mask);
    snafu::ensure_whatever!(
        supervised > 0,
        "the loss mask supervises no position, so this example trains nothing"
    );
    let summed = masked_logprob_sum::<AD>(hidden, lm_head_transposed, targets, mask, chunk, device)?;
    Ok(summed.neg().div_scalar(supervised as f64))
}

/// A sequence's total log-probability over the masked positions - the
/// quantity both the preference log-ratio and the policy gradient are
/// defined on. Deliberately NOT length-normalized.
pub(crate) fn sequence_logprob<AD: AutodiffBackend>(
    hidden: FloatTensor<AD, 2>,
    lm_head_transposed: &FloatTensor<AD, 2>,
    targets: &[u32],
    mask: &[u8],
    chunk: usize,
    device: &AD::Device,
) -> BquestResult<FloatTensor<AD, 1>> {
    snafu::ensure_whatever!(
        supervised_count(mask) > 0,
        "the sequence masks no position, so it carries no log-probability"
    );
    masked_logprob_sum::<AD>(hidden, lm_head_transposed, targets, mask, chunk, device)
}

/// softplus in the overflow-stable form max(x, 0) + ln(1 + exp(-|x|)).
/// The naive form's exp overflows on a confident pair and the
/// BACKWARD then computes inf/inf = NaN - the same guard the gating
/// path needed, and the reason a preference loss is written this way
/// rather than as -log(sigmoid(x)).
fn softplus_stable<AD: AutodiffBackend>(x: FloatTensor<AD, 1>) -> FloatTensor<AD, 1> {
    let positive_part = burn::tensor::activation::relu(x.clone());
    let negative_magnitude = x.abs().neg();
    positive_part + negative_magnitude.exp().add_scalar(1.0).log()
}

/// The DPO loss for one pair: -log sigmoid(beta * (policy log-ratio -
/// reference log-ratio)), written as softplus(-x).
///
/// The reference log-probabilities arrive as plain scalars because
/// the reference model is the ADAPTER-OFF path - zero-init adapters
/// are bit-exact to the base and the gate locks that - so they are
/// computed once without gradients and never need a second copy of
/// the weights.
pub(crate) fn dpo_loss<AD: AutodiffBackend>(
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

/// Group-relative advantages: each rollout's reward centred on its
/// group's mean and scaled by its spread. A group whose rollouts all
/// scored alike yields zero advantages - there is nothing to prefer,
/// and that is the honest gradient rather than a divide-by-noise.
pub(crate) fn group_advantages(rewards: &[f32]) -> Vec<f32> {
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

/// The policy-gradient term for one rollout: minimising
/// -advantage * log p(response) raises the probability of
/// better-than-average rollouts and lowers the rest.
pub(crate) fn policy_gradient_loss<AD: AutodiffBackend>(
    sequence_logprob: FloatTensor<AD, 1>,
    advantage: f32,
) -> FloatTensor<AD, 1> {
    sequence_logprob.mul_scalar(-(advantage as f64))
}

/// Read a rank-1 scalar off the graph.
pub(crate) fn scalar_of<AD: AutodiffBackend>(value: &FloatTensor<AD, 1>) -> f32 {
    value.clone().into_scalar().elem()
}
