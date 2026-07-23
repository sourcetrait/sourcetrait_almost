//! The LoRA pair: frozen-base additive adapters (the era-one heat
//! method carried; direction-3 placement decides WHICH weights get
//! one - state-carrying GDN weights never do). Consumed by the
//! train-cpt loop and the AdapterLoad merge; seeded fresh here.
// Consumed by the trainer stage (next in CptLoop); the allows
// retire with it.
#![allow(dead_code)]
use crate::*;

use burn::tensor::backend::Backend;

/// One adapted projection's low-rank pair: a [in, rank] drawn
/// N(0, 0.02), b [rank, out] zeros (an untrained pair contributes
/// exactly nothing), contribution (x.a).b * (alpha / rank).
pub(crate) struct LoraPair<B: Backend> {
    pub(crate) a: burn::tensor::Tensor<B, 2>,
    pub(crate) b: burn::tensor::Tensor<B, 2>,
    pub(crate) scale: f64,
}

/// One standard-normal draw (Box-Muller over SplitMix64; the
/// deterministic init discipline - no external rng).
fn normal_draw(rng: &mut SplitMix64) -> f64 {
    let mut first_unit = rng.next_unit();
    if first_unit <= f64::MIN_POSITIVE {
        first_unit = f64::MIN_POSITIVE;
    }
    let second_unit = rng.next_unit();
    (-2.0 * first_unit.ln()).sqrt() * (2.0 * std::f64::consts::PI * second_unit).cos()
}

impl<B: Backend> LoraPair<B> {
    /// Seeded init: a ~ N(0, 0.02) host-drawn then uploaded, b zeros.
    pub(crate) fn init(
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f64,
        rng: &mut SplitMix64,
        device: &B::Device,
    ) -> Self {
        let mut a_values = Vec::with_capacity(in_dim * rank);
        for _ in 0..in_dim * rank {
            a_values.push((normal_draw(rng) * 0.02) as f32);
        }
        let a = burn::tensor::Tensor::from_data(
            burn::tensor::TensorData::new(a_values, [in_dim, rank]),
            device,
        );
        let b = burn::tensor::Tensor::zeros([rank, out_dim], device);
        Self { a, b, scale: alpha / rank as f64 }
    }

    /// The additive contribution for an [n, in] activation:
    /// (x.a).b * scale.
    pub(crate) fn contribution(
        &self,
        x: burn::tensor::Tensor<B, 2>,
    ) -> burn::tensor::Tensor<B, 2> {
        x.matmul(self.a.clone()).matmul(self.b.clone()).mul_scalar(self.scale)
    }

    /// The dense delta this pair merges into a base weight:
    /// (a.b) * scale, [in, out] (the AdapterLoad consumer; the
    /// caller owns any orientation transpose).
    pub(crate) fn merged_delta(&self) -> burn::tensor::Tensor<B, 2> {
        self.a.clone().matmul(self.b.clone()).mul_scalar(self.scale)
    }
}
