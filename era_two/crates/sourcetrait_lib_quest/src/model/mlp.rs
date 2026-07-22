//! The SwiGLU MLP shared by both block families.
use crate::*;

/// SwiGLU MLP, present on every layer: down(silu(gate(x)) * up(x)).
/// gate/up ride one row-fused projection ([2 * intermediate, hidden],
/// row order gate | up); down stays separate (its input is the
/// product, not x).
pub(crate) struct Mlp {
    gate_up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
    intermediate: usize,
}

impl Mlp {
    pub(crate) fn new(
        hidden: usize,
        intermediate: usize,
        vb: candle_nn::VarBuilder,
    ) -> LibQuestResult<Self> {
        let gate_up = candle_core::Tensor::cat(
            &[
                &vb.pp("gate_proj").get((intermediate, hidden), "weight")?,
                &vb.pp("up_proj").get((intermediate, hidden), "weight")?,
            ],
            0,
        )?;
        Ok(Self {
            gate_up_proj: candle_nn::Linear::new(gate_up, None),
            down_proj: candle_nn::linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?,
            intermediate,
        })
    }

    pub(crate) fn forward(&self, x: &candle_core::Tensor) -> LibQuestResult<candle_core::Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let gated = gate_up.narrow(1, 0, self.intermediate)?.silu()?;
        let up = gate_up.narrow(1, self.intermediate, self.intermediate)?;
        Ok(self.down_proj.forward(&gated.mul(&up)?)?)
    }
}
