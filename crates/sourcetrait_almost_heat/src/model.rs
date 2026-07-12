use crate::*;

/// Stateless Olmo 3 forward on burn/ndarray f32: no KV cache, no offsets,
/// no decode path - every call is one full-sequence prefill over all
/// positions. Deliberately structured unlike the candle implementation so
/// the two stacks share no incremental machinery.
pub(crate) struct HeatModel {
    embed_rows: Vec<f32>,
    embed_hidden: usize,
    layers: Vec<HeatLayer>,
    final_norm: Tensor1,
    lm_head_transposed: Tensor2,
    cos_full: Vec<f32>,
    sin_full: Vec<f32>,
    cos_sliding: Vec<f32>,
    sin_sliding: Vec<f32>,
    heads: usize,
    head_dim: usize,
    window: usize,
    eps: f64,
    max_positions: usize,
    device: BackDevice,
}

impl HeatModel {
    pub(crate) fn new(config: &HeatConfig, mut weights: Weights) -> HeatResult<Self> {
        let kv_heads = config.num_key_value_heads.unwrap_or(config.num_attention_heads);
        snafu::ensure_whatever!(
            kv_heads == config.num_attention_heads,
            "heat supports MHA only (num_key_value_heads {kv_heads} != heads {})",
            config.num_attention_heads
        );
        let device = BackDevice::default();
        let head_dim = config.head_dim();

        let (vocab, embed_hidden, embed_rows) = weights.take_host_matrix("model.embed_tokens.weight")?;
        snafu::ensure_whatever!(
            vocab == config.vocab_size && embed_hidden == config.hidden_size,
            "embedding shape ({vocab}, {embed_hidden}) does not match the config"
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            layers.push(HeatLayer::new(index, config, &mut weights, &device)?);
        }
        let final_norm = weights.take_vector("model.norm.weight", &device)?;
        let lm_head_transposed = if config.tie_word_embeddings {
            let data = burn::tensor::TensorData::new(embed_rows.clone(), [vocab, embed_hidden]);
            Tensor2::from_data(data, &device).swap_dims(0, 1)
        } else {
            weights.take_linear_transposed("lm_head.weight", &device)?
        };

        let (full_inv_freq, attention_factor) = match &config.rope_scaling {
            None => (rope::default_inv_freq(head_dim, config.rope_theta), 1.0),
            Some(scaling) if scaling.rope_type == "default" => {
                (rope::default_inv_freq(head_dim, config.rope_theta), 1.0)
            }
            Some(scaling) if scaling.rope_type == "yarn" => rope::yarn_inv_freq(
                head_dim,
                config.rope_theta,
                config.max_position_embeddings,
                scaling,
            )?,
            Some(scaling) => snafu::whatever!("unsupported rope_type {}", scaling.rope_type),
        };
        let (cos_full, sin_full) =
            rope::rope_tables(&full_inv_freq, attention_factor, config.max_position_embeddings);
        let sliding_inv_freq = rope::default_inv_freq(head_dim, config.rope_theta);
        let (cos_sliding, sin_sliding) =
            rope::rope_tables(&sliding_inv_freq, 1.0, config.max_position_embeddings);

        Ok(Self {
            embed_rows,
            embed_hidden,
            layers,
            final_norm,
            lm_head_transposed,
            cos_full,
            sin_full,
            cos_sliding,
            sin_sliding,
            heads: config.num_attention_heads,
            head_dim,
            window: config.sliding_window,
            eps: config.rms_norm_eps,
            max_positions: config.max_position_embeddings,
            device,
        })
    }

    /// Logits for every position, returned as a flat row-major
    /// (n, vocab_size) f32 buffer.
    pub(crate) fn forward_all(&self, ids: &[u32]) -> HeatResult<(usize, Vec<f32>)> {
        let n = ids.len();
        snafu::ensure_whatever!(n > 0, "empty id sequence");
        snafu::ensure_whatever!(
            n <= self.max_positions,
            "sequence length {n} exceeds max positions {}",
            self.max_positions
        );

        let mut gathered = Vec::with_capacity(n * self.embed_hidden);
        for id in ids {
            let start = (*id as usize) * self.embed_hidden;
            gathered.extend_from_slice(&self.embed_rows[start..start + self.embed_hidden]);
        }
        let mut x = Tensor2::from_data(
            burn::tensor::TensorData::new(gathered, [n, self.embed_hidden]),
            &self.device,
        );

        let full_mask = self.mask(n, None);
        let sliding_mask = self.mask(n, Some(self.window));
        let (cos_full, sin_full) = self.table_slice(&self.cos_full, &self.sin_full, n);
        let (cos_sliding, sin_sliding) = self.table_slice(&self.cos_sliding, &self.sin_sliding, n);

        for layer in &self.layers {
            let (mask, cos, sin) = match layer.kind {
                LayerKind::Full => (&full_mask, &cos_full, &sin_full),
                LayerKind::Sliding => (&sliding_mask, &cos_sliding, &sin_sliding),
            };
            x = layer.forward(x, mask, cos, sin, self.heads, self.head_dim, self.eps);
        }

        let x = rms_norm(x, &self.final_norm, self.eps);
        let logits = x.matmul(self.lm_head_transposed.clone());
        let values = match logits.into_data().to_vec::<f32>() {
            Ok(values) => values,
            Err(error) => snafu::whatever!("logits extraction failed: {error:?}"),
        };
        Ok((n, values))
    }

    /// Additive 0/-inf visibility mask (n, n); window w hides keys more
    /// than w-1 positions back.
    fn mask(&self, n: usize, window: Option<usize>) -> Tensor2 {
        let mut values = vec![0f32; n * n];
        for query in 0..n {
            for key in 0..n {
                let causal = key <= query;
                let within = window.is_none_or(|w| key + w > query);
                if !(causal && within) {
                    values[query * n + key] = f32::NEG_INFINITY;
                }
            }
        }
        Tensor2::from_data(burn::tensor::TensorData::new(values, [n, n]), &self.device)
    }

    fn table_slice(&self, cos: &[f32], sin: &[f32], n: usize) -> (Tensor2, Tensor2) {
        let width = self.head_dim;
        let cos = Tensor2::from_data(
            burn::tensor::TensorData::new(cos[..n * width].to_vec(), [n, width]),
            &self.device,
        );
        let sin = Tensor2::from_data(
            burn::tensor::TensorData::new(sin[..n * width].to_vec(), [n, width]),
            &self.device,
        );
        (cos, sin)
    }
}

struct HeatLayer {
    q_transposed: Tensor2,
    k_transposed: Tensor2,
    v_transposed: Tensor2,
    o_transposed: Tensor2,
    q_norm: Tensor1,
    k_norm: Tensor1,
    gate_transposed: Tensor2,
    up_transposed: Tensor2,
    down_transposed: Tensor2,
    post_attention_norm: Tensor1,
    post_feedforward_norm: Tensor1,
    kind: LayerKind,
}

impl HeatLayer {
    fn new(
        index: usize,
        config: &HeatConfig,
        weights: &mut Weights,
        device: &BackDevice,
    ) -> HeatResult<Self> {
        let prefix = format!("model.layers.{index}");
        let take = |weights: &mut Weights, suffix: &str| -> HeatResult<Tensor2> {
            weights.take_linear_transposed(&format!("{prefix}.{suffix}"), device)
        };
        Ok(Self {
            q_transposed: take(weights, "self_attn.q_proj.weight")?,
            k_transposed: take(weights, "self_attn.k_proj.weight")?,
            v_transposed: take(weights, "self_attn.v_proj.weight")?,
            o_transposed: take(weights, "self_attn.o_proj.weight")?,
            q_norm: weights.take_vector(&format!("{prefix}.self_attn.q_norm.weight"), device)?,
            k_norm: weights.take_vector(&format!("{prefix}.self_attn.k_norm.weight"), device)?,
            gate_transposed: take(weights, "mlp.gate_proj.weight")?,
            up_transposed: take(weights, "mlp.up_proj.weight")?,
            down_transposed: take(weights, "mlp.down_proj.weight")?,
            post_attention_norm: weights
                .take_vector(&format!("{prefix}.post_attention_layernorm.weight"), device)?,
            post_feedforward_norm: weights
                .take_vector(&format!("{prefix}.post_feedforward_layernorm.weight"), device)?,
            kind: config.layer_kind(index)?,
        })
    }

    /// Post-norm block: x + norm(attn(x)), then x + norm(mlp(x)).
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: Tensor2,
        mask: &Tensor2,
        cos: &Tensor2,
        sin: &Tensor2,
        heads: usize,
        head_dim: usize,
        eps: f64,
    ) -> Tensor2 {
        let n = x.dims()[0];
        let hidden = heads * head_dim;
        let residual = x.clone();

        // q/k rmsnorm over the full projection width, before head reshape.
        let q = rms_norm(x.clone().matmul(self.q_transposed.clone()), &self.q_norm, eps);
        let k = rms_norm(x.clone().matmul(self.k_transposed.clone()), &self.k_norm, eps);
        let v = x.matmul(self.v_transposed.clone());

        let q = to_heads(q, n, heads, head_dim);
        let k = to_heads(k, n, heads, head_dim);
        let v = to_heads(v, n, heads, head_dim);

        let q = apply_rope(q, cos, sin, heads);
        let k = apply_rope(k, cos, sin, heads);

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q.matmul(k.swap_dims(1, 2)).mul_scalar(scale)
            + mask.clone().unsqueeze::<3>().expand([heads, n, n]);
        let attention = burn::tensor::activation::softmax(scores, 2);
        let context = attention
            .matmul(v)
            .swap_dims(0, 1)
            .reshape([n, hidden])
            .matmul(self.o_transposed.clone());
        let x = residual + rms_norm(context, &self.post_attention_norm, eps);

        let residual = x.clone();
        let gate = burn::tensor::activation::silu(x.clone().matmul(self.gate_transposed.clone()));
        let up = x.matmul(self.up_transposed.clone());
        let feedforward = (gate * up).matmul(self.down_transposed.clone());
        residual + rms_norm(feedforward, &self.post_feedforward_norm, eps)
    }
}

/// (n, hidden) -> (heads, n, head_dim).
fn to_heads(x: Tensor2, n: usize, heads: usize, head_dim: usize) -> Tensor3 {
    x.reshape([n, heads, head_dim]).swap_dims(0, 1)
}

/// HF rotate-half rope: t*cos + rotate_half(t)*sin, cos/sin (n, head_dim).
fn apply_rope(t: Tensor3, cos: &Tensor2, sin: &Tensor2, heads: usize) -> Tensor3 {
    let [_, n, head_dim] = t.dims();
    let half = head_dim / 2;
    let first = t.clone().slice_dim(2, 0..half);
    let second = t.clone().slice_dim(2, half..head_dim);
    let rotated = burn::tensor::Tensor::cat(vec![second.neg(), first], 2);
    let cos = cos.clone().unsqueeze::<3>().expand([heads, n, head_dim]);
    let sin = sin.clone().unsqueeze::<3>().expand([heads, n, head_dim]);
    t * cos + rotated * sin
}

/// HF RMSNorm: x * rsqrt(mean(x^2) + eps) * weight (all f32 here).
fn rms_norm(x: Tensor2, weight: &Tensor1, eps: f64) -> Tensor2 {
    let [n, width] = x.dims();
    let mean_square = (x.clone() * x.clone()).mean_dim(1);
    let scale = (mean_square + eps).sqrt().recip().expand([n, width]);
    x * scale * weight.clone().unsqueeze::<2>().expand([n, width])
}
