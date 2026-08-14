//! The BiquestTrainer's init half: the organism's fresh checkpoint.
use crate::*;

use crate::matrix::EMBED_TENSOR;
use crate::matrix::MATRIX_HEADS;

/// The organism's layer cycle: three GDN layers then one attention
/// layer, the hybrid's own 3:1 rhythm.
const LAYER_CYCLE: usize = 4;
/// The linear-init scale, the transformer convention.
const INIT_SIGMA: f64 = 0.02;

/// The organism's derived geometry over one matrix artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OrganismSpec {
    pub(crate) vocab: usize,
    pub(crate) hidden: usize,
    pub(crate) layers: usize,
    pub(crate) head_dim: usize,
    pub(crate) intermediate: usize,
    pub(crate) key_head_dim: usize,
    pub(crate) value_head_dim: usize,
    pub(crate) conv_kernel: usize,
}

impl OrganismSpec {
    /// Derive the organism from the matrix geometry: intermediate at
    /// 2.75x hidden, GDN head dims at the hybrid's 3/4 and 3/2 of the
    /// attention head dim, conv kernel 4.
    pub(crate) fn derive(vocab: usize, hidden: usize, layers: usize) -> BiquestResult<Self> {
        snafu::ensure_whatever!(
            hidden.is_multiple_of(MATRIX_HEADS),
            "hidden {hidden} is not a multiple of the ruled {MATRIX_HEADS} heads"
        );
        let head_dim = hidden / MATRIX_HEADS;
        snafu::ensure_whatever!(
            head_dim.is_multiple_of(4),
            "head_dim {head_dim} must divide by 4 for the GDN 3/4 and 3/2 ratios"
        );
        snafu::ensure_whatever!(
            layers >= LAYER_CYCLE && layers.is_multiple_of(LAYER_CYCLE),
            "the organism wants a multiple of {LAYER_CYCLE} layers (three GDN then \
             one attention), got {layers}"
        );
        Ok(Self {
            vocab,
            hidden,
            layers,
            head_dim,
            intermediate: hidden * 11 / 4,
            key_head_dim: head_dim * 3 / 4,
            value_head_dim: head_dim * 3 / 2,
            conv_kernel: 4,
        })
    }

    /// Whether a layer index is the cycle's attention slot.
    pub(crate) fn is_attention(&self, layer: usize) -> bool {
        layer % LAYER_CYCLE == LAYER_CYCLE - 1
    }

    pub(crate) fn key_width(&self) -> usize {
        MATRIX_HEADS * self.key_head_dim
    }

    pub(crate) fn value_width(&self) -> usize {
        MATRIX_HEADS * self.value_head_dim
    }

    /// The config.json both loaders read: NoPE attention (no
    /// rope_parameters), tied embeddings, negative eigenvalues
    /// allowed (the gated-delta convention). The eos and pad ids sit
    /// at NULL (0x00) until the organism's stop story is designed.
    pub(crate) fn config_json(&self) -> serde_json::Value {
        let layer_types: Vec<&str> = (0..self.layers)
            .map(|layer| {
                if self.is_attention(layer) {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        serde_json::json!({
            "vocab_size": self.vocab,
            "hidden_size": self.hidden,
            "intermediate_size": self.intermediate,
            "num_hidden_layers": self.layers,
            "num_attention_heads": MATRIX_HEADS,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "max_position_embeddings": 65536,
            "layer_types": layer_types,
            "tie_word_embeddings": true,
            "linear_num_key_heads": MATRIX_HEADS,
            "linear_num_value_heads": MATRIX_HEADS,
            "linear_key_head_dim": self.key_head_dim,
            "linear_value_head_dim": self.value_head_dim,
            "linear_conv_kernel_dim": self.conv_kernel,
            "linear_allow_neg_eigval": true,
            "eos_token_id": 0,
            "pad_token_id": 0,
        })
    }
}

/// One standard-normal draw, Box-Muller over SplitMix64.
fn normal_draw(rng: &mut llm::SplitMix64) -> f64 {
    let mut first_unit = rng.next_unit();
    if first_unit <= f64::MIN_POSITIVE {
        first_unit = f64::MIN_POSITIVE;
    }
    let second_unit = rng.next_unit();
    (-2.0 * first_unit.ln()).sqrt() * (2.0 * std::f64::consts::PI * second_unit).cos()
}

fn bf16_bytes_of(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values
        .flat_map(|value| half::bf16::from_f32(value).to_le_bytes())
        .collect()
}

/// A seeded gaussian tensor at the linear-init scale, as bf16 bytes.
fn seeded_linear(rng: &mut llm::SplitMix64, count: usize) -> Vec<u8> {
    bf16_bytes_of((0..count).map(|_| (normal_draw(rng) * INIT_SIGMA) as f32))
}

/// An all-ones norm scale, as bf16 bytes.
fn ones(count: usize) -> Vec<u8> {
    bf16_bytes_of((0..count).map(|_| 1.0f32))
}

/// A_log seeded ln(U(1, 16)): decay rates spread across the gated
/// delta rule's useful range, the DeltaNet convention.
fn seeded_a_log(rng: &mut llm::SplitMix64, count: usize) -> Vec<u8> {
    bf16_bytes_of((0..count).map(|_| {
        let a = 1.0 + rng.next_unit() * 15.0;
        a.ln() as f32
    }))
}

/// dt_bias seeded softplus-inverse of log-uniform dt in (0.001, 0.1),
/// the Mamba-family convention the gated rule inherits.
fn seeded_dt_bias(rng: &mut llm::SplitMix64, count: usize) -> Vec<u8> {
    bf16_bytes_of((0..count).map(|_| {
        let exponent = rng.next_unit() * 2.0 - 3.0;
        let dt = 10f64.powf(exponent);
        (dt.exp_m1()).ln() as f32
    }))
}

/// The matrix artifact's embedding: bf16 bytes passed through
/// unconverted, plus the vocab and hidden its metadata declares.
fn read_matrix(path: &Path) -> BiquestResult<(usize, usize, Vec<u8>)> {
    let bytes = fs::read(path)?;
    let parsed = match safetensors::SafeTensors::deserialize(&bytes) {
        Ok(parsed) => parsed,
        Err(e) => snafu::whatever!("matrix artifact parse failed ({}): {e}", path.display()),
    };
    let Ok(view) = parsed.tensor(EMBED_TENSOR) else {
        snafu::whatever!("the matrix artifact carries no {EMBED_TENSOR}");
    };
    snafu::ensure_whatever!(
        view.dtype() == safetensors::Dtype::BF16 && view.shape().len() == 2,
        "the matrix artifact wants bf16 rank 2, got {:?} {:?}",
        view.dtype(),
        view.shape()
    );
    Ok((view.shape()[0], view.shape()[1], view.data().to_vec()))
}

/// `biquest trainer init`: the organism's fresh full checkpoint from
/// the ImagineQuestMatrix artifact - config.json plus a seeded
/// model.safetensors, validated through the llm loader before return.
pub(crate) fn trainer_init(args: &TrainerInitArgs) -> BiquestResult<()> {
    let started = std::time::Instant::now();
    let (vocab, hidden, embed_bytes) = read_matrix(&args.matrix)?;
    let spec = OrganismSpec::derive(vocab, hidden, args.layers)?;
    let mut rng = llm::SplitMix64::new(args.seed);

    let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    tensors.push((
        String::from(EMBED_TENSOR),
        vec![spec.vocab, spec.hidden],
        embed_bytes.clone(),
    ));
    // The candle engine reads lm_head.weight unconditionally, so the
    // tied head writes under both names.
    tensors.push((
        String::from("lm_head.weight"),
        vec![spec.vocab, spec.hidden],
        embed_bytes,
    ));
    for layer in 0..spec.layers {
        let prefix = format!("model.layers.{layer}");
        let hidden = spec.hidden;
        let intermediate = spec.intermediate;
        let mut push = |name: String, shape: Vec<usize>, bytes: Vec<u8>| {
            tensors.push((name, shape, bytes));
        };
        if spec.is_attention(layer) {
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                push(
                    format!("{prefix}.self_attn.{name}.weight"),
                    vec![hidden, hidden],
                    seeded_linear(&mut rng, hidden * hidden),
                );
            }
            for name in ["q_norm", "k_norm"] {
                push(
                    format!("{prefix}.self_attn.{name}.weight"),
                    vec![hidden],
                    ones(hidden),
                );
            }
            for name in ["post_attention_layernorm", "post_feedforward_layernorm"] {
                push(format!("{prefix}.{name}.weight"), vec![hidden], ones(hidden));
            }
        } else {
            let key_width = spec.key_width();
            let value_width = spec.value_width();
            for name in ["input_layernorm", "post_attention_layernorm"] {
                push(format!("{prefix}.{name}.weight"), vec![hidden], ones(hidden));
            }
            for name in ["q_proj", "k_proj"] {
                push(
                    format!("{prefix}.linear_attn.{name}.weight"),
                    vec![key_width, hidden],
                    seeded_linear(&mut rng, key_width * hidden),
                );
            }
            for name in ["v_proj", "g_proj"] {
                push(
                    format!("{prefix}.linear_attn.{name}.weight"),
                    vec![value_width, hidden],
                    seeded_linear(&mut rng, value_width * hidden),
                );
            }
            push(
                format!("{prefix}.linear_attn.o_proj.weight"),
                vec![hidden, value_width],
                seeded_linear(&mut rng, hidden * value_width),
            );
            for name in ["a_proj", "b_proj"] {
                push(
                    format!("{prefix}.linear_attn.{name}.weight"),
                    vec![MATRIX_HEADS, hidden],
                    seeded_linear(&mut rng, MATRIX_HEADS * hidden),
                );
            }
            for (name, channels) in [
                ("q_conv1d", key_width),
                ("k_conv1d", key_width),
                ("v_conv1d", value_width),
            ] {
                push(
                    format!("{prefix}.linear_attn.{name}.weight"),
                    vec![channels, 1, spec.conv_kernel],
                    seeded_linear(&mut rng, channels * spec.conv_kernel),
                );
            }
            push(
                format!("{prefix}.linear_attn.A_log"),
                vec![MATRIX_HEADS],
                seeded_a_log(&mut rng, MATRIX_HEADS),
            );
            push(
                format!("{prefix}.linear_attn.dt_bias"),
                vec![MATRIX_HEADS],
                seeded_dt_bias(&mut rng, MATRIX_HEADS),
            );
            push(
                format!("{prefix}.linear_attn.o_norm.weight"),
                vec![spec.value_head_dim],
                ones(spec.value_head_dim),
            );
        }
        for name in ["gate_proj", "up_proj"] {
            tensors.push((
                format!("{prefix}.mlp.{name}.weight"),
                vec![intermediate, hidden],
                seeded_linear(&mut rng, intermediate * hidden),
            ));
        }
        tensors.push((
            format!("{prefix}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
            seeded_linear(&mut rng, hidden * intermediate),
        ));
    }
    tensors.push((
        String::from("model.norm.weight"),
        vec![spec.hidden],
        ones(spec.hidden),
    ));

    let parameter_count: usize = tensors
        .iter()
        .map(|(_, shape, _)| shape.iter().product::<usize>())
        .sum::<usize>()
        - spec.vocab * spec.hidden; // the tied head counts once
    let views: Vec<(String, safetensors::tensor::TensorView)> = tensors
        .iter()
        .map(|(name, shape, bytes)| {
            match safetensors::tensor::TensorView::new(
                safetensors::Dtype::BF16,
                shape.clone(),
                bytes,
            ) {
                Ok(view) => Ok((name.clone(), view)),
                Err(e) => snafu::whatever!("tensor view {name} failed: {e}"),
            }
        })
        .collect::<BiquestResult<Vec<_>>>()?;
    let mut metadata: HashMap<String, String> = HashMap::new();
    metadata.insert(String::from("kind"), String::from("imagine_quest_organism"));
    metadata.insert(String::from("matrix"), args.matrix.display().to_string());
    metadata.insert(String::from("seed"), args.seed.to_string());
    metadata.insert(String::from("tool_version"), String::from(env!("CARGO_PKG_VERSION")));

    fs::create_dir_all(&args.out)?;
    let config = spec.config_json();
    fs::write(
        args.out.join("config.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    match safetensors::serialize_to_file(
        views,
        Some(metadata),
        &args.out.join("model.safetensors"),
    ) {
        Ok(()) => {}
        Err(e) => snafu::whatever!("checkpoint write failed: {e}"),
    }

    // The llm loader is the acceptance check: parse and validate the
    // organism exactly as the engine would.
    let validated = llm::load_config(&args.out)?;
    snafu::ensure_whatever!(
        validated.vocab_size == spec.vocab && validated.hidden_size == spec.hidden,
        "the written config re-reads differently ({} x {})",
        validated.vocab_size,
        validated.hidden_size
    );

    let provenance = harness::nu::Value::record(
        harness::nu::record! {
            "matrix" => v_str(&args.matrix.display().to_string()),
            "vocab_size" => v_int(spec.vocab as i64),
            "hidden_size" => v_int(spec.hidden as i64),
            "layers" => v_int(spec.layers as i64),
            "attention_layers" => v_int((0..spec.layers).filter(|&l| spec.is_attention(l)).count() as i64),
            "intermediate_size" => v_int(spec.intermediate as i64),
            "key_head_dim" => v_int(spec.key_head_dim as i64),
            "value_head_dim" => v_int(spec.value_head_dim as i64),
            "parameters" => v_int(parameter_count as i64),
            "seed" => v_int(args.seed as i64),
            "biquest_version" => v_str(env!("CARGO_PKG_VERSION")),
            "built_at" => v_int(epoch_seconds()),
        },
        span(),
    );
    harness::nu::save_value(&args.out.join("provenance.nuon"), &provenance)?;

    let summary = harness::nu::Value::record(
        harness::nu::record! {
            "vocab_size" => v_int(spec.vocab as i64),
            "hidden_size" => v_int(spec.hidden as i64),
            "layers" => v_int(spec.layers as i64),
            "parameters" => v_int(parameter_count as i64),
            "out" => v_str(&args.out.display().to_string()),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", harness::nu::to_nuon_text(&summary)?);
    Ok(())
}
