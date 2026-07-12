pub(crate) const DEFAULT_MODEL_ID: &str = "allenai/Olmo-3-7B-Instruct";

/// Stop strings resolved to ids at runtime; matches the checkpoint's
/// generation_config eos set (<|im_end|> = 100265, <|endoftext|> = 100257).
pub(crate) const STOP_TOKENS: [&str; 2] = ["<|im_end|>", "<|endoftext|>"];

/// Pinned sampling defaults from the checkpoint's generation_config.json.
pub(crate) const DEFAULT_TEMPERATURE: f64 = 0.6;
pub(crate) const DEFAULT_TOP_P: f64 = 0.95;
