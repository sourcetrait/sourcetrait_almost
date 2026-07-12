pub const DEFAULT_MODEL_ID: &str = "allenai/Olmo-3-7B-Instruct";

/// Stop strings resolved to ids at runtime; matches the checkpoint's
/// generation_config eos set (<|im_end|> = 100265, <|endoftext|> = 100257).
pub const STOP_TOKENS: [&str; 2] = ["<|im_end|>", "<|endoftext|>"];

/// Pinned sampling defaults from the checkpoint's generation_config.json.
pub const DEFAULT_TEMPERATURE: f64 = 0.6;
pub const DEFAULT_TOP_P: f64 = 0.95;

/// Prompt prefill chunk (tokens per forward). Bounds the eager attention
/// transient - scores plus the f32-softmax parity copies scale with
/// chunk * context: ~1.3 GiB at 32K bf16, which beside the weights
/// (13.6 GiB), the trimmed KV (~5.5 GiB at 32K), the cat-growth double
/// buffer, and a display-loaded card still fits Tier A. Chunked prefill
/// is logit-exact vs single-shot (the battery pins it).
pub const PREFILL_CHUNK: usize = 128;
