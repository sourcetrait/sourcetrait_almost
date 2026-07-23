pub const DEFAULT_MODEL_ID: &str = "allenai/Olmo-3-7B-Instruct";

/// Stop strings resolved to ids at runtime; matches the checkpoint's
/// generation_config eos set (<|im_end|> = 100265, <|endoftext|> = 100257).
pub const STOP_TOKENS: [&str; 2] = ["<|im_end|>", "<|endoftext|>"];

/// Pinned sampling defaults from the checkpoint's generation_config.json.
pub const DEFAULT_TEMPERATURE: f64 = 0.6;
pub const DEFAULT_TOP_P: f64 = 0.95;

/// Decode budget default: the checkpoint's generation_config
/// max_new_tokens (the model card's recommendation). A budget, not a
/// reservation - caches grow with the actual decode, the graph arm
/// pre-pays only a reserve-sized headroom, and generation clamps the
/// effective budget to the position ceiling at start.
pub const DEFAULT_SAMPLE_LEN: usize = 32768;

/// Eager prompt-prefill chunk (tokens per forward). Bounds the eager
/// attention transient - scores plus the f32-softmax parity copies scale
/// with chunk * context: ~1.3 GiB at 32K bf16, which beside the weights
/// (13.6 GiB), the trimmed KV (~5.5 GiB at 32K), the cat-growth double
/// buffer, and a display-loaded card still fits Tier A. Chunked prefill
/// is logit-exact vs single-shot (the battery pins it).
pub const PREFILL_CHUNK_EAGER: usize = 128;

/// Flash prompt-prefill chunk. Fused attention materializes no score
/// matrix, so the transient bound disappears and the chunk is sized for
/// throughput: each chunk re-streams the full weights, so bigger chunks
/// amortize weight traffic (swept on Tier A at 8K/32K). The 32K peak is
/// NOT chunk-bound in this regime - 1024 was measured to buy only
/// ~80 MiB for -12%/-18% prefill at 32K/8K; 512 is the emergency
/// contract knob (~-640 MiB at ~-18% 32K prefill) should A2's KV
/// reduction ever be unavailable.
pub const PREFILL_CHUNK_FLASH: usize = 2048;

/// Decode margin generate() adds to the known prompt length when
/// pre-reserving full-layer cache capacity; a longer decode grows
/// coarsely past the reserve.
pub const RESERVE_DECODE_MARGIN: usize = 256;
