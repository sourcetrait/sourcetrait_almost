use crate::*;

/// camp: the interactive terminal chat (thin driver over the lib).
#[derive(Debug, clap::Parser)]
#[command(name = "camp", version, about)]
struct Cli {
    /// Config profile snake or TOML path (declared intent: model,
    /// device, dtype, generation, eviction, chat)
    #[arg(short = 'c', long)]
    config: Option<String>,
    /// Settings profile snake or TOML path (surgical runtime overrides;
    /// absent auto-pairs the config profile's name)
    #[arg(short = 's', long)]
    settings: Option<String>,
}

/// Binary entry: parse, run, exit non-zero on error. The TUI owns
/// stdout once the alternate screen opens; diagnostics before that go
/// to stderr.
pub fn run() {
    let cli = <Cli as clap::Parser>::parse();
    if let Err(error) = drive(cli) {
        eprintln!("camp: {error}");
        std::process::exit(1);
    }
}

fn drive(cli: Cli) -> lib::QuestResult<()> {
    let (config, profile) = lib::load_config(cli.config.as_deref())?;
    let mut settings = lib::load_settings(cli.settings.as_deref(), profile.as_deref(), &config)?;
    // Chained sessions regrow the kv buffers, which parks captured
    // graph buffers for near-zero replay benefit (phase D: +-1-3%);
    // chat therefore always runs capture-off.
    if settings.graph {
        eprintln!("camp: chained chat runs capture-off (graph profile setting ignored)");
        settings.graph = false;
    }
    let device = lib::resolve_device(&config)?;
    let dtype = lib::resolve_dtype(&config, &device);
    eprintln!(
        "camp: device {device:?}, dtype {dtype:?}{}",
        if settings.use_flash_attn { ", flash" } else { "" }
    );
    let dir = config
        .model_dir
        .clone()
        .unwrap_or_else(|| lib::default_model_dir(&config.model_id));
    let paths = lib::ensure_model(&config.model_id, &dir)?;
    let loaded = lib::load_model(&paths, &device, dtype, settings)?;
    eprintln!("camp: weights loaded in {:.1}s", loaded.load_seconds);
    chat::chat(loaded, &config)
}
