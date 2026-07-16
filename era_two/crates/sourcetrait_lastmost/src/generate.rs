//! The generate verb: spawn the pinned driver, split payload from events.
use crate::*;

/// Run `lastmost generate`: drives pysrc/generate.py through the pinned
/// environment's python. Driver stdout (JSON-lines events) becomes: text
/// payload on our stdout, terse summaries on stderr, and the full event
/// record at --record when given. Driver stderr passes through.
pub(crate) fn generate(args: &GenerateArgs) -> LastmostResult<()> {
    let model_dir = match &args.model_dir {
        Some(dir) => dir.clone(),
        None => resolve_model_dir(args.model)?,
    };
    let python = env_python()?;
    let driver = materialize_pysrc()?.join("generate.py");

    let mut cmd = process::Command::new(&python);
    cmd.arg(&driver)
        .arg("--model-dir")
        .arg(&model_dir)
        .arg("--prompt-file")
        .arg(&args.prompt_file)
        .arg("--max-new-tokens")
        .arg(args.max_new_tokens.to_string())
        .arg("--seed")
        .arg(args.seed.to_string())
        .arg("--device")
        .arg(match args.device {
            DevicePick::Cuda => "cuda",
            DevicePick::Cpu => "cpu",
        })
        .arg("--dtype")
        .arg(match args.dtype {
            DtypePick::Bf16 => "bf16",
            DtypePick::F32 => "f32",
        })
        .arg("--attn")
        .arg(match args.attn {
            AttnPick::Eager => "eager",
            AttnPick::Sdpa => "sdpa",
        });
    if args.raw {
        cmd.arg("--raw");
    }
    if args.sample {
        cmd.arg("--sample")
            .arg("--temperature")
            .arg(args.temperature.to_string())
            .arg("--top-p")
            .arg(args.top_p.to_string());
    }
    if args.no_fla || args.device == DevicePick::Cpu {
        cmd.arg("--no-fla");
    }
    if args.determinism {
        cmd.arg("--determinism");
    }
    if args.ignore_stops {
        cmd.arg("--ignore-stops");
    }
    for stop in &args.stop {
        cmd.arg("--stop").arg(stop);
    }
    if args.device == DevicePick::Cpu {
        // Keep the cpu grade genuinely cuda-free (fla gating + no context).
        cmd.env("CUDA_VISIBLE_DEVICES", "");
    }

    run_driver(cmd, args.record.as_deref())
}
