//! The dump verb: all-position f32 logits in the era-one contract.
use crate::*;

/// Run `lastmost dump`: drives pysrc/dump.py through the pinned env.
pub(crate) fn dump(args: &DumpArgs) -> LastmostResult<()> {
    let model_dir = match &args.model_dir {
        Some(dir) => dir.clone(),
        None => resolve_model_dir(args.model)?,
    };
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    let python = env_python()?;
    let driver = materialize_pysrc()?.join("dump.py");

    let mut cmd = process::Command::new(&python);
    cmd.arg(&driver)
        .arg("--model-dir")
        .arg(&model_dir)
        .arg("--out")
        .arg(&args.out)
        .arg("--gen")
        .arg(args.gen_tokens.to_string())
        .arg("--seed")
        .arg(args.seed.to_string())
        .arg("--mode")
        .arg(match args.mode {
            ModePick::Single => "single",
            ModePick::Incremental => "incremental",
        })
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
    if let Some(prompt_file) = &args.prompt_file {
        cmd.arg("--prompt-file").arg(prompt_file);
    }
    if let Some(ids_from) = &args.ids_from {
        cmd.arg("--ids-from").arg(ids_from);
    }
    if args.raw {
        cmd.arg("--raw");
    }
    if args.no_fla || args.device == DevicePick::Cpu {
        cmd.arg("--no-fla");
    }
    if args.determinism {
        cmd.arg("--determinism");
    }
    if args.device == DevicePick::Cpu {
        cmd.env("CUDA_VISIBLE_DEVICES", "");
    }

    run_driver(cmd, args.record.as_deref())
}
