//! The eval verb: evaluation batteries through the original stack.
use crate::*;

/// Dispatch `refquest eval <suite>`.
pub(crate) fn eval(args: &EvalArgs) -> RefquestResult<()> {
    match &args.suite {
        EvalSuite::Needle(needle_args) => eval_needle(needle_args),
    }
}

/// One needle-battery length batch, routed by backend.
fn eval_needle(args: &NeedleArgs) -> RefquestResult<()> {
    let model_dir = match &args.model_dir {
        Some(dir) => dir.clone(),
        None => resolve_model_dir(args.model)?,
    };
    let python = env_python()?;
    let pysrc = materialize_pysrc()?;

    let mut cmd = process::Command::new(&python);
    match args.backend {
        BackendPick::Hf => {
            cmd.arg(pysrc.join("needle.py"))
                .arg("--model-dir")
                .arg(&model_dir)
                .arg("--spec")
                .arg(&args.spec)
                .arg("--length")
                .arg(args.length.to_string())
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
            if args.no_fla || args.device == DevicePick::Cpu {
                cmd.arg("--no-fla");
            }
            if args.device == DevicePick::Cpu {
                cmd.env("CUDA_VISIBLE_DEVICES", "");
            }
        }
        BackendPick::Vllm => {
            cmd.arg(pysrc.join("needle_vllm.py"))
                .arg("--model-dir")
                .arg(&model_dir)
                .arg("--spec")
                .arg(&args.spec)
                .arg("--length")
                .arg(args.length.to_string())
                .arg("--seed")
                .arg(args.seed.to_string())
                .arg("--gpu-mem-util")
                .arg(args.gpu_mem_util.to_string());
            if args.enforce_eager {
                cmd.arg("--enforce-eager");
            }
            if args.allow_long {
                cmd.env("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1");
            }
        }
    }

    run_driver(cmd, args.record.as_deref())
}
