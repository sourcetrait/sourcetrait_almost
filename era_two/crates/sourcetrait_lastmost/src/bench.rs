//! The bench verb: one timed perf row per invocation.
use crate::*;

/// Run `lastmost bench`: hf rows ride the generate driver (its timing +
/// vram events ARE the row); vllm rows ride the bench_vllm driver.
pub(crate) fn bench(args: &BenchArgs) -> LastmostResult<()> {
    match args.backend {
        BackendPick::Hf => {
            let generate_args = GenerateArgs {
                model: args.model,
                model_dir: args.model_dir.clone(),
                prompt_file: args.prompt_file.clone(),
                raw: true,
                sample: false,
                temperature: 0.6,
                top_p: 0.95,
                seed: args.seed,
                max_new_tokens: args.max_new_tokens,
                device: args.device,
                dtype: args.dtype,
                attn: args.attn,
                no_fla: args.no_fla,
                determinism: false,
                stop: Vec::new(),
                ignore_stops: true,
                record: args.record.clone(),
            };
            generate(&generate_args)
        }
        BackendPick::Vllm => {
            let model_dir = match &args.model_dir {
                Some(dir) => dir.clone(),
                None => resolve_model_dir(args.model)?,
            };
            let python = env_python()?;
            let driver = materialize_pysrc()?.join("bench_vllm.py");
            let mut cmd = process::Command::new(&python);
            cmd.arg(&driver)
                .arg("--model-dir")
                .arg(&model_dir)
                .arg("--prompt-file")
                .arg(&args.prompt_file)
                .arg("--max-new-tokens")
                .arg(args.max_new_tokens.to_string())
                .arg("--gpu-mem-util")
                .arg(args.gpu_mem_util.to_string())
                .arg("--seed")
                .arg(args.seed.to_string());
            if args.enforce_eager {
                cmd.arg("--enforce-eager");
            }
            run_driver(cmd, args.record.as_deref())
        }
    }
}
