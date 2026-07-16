"""lastmost bench driver, vLLM backend: one timed row per invocation.

Single raw prompt, temperature 0, ignore_eos so decode runs the exact
requested length. Emits meta + bench events; prefill/decode split comes
from vllm's request metrics when present, else wall-clock totals only.
"""

import argparse
import json
import sys
import time

import common


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="lastmost-bench-vllm", description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--gpu-mem-util", type=float, default=0.82)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)

    t0 = time.perf_counter()
    llm = LLM(
        model=args.model_dir,
        dtype="bfloat16",
        mamba_ssm_cache_dtype="float32",
        max_model_len=prompt_tokens + args.max_new_tokens + 64,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        seed=args.seed,
    )
    load_s = time.perf_counter() - t0

    common.emit(
        "meta",
        versions=common.package_versions(),
        model_dir=args.model_dir,
        file_stamps=common.small_file_stamps(args.model_dir),
        backend="vllm",
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        load_s=round(load_s, 3),
        gpu_mem_util=args.gpu_mem_util,
        enforce_eager=args.enforce_eager,
        mamba_ssm_cache_dtype="float32",
        argv=sys.argv[1:],
    )

    # Two-pass split (prefix caching disabled, so both passes pay the
    # full prefill): pass 1 = prefill + 1 decode step; pass 2 = prefill
    # + N steps; decode = pass2 - pass1 over N-1 steps.
    probe = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, seed=args.seed)
    t1 = time.perf_counter()
    llm.generate([prompt], probe)
    prefill1_s = time.perf_counter() - t1

    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        ignore_eos=True,
        seed=args.seed,
    )
    t2 = time.perf_counter()
    outputs = llm.generate([prompt], params)
    full_s = time.perf_counter() - t2

    generated = len(outputs[0].outputs[0].token_ids)
    decode_s = max(full_s - prefill1_s, 1e-9)
    common.emit(
        "bench",
        backend="vllm",
        prompt_tokens=prompt_tokens,
        generated_tokens=generated,
        prefill_s=round(prefill1_s, 4),
        prefill_tok_s=round(prompt_tokens / prefill1_s, 2),
        full_s=round(full_s, 3),
        decode_s=round(decode_s, 4),
        decode_tok_s=round((generated - 1) / decode_s, 2) if generated > 1 else None,
        overall_tok_s=round(generated / full_s, 2) if full_s > 0 else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
