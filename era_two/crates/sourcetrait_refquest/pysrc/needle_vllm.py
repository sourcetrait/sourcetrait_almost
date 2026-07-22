"""refquest needle battery driver, vLLM backend (the production grade).

Same spec/cells as needle.py, driven through vllm's continuous-batching
engine: all cells of the length batch are submitted together and the
scheduler manages KV within its budget (the honest production posture).
Greedy via temperature 0; stop strings as in the hf driver.
"""

import argparse
import json
import sys
import time

import common


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="refquest-needle-vllm", description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--gpu-mem-util", type=float, default=0.82)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def lcg_words(words, count, seed):
    state = seed
    n = len(words)
    picked = []
    for _ in range(count):
        state = (state * 1103515245 + 12345) % 2147483648
        picked.append(words[state % n])
    return picked


def main(argv):
    args = parse_args(argv)
    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    def render(content):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def n_tokens(text):
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    target = args.length
    base_words = lcg_words(spec["words"], max(4096, target * 2), spec["filler_seed"] + target)
    ratio = n_tokens(" ".join(base_words[:2000])) / 2000.0

    cells = []
    prompts = []
    for key in spec["keys"]:
        for depth in spec["depths"]:
            needle = spec["needle_template"].format(key=key["name"], value=key["value"])
            question = spec["question_template"].format(key=key["name"])
            overhead = n_tokens(render(needle + "\n\n" + question))
            count = max(64, int((target - overhead) / ratio))
            rendered = None
            actual = 0
            for _ in range(4):
                words = base_words[:count]
                pre_n = max(1, min(len(words) - 1, int(len(words) * depth / 100)))
                content = (
                    " ".join(words[:pre_n])
                    + " "
                    + needle
                    + " "
                    + " ".join(words[pre_n:])
                    + "\n\n"
                    + question
                )
                rendered = render(content)
                actual = n_tokens(rendered)
                if abs(actual - target) <= max(8, target // 200):
                    break
                count += int((target - actual) / ratio)
            cells.append({"key": key["name"], "value": key["value"], "depth": depth, "actual_tokens": actual})
            prompts.append(rendered)

    t0 = time.perf_counter()
    llm = LLM(
        model=args.model_dir,
        dtype="bfloat16",
        mamba_ssm_cache_dtype="float32",
        max_model_len=target + spec["gen_tokens"] + 64,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    load_s = time.perf_counter() - t0

    common.emit(
        "meta",
        versions=common.package_versions(),
        model_dir=args.model_dir,
        file_stamps=common.small_file_stamps(args.model_dir),
        backend="vllm",
        length=target,
        load_s=round(load_s, 3),
        gpu_mem_util=args.gpu_mem_util,
        enforce_eager=args.enforce_eager,
        mamba_ssm_cache_dtype="float32",
        cells=len(cells),
        argv=sys.argv[1:],
    )

    params = SamplingParams(
        temperature=0.0,
        max_tokens=spec["gen_tokens"],
        stop=["<|im_end|>", "<|endoftext|>"],
        seed=args.seed,
    )
    t1 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    batch_s = time.perf_counter() - t1

    results = []
    for cell, output in zip(cells, outputs):
        text = output.outputs[0].text
        found = cell["value"] in text
        results.append({"key": cell["key"], "depth": cell["depth"], "found": found})
        common.emit(
            "cell",
            key=cell["key"],
            depth=cell["depth"],
            target_tokens=target,
            actual_tokens=cell["actual_tokens"],
            found=found,
            text=text,
        )

    common.emit(
        "needle_summary",
        length=target,
        total=len(results),
        found=sum(1 for r in results if r["found"]),
        misses=[r for r in results if not r["found"]],
        batch_s=round(batch_s, 3),
        backend="vllm",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
