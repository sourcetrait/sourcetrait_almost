"""lastmost needle battery driver: retrieval grid at one context length.

One invocation = one length batch: the model loads once, then every
(key x depth) cell from the spec runs greedy through the chat template.
A cell passes when the needle's value string appears in the generated
text. Emits one `cell` event per cell and a final `needle_summary`.
"""

import argparse
import json
import sys
import time

import common


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="lastmost-needle", description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--length", type=int, required=True, help="target total prompt tokens")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    ap.add_argument("--attn", choices=("eager", "sdpa"), default="sdpa")
    ap.add_argument("--no-fla", action="store_true")
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

    no_fla = common.prepare_process(args.device, args.no_fla, False)

    import torch

    if no_fla:
        common.apply_fla_block()
    common.set_determinism(args.seed, False)

    tokenizer, model, load_s = common.load_model(args.model_dir, args.dtype, args.attn, args.device)
    device = args.device

    def render(content):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def encode(text):
        return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids

    stop_ids = common.resolve_stop_ids(tokenizer, ["<|im_end|>", "<|endoftext|>"])

    target = args.length
    base_words = lcg_words(spec["words"], max(4096, target * 2), spec["filler_seed"] + target)
    probe = " ".join(base_words[:2000])
    ratio = encode(probe).shape[1] / 2000.0

    common.emit(
        "meta",
        versions=common.package_versions(),
        model_dir=args.model_dir,
        file_stamps=common.small_file_stamps(args.model_dir),
        device=device,
        dtype=args.dtype,
        attn=args.attn,
        fla_blocked=no_fla,
        length=target,
        load_s=round(load_s, 3),
        tokens_per_word=round(ratio, 4),
        spec_keys=[k["name"] for k in spec["keys"]],
        depths=spec["depths"],
        gen_tokens=spec["gen_tokens"],
        argv=sys.argv[1:],
    )

    results = []
    for key in spec["keys"]:
        for depth in spec["depths"]:
            needle = spec["needle_template"].format(key=key["name"], value=key["value"])
            question = spec["question_template"].format(key=key["name"])
            overhead = encode(render(needle + "\n\n" + question)).shape[1]
            count = max(64, int((target - overhead) / ratio))

            input_ids = None
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
                input_ids = encode(render(content))
                actual = input_ids.shape[1]
                if abs(actual - target) <= max(8, target // 200):
                    break
                count += int((target - actual) / ratio)

            input_ids = input_ids.to(device)
            generated = []
            with torch.inference_mode():
                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
                if device == "cuda":
                    torch.cuda.synchronize()
                prefill_s = time.perf_counter() - t1

                cache = out.past_key_values
                next_id = int(torch.argmax(out.logits[0, -1], dim=-1).item())
                t2 = time.perf_counter()
                while True:
                    if next_id in stop_ids:
                        break
                    generated.append(next_id)
                    if len(generated) >= spec["gen_tokens"]:
                        break
                    step = torch.tensor([[next_id]], dtype=torch.long, device=device)
                    out = model(input_ids=step, past_key_values=cache, use_cache=True)
                    cache = out.past_key_values
                    next_id = int(torch.argmax(out.logits[0, -1], dim=-1).item())
                decode_s = time.perf_counter() - t2

            text = tokenizer.decode(generated, skip_special_tokens=True)
            found = key["value"] in text
            results.append({"key": key["name"], "depth": depth, "found": found})
            common.emit(
                "cell",
                key=key["name"],
                depth=depth,
                target_tokens=target,
                actual_tokens=int(actual),
                found=found,
                text=text,
                prefill_s=round(prefill_s, 3),
                decode_s=round(decode_s, 3),
            )

            del cache, out, input_ids
            if device == "cuda":
                torch.cuda.empty_cache()

    summary = {
        "length": target,
        "total": len(results),
        "found": sum(1 for r in results if r["found"]),
        "misses": [r for r in results if not r["found"]],
    }
    if device == "cuda":
        summary["max_allocated"] = int(torch.cuda.max_memory_allocated())
        summary["max_reserved"] = int(torch.cuda.max_memory_reserved())
    common.emit("needle_summary", **summary)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
