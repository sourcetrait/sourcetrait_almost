"""lastmost generate driver: original-stack (transformers) generation.

Pinned script shipped with the sourcetrait_lastmost crate and materialized
into the lastmost data home; invoked by the `lastmost generate` verb through
the pinned uv environment's python. The python here is the subject under
test (ai2's original tooling), not authored product logic.

Contract: stdout carries JSON-lines events only (meta / timing / vram /
text); stderr carries free-form diagnostics from the underlying stack.
Exit code 0 on a completed generation.
"""

import argparse
import hashlib
import json
import os
import sys
import time


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def small_file_stamps(model_dir):
    stamps = {}
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            stamps[name] = sha256_file(path)
    shard = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(shard):
        st = os.stat(shard)
        stamps["model.safetensors"] = {"bytes": st.st_size, "mtime": int(st.st_mtime)}
    return stamps


def package_versions():
    from importlib import metadata

    versions = {"python": sys.version.split()[0]}
    for dist in ("torch", "transformers", "flash-linear-attention", "triton", "accelerate", "vllm"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    return versions


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="lastmost-generate", description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--raw", action="store_true", help="no chat template; pure continuation")
    ap.add_argument("--sample", action="store_true", help="sampled decode (default greedy)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    ap.add_argument("--attn", choices=("eager", "sdpa"), default="eager")
    ap.add_argument("--no-fla", action="store_true", help="force the in-tree torch GDN paths")
    ap.add_argument("--determinism", action="store_true", help="torch.use_deterministic_algorithms(True)")
    ap.add_argument("--stop", action="append", default=None, help="stop token string (repeatable)")
    return ap.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    # The cpu grade must never touch cuda: fla availability is gated on
    # torch-cuda visibility, and the fla norm module constructs itself on
    # the current cuda device at model init.
    no_fla = args.no_fla or args.device == "cpu"
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.determinism:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    import transformers.utils.import_utils as tf_import_utils

    if no_fla:
        tf_import_utils.is_flash_linear_attention_available = lambda: False

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    determinism_state = "off"
    if args.determinism:
        try:
            torch.use_deterministic_algorithms(True)
            determinism_state = "on"
        except Exception as e:  # noqa: BLE001 - record, do not die
            determinism_state = f"failed: {e}"

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = args.device

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=dtype, attn_implementation=args.attn
    )
    model.to(device)
    model.eval()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    load_s = time.perf_counter() - t0

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    if args.raw:
        rendered = prompt_text
        default_stops = ["<|endoftext|>"]
    else:
        if tokenizer.chat_template is None:
            emit("error", message="checkpoint has no chat template; use --raw")
            return 2
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        default_stops = ["<|im_end|>", "<|endoftext|>"]

    stops = args.stop if args.stop else default_stops
    stop_ids = []
    for s in stops:
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is not None and tid >= 0:
            stop_ids.append(tid)

    input_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    emit(
        "meta",
        versions=package_versions(),
        model_dir=args.model_dir,
        file_stamps=small_file_stamps(args.model_dir),
        device=device,
        dtype=args.dtype,
        attn=args.attn,
        fla_blocked=no_fla,
        determinism=determinism_state,
        sampling={
            "mode": "sample" if args.sample else "greedy",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        template="raw" if args.raw else "chat",
        rendered_prompt=rendered,
        prompt_tokens=int(input_ids.shape[1]),
        stops=stops,
        stop_ids=stop_ids,
        config_eos=getattr(model.generation_config, "eos_token_id", None),
        argv=sys.argv[1:],
    )

    generator = None
    if args.sample:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)

    def pick_next(logits):
        if not args.sample:
            return int(torch.argmax(logits, dim=-1).item())
        probs = torch.softmax(logits.float() / args.temperature, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum - sorted_probs < args.top_p
        keep[..., 0] = True
        filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        filtered = filtered / filtered.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(filtered, 1, generator=generator)
        return int(sorted_idx.gather(-1, choice).item())

    generated = []
    stopped_on = None
    with torch.inference_mode():
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        if device == "cuda":
            torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t1

        cache = out.past_key_values
        next_id = pick_next(out.logits[0, -1])

        t2 = time.perf_counter()
        steps = 0
        while True:
            if next_id in stop_ids:
                stopped_on = next_id
                break
            generated.append(next_id)
            if len(generated) >= args.max_new_tokens:
                break
            step_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
            out = model(input_ids=step_ids, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_id = pick_next(out.logits[0, -1])
            steps += 1
        if device == "cuda":
            torch.cuda.synchronize()
        decode_s = time.perf_counter() - t2

    prompt_tokens = int(input_ids.shape[1])
    emit(
        "timing",
        load_s=round(load_s, 3),
        prefill_s=round(prefill_s, 4),
        prefill_tok_s=round(prompt_tokens / prefill_s, 2) if prefill_s > 0 else None,
        decode_s=round(decode_s, 4),
        decode_steps=steps,
        decode_tok_s=round(steps / decode_s, 2) if decode_s > 0 and steps > 0 else None,
    )
    if device == "cuda":
        emit(
            "vram",
            allocated_after_load=None,
            max_allocated=int(torch.cuda.max_memory_allocated()),
            max_reserved=int(torch.cuda.max_memory_reserved()),
        )

    emit(
        "text",
        text=tokenizer.decode(generated, skip_special_tokens=True),
        ids=generated,
        stopped_on=stopped_on,
        stop_reason="stop_token" if stopped_on is not None else "max_new_tokens",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
