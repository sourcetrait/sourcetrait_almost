"""refquest dump driver: all-position f32 logits in the era-one contract.

Writes a safetensors file holding:
- "logits"        f32 [rows, vocab] - one row per FED position of
                  [prompt_ids ++ fed_ids]; row r holds the logits that
                  predict token r+1;
- "prompt_ids"    u32 [P];
- "fed_ids"       u32 [F] (the generated tokens that were fed back, i.e.
                  all but the last generated token, or a replayed set);
- "generated_ids" u32 [G].

Modes:
- single:      one full forward over [prompt_ids ++ fed_ids] (the chunked
               GDN prefill path);
- incremental: prefill the prompt, then feed fed_ids one token at a time
               (the recurrent GDN decode path).

Fed source: --gen N (greedy self-generation) or --ids-from OTHER (replay
that dump's exact ids; required for cross-path/cross-device diffs).
"""

import argparse
import sys
import time

import common


def parse_args(argv):
    ap = argparse.ArgumentParser(prog="refquest-dump", description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-file")
    ap.add_argument("--ids-from", help="replay this dump's prompt+fed ids verbatim")
    ap.add_argument("--gen", type=int, default=64, help="greedy tokens to generate when not replaying")
    ap.add_argument("--mode", choices=("single", "incremental"), default="single")
    ap.add_argument("--raw", action="store_true", help="no chat template; pure continuation")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    ap.add_argument("--attn", choices=("eager", "sdpa"), default="eager")
    ap.add_argument("--no-fla", action="store_true")
    ap.add_argument("--determinism", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def all_rows_forward(model, ids, device, mode, torch):
    """Logits rows for every position of `ids`, by the requested path."""
    full = torch.tensor([ids], dtype=torch.long, device=device)
    if mode == "single":
        out = model(input_ids=full, use_cache=False, logits_to_keep=0)
        return out.logits[0].cpu().float()
    # incremental: one forward for position 0..0, then one per next token,
    # collecting the last row each time
    rows = []
    out = model(input_ids=full[:, :1], use_cache=True, logits_to_keep=0)
    cache = out.past_key_values
    rows.append(out.logits[0, -1].cpu().float())
    for i in range(1, len(ids)):
        out = model(
            input_ids=full[:, i : i + 1],
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        rows.append(out.logits[0, -1].cpu().float())
    return torch.stack(rows, dim=0)


def main(argv):
    args = parse_args(argv)
    if (args.prompt_file is None) == (args.ids_from is None):
        common.emit("error", message="exactly one of --prompt-file / --ids-from is required")
        return 2
    no_fla = common.prepare_process(args.device, args.no_fla, args.determinism)

    import torch

    if no_fla:
        common.apply_fla_block()
    determinism_state = common.set_determinism(args.seed, args.determinism)

    tokenizer, model, load_s = common.load_model(args.model_dir, args.dtype, args.attn, args.device)
    device = args.device

    if args.ids_from:
        from safetensors import safe_open

        with safe_open(args.ids_from, framework="pt") as f:
            prompt_ids = f.get_tensor("prompt_ids").tolist()
            fed_ids = f.get_tensor("fed_ids").tolist()
            generated_ids = f.get_tensor("generated_ids").tolist()
        template = "replay"
        rendered = None
    else:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()
        rendered, _stops = common.render_prompt(tokenizer, prompt_text, args.raw)
        prompt_ids = tokenizer(rendered, add_special_tokens=False).input_ids
        template = "raw" if args.raw else "chat"

        # Greedy self-generation to obtain fed/generated ids (era-one
        # discipline: diffs replay SAME ids; generation here is only the
        # id source).
        generated_ids = []
        with torch.inference_mode():
            ids_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            out = model(input_ids=ids_t, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            next_id = int(torch.argmax(out.logits[0, -1], dim=-1).item())
            for _ in range(args.gen):
                generated_ids.append(next_id)
                if len(generated_ids) >= args.gen:
                    break
                step = torch.tensor([[next_id]], dtype=torch.long, device=device)
                out = model(input_ids=step, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                next_id = int(torch.argmax(out.logits[0, -1], dim=-1).item())
        fed_ids = generated_ids[:-1]

    common.emit(
        "meta",
        versions=common.package_versions(),
        model_dir=args.model_dir,
        file_stamps=common.small_file_stamps(args.model_dir),
        device=device,
        dtype=args.dtype,
        attn=args.attn,
        fla_blocked=no_fla,
        determinism=determinism_state,
        mode=args.mode,
        template=template,
        ids_from=args.ids_from,
        prompt_tokens=len(prompt_ids),
        fed_tokens=len(fed_ids),
        load_s=round(load_s, 3),
        argv=sys.argv[1:],
    )

    all_ids = list(prompt_ids) + list(fed_ids)
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = all_rows_forward(model, all_ids, device, args.mode, torch)
    forward_s = time.perf_counter() - t0

    from safetensors.torch import save_file

    tensors = {
        "logits": logits.contiguous(),
        "prompt_ids": torch.tensor(prompt_ids, dtype=torch.uint32),
        "fed_ids": torch.tensor(fed_ids, dtype=torch.uint32),
        "generated_ids": torch.tensor(generated_ids, dtype=torch.uint32),
    }
    save_file(tensors, args.out)

    vram = {}
    if device == "cuda":
        vram = {
            "max_allocated": int(torch.cuda.max_memory_allocated()),
            "max_reserved": int(torch.cuda.max_memory_reserved()),
        }
    common.emit(
        "dump",
        out=args.out,
        rows=int(logits.shape[0]),
        vocab=int(logits.shape[1]),
        forward_s=round(forward_s, 3),
        rows_per_s=round(logits.shape[0] / forward_s, 2) if forward_s > 0 else None,
        **vram,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
