"""Shared helpers for the lastmost pinned drivers.

Process-preparation MUST run before torch is imported anywhere, so this
module imports torch/transformers lazily inside functions only.
"""

import hashlib
import json
import os
import sys


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


def prepare_process(device, no_fla, determinism):
    """Env fixups that must precede the first torch import; returns the
    effective fla-block decision (cpu always blocks)."""
    effective_no_fla = no_fla or device == "cpu"
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if determinism:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return effective_no_fla


def apply_fla_block():
    """Patch fla availability before the lazy model import binds it."""
    import transformers.utils.import_utils as tf_import_utils

    tf_import_utils.is_flash_linear_attention_available = lambda: False


def set_determinism(seed, determinism):
    import torch

    torch.manual_seed(seed)
    if not determinism:
        return "off"
    try:
        torch.use_deterministic_algorithms(True)
        return "on"
    except Exception as e:  # noqa: BLE001 - record, do not die
        return f"failed: {e}"


def load_model(model_dir, dtype_name, attn, device):
    """Load tokenizer + model onto the device; returns (tokenizer, model,
    load_seconds)."""
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, attn_implementation=attn)
    model.to(device)
    model.eval()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    return tokenizer, model, time.perf_counter() - t0


def render_prompt(tokenizer, prompt_text, raw):
    """Returns (rendered_text, default_stops) or raises SystemExit(2) when
    chat mode lacks a template."""
    if raw:
        return prompt_text, ["<|endoftext|>"]
    if tokenizer.chat_template is None:
        emit("error", message="checkpoint has no chat template; use --raw")
        raise SystemExit(2)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return rendered, ["<|im_end|>", "<|endoftext|>"]


def resolve_stop_ids(tokenizer, stops):
    stop_ids = []
    for s in stops:
        tid = tokenizer.convert_tokens_to_ids(s)
        if tid is not None and tid >= 0:
            stop_ids.append(tid)
    return stop_ids
