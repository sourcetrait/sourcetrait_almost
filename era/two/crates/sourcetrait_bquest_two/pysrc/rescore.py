"""Offline rescore: score engine predictions with olmo-eval's own task
scorers (the Seam-A path: rebuild Instance/Request via the task registry
with the render's config overrides, attach the engine's outputs, run
score_responses + compute_metrics). CPU-only; runs in the capability
eval env.

Manifest (JSON list) rows: {"spec": str, "overrides": [str, ...]} where
overrides are the TASK-side dotlist strings the render used (limit=N,
data_source=...); sampling-side keys (max_tokens, ...) must be omitted
- they are not TaskConfig fields and replace() would reject them.

Usage:
  python rescore.py --manifest m.json --predictions-root <run-dir> \
      --out <metrics-out.json>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any

import olmo_eval.evals  # noqa: F401  (task/suite registration)
import olmo_eval.evals.tasks  # noqa: F401
from olmo_eval.cli.run.config import _apply_dotlist_overrides
from olmo_eval.common.types import LMOutput, Response
from olmo_eval.evals.tasks.common import get_task


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sanitize(spec: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in spec)


def find_predictions(root: Path, spec: str) -> Path:
    pattern = f"{sanitize(spec)}_*-predictions.jsonl"
    hits = sorted((root / "predictions").rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"no predictions match {pattern} under {root}")
    if len(hits) > 1:
        raise RuntimeError(f"ambiguous predictions for {spec}: {hits}")
    return hits[0]


async def rescore_task(
    spec: str, overrides: list[str], predictions_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    override_dict: dict[str, Any] = {}
    if overrides:
        _apply_dotlist_overrides(override_dict, overrides)
    task = get_task(spec, config_overrides=override_dict or None)
    instances = list(task.instances)
    # The runner-layer seeded subsample (runners/asynq/preparation.py):
    # doc_id is the index into the SAMPLED order.
    if task.config.limit and len(instances) > task.config.limit:
        rng = random.Random(task.config.seed)
        instances = rng.sample(instances, task.config.limit)
    rows = sorted(load_jsonl(predictions_path), key=lambda r: r["doc_id"])
    if len(rows) != len(instances):
        raise RuntimeError(
            f"{spec}: {len(rows)} prediction rows vs {len(instances)} instances"
        )

    responses = []
    for index, (row, instance) in enumerate(zip(rows, instances)):
        if row["doc_id"] != index:
            raise RuntimeError(f"{spec}: doc_id {row['doc_id']} at position {index}")
        request = task.format_request(instance=instance)
        outputs = []
        for out in row["model_output"]:
            sum_logits = out.get("sum_logits")
            num_tokens = out.get("num_tokens") or 1
            logprobs = None
            metadata: dict[str, Any] = {}
            if sum_logits is not None:
                logprobs = [{"token": "", "logprob": float(sum_logits)}]
                logprobs += [{"token": "", "logprob": 0.0}] * (int(num_tokens) - 1)
                metadata["total_logprob"] = float(sum_logits)
            outputs.append(
                LMOutput(text=out["text"], logprobs=logprobs, metadata=metadata)
            )
        responses.append(Response(instance=instance, request=request, outputs=outputs))

    scored = await task.score_responses(responses=responses)
    metrics = task.compute_metrics(responses=scored)

    per_item = []
    for index, response in enumerate(scored):
        scores = getattr(response, "scores", None)
        per_item.append(
            {
                "doc_id": index,
                "scores": scores if isinstance(scores, dict) else None,
                "extracted": [
                    getattr(output, "extracted_answer", None)
                    for output in response.outputs
                ],
            }
        )
    return {"metrics": metrics, "num_instances": len(instances)}, per_item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    results: dict[str, Any] = {}
    scores_dir = args.predictions_root / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest:
        spec = entry["spec"]
        overrides = entry.get("overrides", [])
        predictions_path = find_predictions(args.predictions_root, spec)
        summary, per_item = asyncio.run(
            rescore_task(spec, overrides, predictions_path)
        )
        results[spec] = summary
        with (scores_dir / f"{sanitize(spec)}-scores.jsonl").open("w") as handle:
            for row in per_item:
                handle.write(json.dumps(row) + "\n")
        print(f"rescored {spec}: {summary['metrics']}")

    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
