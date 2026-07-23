"""refquest env driver: verify the pinned environment; drift = exit 1.

Asserts the ai2 hybrid-era pin trio (the T2 ruling), the python minor,
and cuda visibility; reports everything it read. One JSON event line on
stdout.
"""

import json
import sys
from importlib import metadata

PINS = {
    "transformers": "5.4.0",
    "vllm": "0.19.1",
    "flash-linear-attention": "0.5.0",
}


def main():
    report = {}
    drift = []
    for dist, expected in PINS.items():
        try:
            version = metadata.version(dist)
        except metadata.PackageNotFoundError:
            version = None
        report[dist] = version
        if version != expected:
            drift.append(f"{dist}: {version} != {expected}")

    python_version = sys.version.split()[0]
    report["python"] = python_version
    if not python_version.startswith("3.12."):
        drift.append(f"python: {python_version} != 3.12.*")

    import torch

    report["torch"] = metadata.version("torch")
    report["triton"] = metadata.version("triton")
    report["accelerate"] = metadata.version("accelerate")
    report["cuda_available"] = torch.cuda.is_available()
    report["device"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )
    if not report["cuda_available"]:
        drift.append("cuda unavailable")

    print(
        json.dumps(
            {"event": "env", "ok": not drift, "drift": drift, "report": report},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if not drift else 1


if __name__ == "__main__":
    sys.exit(main())
