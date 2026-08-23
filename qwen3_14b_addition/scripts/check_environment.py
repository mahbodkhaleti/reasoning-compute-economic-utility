#!/usr/bin/env python3
"""Fail fast when the execution environment is unsuitable for the pinned run."""
from __future__ import annotations

import importlib.metadata
import json
import platform

import torch


def main() -> None:
    errors: list[str] = []
    if not torch.cuda.is_available():
        errors.append("CUDA is unavailable")
        devices = []
    else:
        devices = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_gib": torch.cuda.get_device_properties(index).total_memory / 2**30,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
            for index in range(torch.cuda.device_count())
        ]
        if devices[0]["memory_gib"] < 39.0:
            errors.append("GPU 0 has less than 39 GiB; Qwen3-14B BF16 may not fit")
        if not devices[0]["bf16_supported"]:
            errors.append("GPU 0 does not report BF16 support")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "accelerate", "pandas", "numpy")
    }
    expected = {"torch": "2.13.0", "transformers": "5.15.0", "accelerate": "1.14.0"}
    for name, version in expected.items():
        if versions[name].split("+", 1)[0] != version:
            errors.append(f"{name}=={versions[name]} but the protocol requires {name}=={version}")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "python": platform.python_version(),
        "versions": versions,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": devices,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
