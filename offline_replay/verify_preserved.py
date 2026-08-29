#!/usr/bin/env python3
"""Verify completeness and basic integrity of preserved inference records."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


HISTORICAL_EXPECTED = {"no_think_greedy": 648, "reasoning_greedy": 4536, "reasoning_stochastic": 3240}
MODELS = ("phi4_mini_reasoning", "qwen3_4b", "deepseek_r1_llama_8b", "qwen3_8b", "qwen3_14b", "granite_3_2_8b_instruct")


def records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite(row: dict, keys: tuple[str, ...]) -> bool:
    return all(key in row and math.isfinite(float(row[key])) for key in keys)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    old = root / "robustness_extensions/server_outputs/raw"
    qwen = root / "preserved_outputs/qwen3_14b/historical/raw"
    granite = root / "preserved_outputs/granite_3_2_8b_instruct/historical/raw"
    synth_old = root / "preserved_outputs/synthetic_existing4/raw"
    synth_qwen = root / "preserved_outputs/qwen3_14b/synthetic/raw"
    synth_granite = root / "preserved_outputs/granite_3_2_8b_instruct/synthetic/raw"
    errors: list[str] = []
    manifest = json.loads((root / "docs/REPRODUCIBILITY_MANIFEST.json").read_text(encoding="utf-8"))
    if {item["slug"] for item in manifest["models"]} != set(MODELS):
        errors.append("reproducibility manifest does not contain exactly the six reported models")
    for item in manifest["models"]:
        for field in ("model_revision", "tokenizer_revision"):
            value = item.get(field, "")
            try:
                assert len(value) == 40
                int(value, 16)
            except (AssertionError, ValueError):
                errors.append(f"{item.get('slug')}: invalid {field}")
    for field, relative in (
        ("scenario_sha256", "new_models_full_server_package/data/historical/scenarios.csv"),
        ("auction_split_sha256", "new_models_full_server_package/data/historical/auction_splits.csv"),
    ):
        expected = manifest["dataset"][field]
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"{path}: missing or {field} mismatch")
    for relative, expected in manifest["preserved_artifact_sha256"].items():
        if relative == "original_four_historical_outputs":
            continue
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"{path}: missing or SHA-256 mismatch")
    historical = json.loads(
        (root / "robustness_extensions/server_outputs/SHA256_MANIFEST.json").read_text(encoding="utf-8")
    )
    historical_root = root / "robustness_extensions/server_outputs"
    for relative, expected in historical["outputs"].items():
        if not relative.startswith("raw/"):
            continue
        path = historical_root / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"{path}: missing or historical SHA-256 mismatch")

    for model in MODELS:
        base = qwen if model == "qwen3_14b" else granite if model == "granite_3_2_8b_instruct" else old
        for suffix, expected in HISTORICAL_EXPECTED.items():
            path = base / f"fixed_{model}_{suffix}.jsonl"
            rows = records(path)
            key = {(r["scenario_id"], r["budget"], r["repeat_seed"]) for r in rows}
            if len(rows) != expected or len(key) != expected:
                errors.append(f"{path}: rows={len(rows)}, unique={len(key)}, expected={expected}")
            if not all(finite(r, ("test_utility_ratio", "test_regret")) for r in rows):
                errors.append(f"{path}: non-finite score")

        synth_base = synth_qwen if model == "qwen3_14b" else synth_granite if model == "granite_3_2_8b_instruct" else synth_old
        path = synth_base / f"synthetic_{model}_deterministic.jsonl"
        rows = records(path)
        key = {(r["scenario_id"], r["reasoning_budget"], r["option_order_seed"]) for r in rows}
        if len(rows) != 4536 or len(key) != 4536:
            errors.append(f"{path}: rows={len(rows)}, unique={len(key)}, expected=4536")
        if not all(finite(r, ("utility_ratio", "regret")) for r in rows):
            errors.append(f"{path}: non-finite score")

    report = {"status": "failed" if errors else "passed", "models": list(MODELS), "errors": errors}
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
