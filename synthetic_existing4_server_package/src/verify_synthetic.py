#!/usr/bin/env python3
"""Verify synthetic manifests and inference results without loading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

BUDGETS = [0, 64, 128, 256, 384, 512, 768]
ORDER_SEEDS = [11, 23, 37]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", required=True, help="Comma-separated model slugs")
    parser.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    parser.add_argument("--option-order-seeds", default=",".join(map(str, ORDER_SEEDS)))
    parser.add_argument("--scenario-limit", type=int)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    models = [item for item in args.models.split(",") if item]
    budgets = [int(item) for item in args.budgets.split(",")]
    order_seeds = [int(item) for item in args.option_order_seeds.split(",")]
    manifest = args.data_dir / "synthetic_ipv_manifest.jsonl"
    expected_hash = (args.data_dir / "manifest.sha256").read_text().split()[0]
    actual_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    scenarios = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    if args.scenario_limit:
        scenarios = scenarios[:args.scenario_limit]
    report = {"benchmark_hash_expected": expected_hash, "benchmark_hash_actual": actual_hash, "models": {}}
    failures = actual_hash != expected_hash or (args.scenario_limit is None and len(scenarios) != 216)
    scenario_ids = {row["scenario_id"] for row in scenarios}
    for model in models:
        path = args.output_dir / "raw" / f"synthetic_{model}_deterministic.jsonl"
        rows, corrupt = [], []
        if path.exists():
            for number, line in enumerate(path.read_text().splitlines(), 1):
                try:
                    rows.append(json.loads(line))
                except Exception as exc:
                    corrupt.append({"line": number, "error": str(exc)})
        keys = [(row.get("scenario_id"), row.get("reasoning_budget"), row.get("option_order_seed")) for row in rows]
        counts = Counter(keys)
        expected = {(sid, budget, seed) for sid in scenario_ids for budget in budgets for seed in order_seeds}
        observed = set(keys)
        invalid = []
        nan_scores = []
        token_violations = []
        revision_mismatches = []
        hash_mismatches = []
        for index, row in enumerate(rows):
            if row.get("selected_option_label") not in [chr(65 + i) for i in range(int(row.get("option_count", 0)))] or row.get("selected_bid") is None:
                invalid.append(index)
            if any(not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) for key in ("utility", "optimal_utility", "utility_ratio", "regret")):
                nan_scores.append(index)
            budget = row.get("reasoning_budget")
            if budget == 0 and row.get("realized_reasoning_tokens") != 0 or isinstance(budget, int) and budget > 0 and row.get("realized_reasoning_tokens") != budget:
                token_violations.append(index)
            if row.get("requested_model_revision") != row.get("resolved_model_revision") or row.get("requested_tokenizer_revision") != row.get("resolved_tokenizer_revision"):
                revision_mismatches.append(index)
            if row.get("benchmark_sha256") != expected_hash:
                hash_mismatches.append(index)
        result = {
            "expected_cells": len(expected), "completed_rows": len(rows), "unique_completed_cells": len(observed & expected),
            "missing_cells": len(expected - observed), "unexpected_cells": len(observed - expected),
            "duplicate_keys": sum(count - 1 for count in counts.values() if count > 1), "corrupt_rows": len(corrupt),
            "invalid_constrained_outputs": len(invalid), "nan_or_infinite_scores": len(nan_scores),
            "reasoning_token_violations": len(token_violations), "revision_mismatches": len(revision_mismatches),
            "benchmark_hash_mismatches": len(hash_mismatches),
        }
        report["models"][model] = result
        failures = failures or any(result[key] for key in ("unexpected_cells", "duplicate_keys", "corrupt_rows", "invalid_constrained_outputs", "nan_or_infinite_scores", "reasoning_token_violations", "revision_mismatches", "benchmark_hash_mismatches"))
        if not args.allow_incomplete and result["missing_cells"]:
            failures = True
    report["valid"] = not failures
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
