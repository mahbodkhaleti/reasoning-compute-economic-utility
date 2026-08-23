#!/usr/bin/env python3
"""Estimate remaining wall time from cells already completed on this GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "fixed_qwen3_14b_reasoning_greedy.jsonl": 216 * 7 * 3,
    "fixed_qwen3_14b_no_think_greedy.jsonl": 216 * 1 * 3,
    "fixed_qwen3_14b_reasoning_stochastic.jsonl": 216 * 3 * 5,
}


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    raw = Path(args.output_dir) / "raw"
    total_seconds = 0.0
    estimable = True
    for name, expected in EXPECTED.items():
        data = rows(raw / name)
        unique = {
            (row["scenario_id"], int(row["budget"]), int(row["repeat_seed"])): row
            for row in data
        }
        elapsed = [float(row["elapsed_seconds"]) for row in unique.values() if row.get("elapsed_seconds") is not None]
        remaining = max(0, expected - len(unique))
        if elapsed:
            mean = sum(elapsed) / len(elapsed)
            estimate = mean * remaining
            total_seconds += estimate
            print(f"{name}: {len(unique)}/{expected}; mean {mean:.2f}s/cell; remaining estimate {estimate / 3600:.2f}h")
        else:
            estimable = False
            print(f"{name}: 0/{expected}; run a smoke/pilot first to estimate this phase")
    if estimable:
        print(f"Total sequential remaining estimate: {total_seconds / 3600:.2f}h")
        print("This extrapolation is diagnostic; long-budget cells can be slower than early cells.")


if __name__ == "__main__":
    main()
