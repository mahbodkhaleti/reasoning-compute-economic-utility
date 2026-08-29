#!/usr/bin/env python3
"""Recompute runtime/energy slopes and the paper's hardware-matched cost benchmark."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


LABELS = {
    "deepseek_r1_llama_8b": "DeepSeek-8B",
    "phi4_mini_reasoning": "Phi-mini",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
    "qwen3_14b": "Qwen3-14B",
    "granite_3_2_8b_instruct": "Granite-8B",
}


def slope(xs: list[float], ys: list[float]) -> float:
    xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rtx4090-usd-per-hour", type=float, default=0.74)
    args = parser.parse_args()
    heldout: list[float] = []
    with (args.output_dir / "data/auction_splits.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                heldout.append(float(row["final_price"]))
    prize = sum(heldout) / len(heldout)
    rows = []
    for model, label in LABELS.items():
        path = args.output_dir / "raw" / f"fixed_{model}_reasoning_stochastic.jsonl"
        data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        xs = [float(row["budget"]) for row in data]
        seconds = slope(xs, [float(row["elapsed_seconds"]) for row in data])
        energy_values = [row.get("energy_joules") for row in data]
        joules = slope(xs, [float(value) for value in energy_values]) if all(value is not None and math.isfinite(float(value)) for value in energy_values) else math.nan
        matched = model != "qwen3_14b"
        price = args.rtx4090_usd_per_hour if matched else math.nan
        lam = seconds * price / 3600 if matched else math.nan
        rows.append({
            "model": model, "model_label": label,
            "seconds_per_reasoning_token": seconds,
            "joules_per_reasoning_token": joules,
            "hardware_matched_rtx4090_price_usd_per_hour": price,
            "lambda_usd_per_token": lam,
            "heldout_prize_benchmark_usd": prize,
            "kappa": lam / prize if matched else math.nan,
            "pricing_status": "RTX 4090 matched benchmark" if matched else "unpriced: L40S run excluded from RTX 4090 comparison",
        })
    path = args.output_dir / "tables/runtime_cost_recomputed.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}; held-out prize benchmark={prize:.2f}")


if __name__ == "__main__":
    main()
