#!/usr/bin/env python3
"""Recompute held-out utility from valid stored free-form bids.

The constrained option-letter readout remains primary.  This script reports
available-case free-form summaries and a paired 0-to-768-token sensitivity on
scenario/order cells whose free-form bid is valid at both endpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


BUDGETS = (0, 64, 128, 256, 512, 768)
ORDER_SEEDS = (11, 23, 37)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(probability * len(ordered))))
    return ordered[index]


def load_scenarios(path: Path) -> dict[str, dict[float, float]]:
    utilities: dict[str, dict[float, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            utilities[row["scenario_id"]] = {
                float(candidate["bid"]): float(candidate["test_u"])
                for candidate in json.loads(row["candidates_json"])
            }
    return utilities


def matched_bid(bid: object, candidates: dict[float, float]) -> float | None:
    if bid is None:
        return None
    value = float(bid)
    matches = [candidate for candidate in candidates if abs(candidate - value) < 1e-8]
    return matches[0] if matches else None


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="robustness_extensions/server_outputs",
        help="Directory containing data/, raw/, and tables/.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    root = Path(args.results_root)
    utilities = load_scenarios(root / "data" / "scenarios.csv")
    budget_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for path in sorted((root / "raw").glob("fixed_*_reasoning_greedy.jsonl")):
        records: list[dict[str, object]] = []
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                budget = int(row["budget"])
                if budget not in BUDGETS:
                    continue
                candidates = utilities[str(row["scenario_id"])]
                freeform_bid = matched_bid(row.get("freeform_chosen_bid"), candidates)
                valid = freeform_bid is not None
                records.append(
                    {
                        "model": row["model"],
                        "scenario_id": str(row["scenario_id"]),
                        "order_seed": int(row["repeat_seed"]),
                        "budget": budget,
                        "valid": valid,
                        "agreement": valid
                        and abs(float(freeform_bid) - float(row["chosen_bid"])) < 1e-8,
                        "utility_ratio": (
                            candidates[float(freeform_bid)] / float(row["test_optimal_utility"])
                            if valid
                            else None
                        ),
                    }
                )

        model = str(records[0]["model"])
        valid_records = [row for row in records if row["valid"]]
        for budget in BUDGETS:
            cells = [row for row in records if row["budget"] == budget]
            valid_cells = [row for row in cells if row["valid"]]
            budget_rows.append(
                {
                    "model": model,
                    "budget": budget,
                    "n_rows": len(cells),
                    "n_valid": len(valid_cells),
                    "validity_rate": len(valid_cells) / len(cells),
                    "agreement_conditional_valid": (
                        sum(bool(row["agreement"]) for row in valid_cells) / len(valid_cells)
                        if valid_cells else None
                    ),
                    "available_case_mean_utility_ratio": (
                        sum(float(row["utility_ratio"]) for row in valid_cells) / len(valid_cells)
                        if valid_cells else None
                    ),
                }
            )

        index = {
            (str(row["scenario_id"]), int(row["order_seed"]), int(row["budget"])): row
            for row in records
        }
        scenario_endpoints: list[tuple[float, float]] = []
        paired_orders = 0
        for scenario_id in sorted({str(row["scenario_id"]) for row in records}):
            low_values: list[float] = []
            high_values: list[float] = []
            for order_seed in ORDER_SEEDS:
                low = index.get((scenario_id, order_seed, 0))
                high = index.get((scenario_id, order_seed, 768))
                if low and high and low["valid"] and high["valid"]:
                    low_values.append(float(low["utility_ratio"]))
                    high_values.append(float(high["utility_ratio"]))
                    paired_orders += 1
            if low_values:
                scenario_endpoints.append(
                    (
                        sum(low_values) / len(low_values),
                        sum(high_values) / len(high_values),
                    )
                )

        differences = [high - low for low, high in scenario_endpoints]
        rng = random.Random(args.seed)
        bootstrap = ([
            sum(rng.choice(differences) for _ in differences) / len(differences)
            for _ in range(args.bootstrap_samples)
        ] if differences else [])
        summary_rows.append(
            {
                "model": model,
                "n_rows": len(records),
                "freeform_validity_rate": len(valid_records) / len(records),
                "agreement_conditional_valid": sum(
                    bool(row["agreement"]) for row in valid_records
                )
                / len(valid_records),
                "n_paired_scenarios": len(scenario_endpoints),
                "n_paired_scenario_order_cells": paired_orders,
                "paired_q_0": (sum(low for low, _ in scenario_endpoints) / len(scenario_endpoints) if scenario_endpoints else None),
                "paired_q_768": (sum(high for _, high in scenario_endpoints) / len(scenario_endpoints) if scenario_endpoints else None),
                "paired_difference": (sum(differences) / len(differences) if differences else None),
                "paired_difference_ci_low": (percentile(bootstrap, 0.025) if bootstrap else None),
                "paired_difference_ci_high": (percentile(bootstrap, 0.975) if bootstrap else None),
            }
        )

    write_rows(root / "tables" / "freeform_sensitivity_by_budget.csv", budget_rows)
    write_rows(root / "tables" / "freeform_sensitivity_summary.csv", summary_rows)


if __name__ == "__main__":
    main()
