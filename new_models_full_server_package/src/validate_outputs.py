#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from common import ensure_dirs, load_config, parse_candidate_json, read_jsonl


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-unpriced", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(args.output_dir)
    errors: list[str] = []
    warnings: list[str] = []

    scenario_path = paths["data"] / "scenarios.csv"
    check(scenario_path.exists(), "Missing data/scenarios.csv", errors)
    if not scenario_path.exists():
        raise SystemExit("Missing scenarios; run prepare_scenarios.py")
    scenarios = pd.read_csv(scenario_path, dtype={"scenario_id": str})
    check(scenarios["scenario_id"].is_unique, "Scenario IDs are not unique", errors)
    check(set(scenarios["category"]) == set(config["data"]["categories"]), "Not all configured categories are present", errors)
    check(set(scenarios["option_count"]) == set(config["data"]["option_sizes"]), "Not all option sizes are present", errors)
    for row in scenarios.to_dict("records"):
        candidates = parse_candidate_json(row["candidates_json"])
        bids = [x["bid"] for x in candidates]
        check(len(bids) == int(row["option_count"]), f"Wrong option count in {row['scenario_id']}", errors)
        check(len(set(bids)) == len(bids), f"Duplicate bid in {row['scenario_id']}", errors)
        check(all(0 <= bid <= float(row["value"]) for bid in bids), f"Infeasible bid in {row['scenario_id']}", errors)
    position_counts = scenarios.groupby(["option_count", "test_best_sorted_position"]).size()
    for option_count in config["data"]["option_sizes"]:
        unique_positions = len(position_counts.loc[option_count]) if option_count in position_counts.index.levels[0] else 0
        check(unique_positions >= 2, f"Held-out optimum has only one sorted position for k={option_count}", errors)

    expected_fixed = len(scenarios) * (len(config["evaluation"]["budgets"]) + 1) * len(config["evaluation"]["repeat_seeds"])
    expected_stochastic = len(scenarios) * len(config["seed_robustness"]["budgets"]) * len(config["seed_robustness"]["seeds"])
    for model in config["models"]:
        reasoning_path = paths["raw"] / f"fixed_{model['slug']}_reasoning_greedy.jsonl"
        rows = read_jsonl(reasoning_path)
        reasoning_by_key = {
            (x["scenario_id"], int(x["budget"]), int(x["repeat_seed"])): x
            for x in rows
        }
        rows = list(reasoning_by_key.values())
        if not rows:
            errors.append(f"Missing fixed reasoning output for {model['slug']}")
        elif len(rows) != expected_fixed:
            message = f"{model['slug']} fixed reasoning has {len(rows)} rows; expected {expected_fixed}"
            (warnings if args.allow_incomplete else errors).append(message)
        if rows:
            costs = [float(x["total_compute_cost_usd"]) for x in rows]
            priced = any(not math.isnan(x) and x > 0 for x in costs)
            if not priced:
                message = f"No measured dollar compute cost for {model['slug']}"
                (warnings if args.allow_unpriced else errors).append(message)
            order_by_scenario = {}
            for x in rows:
                order_by_scenario.setdefault(x["scenario_id"], set()).add(tuple(x["option_order"]))
            check(any(len(v) > 1 for v in order_by_scenario.values()), f"Option order was not randomized for {model['slug']}", errors)
        stochastic_path = paths["raw"] / f"fixed_{model['slug']}_reasoning_stochastic.jsonl"
        stochastic_rows = read_jsonl(stochastic_path)
        stochastic_by_key = {
            (x["scenario_id"], int(x["budget"]), int(x["repeat_seed"])): x
            for x in stochastic_rows
        }
        stochastic_rows = list(stochastic_by_key.values())
        if len(stochastic_rows) != expected_stochastic:
            message = f"{model['slug']} stochastic run has {len(stochastic_rows)} rows; expected {expected_stochastic}"
            (warnings if args.allow_incomplete else errors).append(message)
        if stochastic_rows:
            check(
                len({int(x["option_order_seed"]) for x in stochastic_rows}) == 1,
                f"Generation-uncertainty run changes option ordering for {model['slug']}",
                errors,
            )
            check(
                all("answer_geomean_probability" in x for x in stochastic_rows),
                f"Missing answer-token confidence for {model['slug']}",
                errors,
            )
            check(
                all(x.get("verbal_confidence") is not None for x in stochastic_rows),
                f"Missing or unparseable verbal confidence for {model['slug']}",
                errors,
            )

    baseline_path = paths["tables"] / "fixed_budget_summary.csv"
    if baseline_path.exists():
        summary = pd.read_csv(baseline_path)
        median = summary.loc[summary["model"] == "median_position", "test_accuracy"]
        check(not median.empty and float(median.mean()) < 0.999, "Median-position baseline is still perfect", errors)
    else:
        errors.append("Missing fixed_budget_summary.csv")

    confidence_outputs = [
        paths["tables"] / "confidence_calibration_summary.csv",
        paths["tables"] / "selective_risk_coverage.csv",
        paths["tables"] / "adaptive_sampling_summary.csv",
        paths["tables"] / "conformal_stopping_summary.csv",
        paths["tables"] / "stochastic_game_uncertainty_summary.csv",
    ]
    for path in confidence_outputs:
        check(path.exists(), f"Missing confidence-aware analysis output: {path.name}", errors)

    if args.skip_direct:
        warnings.append("Price-dependent direct games were intentionally skipped")
    else:
        for model in config["models"]:
            for mode in ["adaptive", "fixed_low", "fixed_high"]:
                path = paths["direct"] / f"direct_{model['slug']}_{mode}.jsonl"
                rows = read_jsonl(path)
                expected = config["direct_game"]["rounds"] * len(config["direct_game"]["experiment_seeds"])
                by_key = {
                    (int(row["experiment_seed"]), int(row["round"])): row
                    for row in rows
                    if row.get("bid_readout_method") == "constrained_option_letter_next_token"
                }
                if len(by_key) != expected:
                    message = f"{model['slug']} {mode} has {len(by_key)} constrained-readout direct rounds; expected {expected}"
                    (warnings if args.allow_incomplete else errors).append(message)

    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "scenario_count": len(scenarios),
        "test_best_sorted_position_counts": {
            f"k={k},position={p}": int(v) for (k, p), v in position_counts.items()
        },
    }
    report_path = paths["base"] / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
