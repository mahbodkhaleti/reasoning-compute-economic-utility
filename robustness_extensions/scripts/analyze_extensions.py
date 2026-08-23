#!/usr/bin/env python3
"""Create the preregistered robustness tables after the 4090 inference runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_fixed import analytic_baselines
from common import parse_candidate_json, read_jsonl


def cluster_ci(frame: pd.DataFrame, column: str, rng: np.random.Generator, samples: int = 2000) -> tuple[float, float]:
    clusters = frame.groupby("scenario_id")[column].mean().dropna().to_numpy(float)
    if not len(clusters):
        return math.nan, math.nan
    draws = clusters[rng.integers(0, len(clusters), size=(samples, len(clusters)))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize_seed(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (model, budget), group in data.groupby(["model", "budget"]):
        row = {
            "model": model,
            "budget": int(budget),
            "n_scenarios": group["scenario_id"].nunique(),
            "n_generations": len(group),
            "mean_selected_bid": group["chosen_bid"].mean(),
            "bid_variance": group["chosen_bid"].var(ddof=1),
            "mean_utility_ratio": group["test_utility_ratio"].mean(),
            "utility_ratio_sd": group["test_utility_ratio"].std(ddof=1),
            "mean_regret": group["test_regret"].mean(),
            "optimal_bid_probability": group["test_correct"].mean(),
        }
        for column, prefix in [
            ("chosen_bid", "mean_selected_bid"),
            ("test_utility_ratio", "mean_utility_ratio"),
            ("test_regret", "mean_regret"),
            ("test_correct", "optimal_bid_probability"),
        ]:
            row[f"{prefix}_ci_low"], row[f"{prefix}_ci_high"] = cluster_ci(group, column, rng)
        rows.append(row)
    return pd.DataFrame(rows)


def slope_ci(group: pd.DataFrame, rng: np.random.Generator, samples: int = 2000) -> dict[str, float]:
    scenario_means = group.loc[group["budget"] >= 0].groupby(["scenario_id", "budget"], as_index=False)["test_utility_ratio"].mean()
    ids = scenario_means["scenario_id"].unique()

    def slope(frame: pd.DataFrame) -> float:
        x = np.log1p(frame["budget"].to_numpy(float))
        y = frame["test_utility_ratio"].to_numpy(float)
        return float(np.polyfit(x, y, 1)[0]) if len(np.unique(x)) > 1 else math.nan

    point = slope(scenario_means)
    draws = []
    by_id = {sid: scenario_means.loc[scenario_means["scenario_id"] == sid] for sid in ids}
    for _ in range(samples):
        selected = rng.choice(ids, size=len(ids), replace=True)
        draws.append(slope(pd.concat([by_id[sid] for sid in selected], ignore_index=True)))
    return {
        "log_budget_slope": point,
        "slope_ci_low": float(np.quantile(draws, 0.025)),
        "slope_ci_high": float(np.quantile(draws, 0.975)),
        "n_scenarios": len(ids),
    }


def paired_endpoints(data: pd.DataFrame, rng: np.random.Generator, samples: int = 2000) -> pd.DataFrame:
    rows = []
    for model, group in data.groupby("model"):
        budgets = sorted(x for x in group["budget"].unique() if x >= 0)
        if len(budgets) < 2:
            continue
        low, high = budgets[0], budgets[-1]
        panel = group.groupby(["scenario_id", "budget"])["test_utility_ratio"].mean().unstack()
        difference = (panel[high] - panel[low]).dropna().to_numpy(float)
        draws = difference[rng.integers(0, len(difference), size=(samples, len(difference)))].mean(axis=1)
        rows.append(
            {
                "model": model,
                "low_budget": low,
                "high_budget": high,
                "n_paired_scenarios": len(difference),
                "mean_paired_difference": difference.mean(),
                "ci_low": np.quantile(draws, 0.025),
                "ci_high": np.quantile(draws, 0.975),
                "two_sided_bootstrap_p": 2 * min(np.mean(draws <= 0), np.mean(draws >= 0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="robustness_extensions/server_outputs")
    parser.add_argument("--export-dir", default="robustness_extensions")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    source, export = Path(args.output_dir), Path(args.export_dir)
    raw_export, table_export = export / "raw_results", export / "tables"
    raw_export.mkdir(parents=True, exist_ok=True)
    table_export.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted((source / "raw").glob("fixed_*.jsonl")):
        records.extend(read_jsonl(path))
    if not records:
        raise RuntimeError(f"No fixed-budget JSONL files found under {source / 'raw'}")
    data = pd.DataFrame(records)
    data["scenario_id"] = data["scenario_id"].astype(str).str.zfill(12)
    data["decoding"] = data.get("decoding", "unknown")
    data = data.drop_duplicates(
        ["model", "chat_mode", "decoding", "scenario_id", "budget", "repeat_seed"],
        keep="last",
    )
    rng = np.random.default_rng(20260819)

    stochastic = data.loc[data["decoding"] == "stochastic"].copy()
    stochastic.to_csv(raw_export / "seed_robustness_raw.csv", index=False)
    summarize_seed(stochastic, rng).to_csv(table_export / "seed_robustness_summary.csv", index=False)

    reasoning = data.loc[(data["chat_mode"] == "reasoning") & (data["decoding"] == "greedy")].copy()
    order_rows = []
    for row in reasoning.to_dict("records"):
        order = row["option_order"] if isinstance(row["option_order"], list) else json.loads(row["option_order"])
        chosen_position = order.index(row["chosen_bid"]) if row.get("chosen_bid") in order else math.nan
        optimal_position = order.index(row["test_optimal_bid"]) if row.get("test_optimal_bid") in order else math.nan
        order_rows.append({**row, "chosen_position": chosen_position, "optimal_position": optimal_position})
    pd.DataFrame(order_rows).to_csv(table_export / "randomized_order_results.csv", index=False)

    decoys = (
        reasoning.groupby(["model", "category", "option_count", "candidate_set_index", "budget"], as_index=False)
        .agg(
            n_scenarios=("scenario_id", "nunique"),
            mean_utility_ratio=("test_utility_ratio", "mean"),
            mean_regret=("test_regret", "mean"),
            accuracy=("test_correct", "mean"),
        )
    )
    decoys.to_csv(table_export / "random_decoy_results.csv", index=False)

    scenarios = pd.read_csv(source / "data" / "scenarios.csv", dtype={"scenario_id": str})
    configured_seeds = sorted(reasoning["repeat_seed"].unique().tolist())
    baselines = analytic_baselines(scenarios, configured_seeds)
    baseline_like = pd.concat([reasoning, data.loc[data["chat_mode"] == "no_think"], baselines], ignore_index=True, sort=False)
    baseline_summary = (
        baseline_like.groupby(["model", "chat_mode", "budget"], dropna=False, as_index=False)
        .agg(
            n_scenarios=("scenario_id", "nunique"),
            mean_utility_ratio=("test_utility_ratio", "mean"),
            mean_regret=("test_regret", "mean"),
            optimal_bid_probability=("test_correct", "mean"),
            mean_elapsed_seconds=("elapsed_seconds", "mean"),
            mean_compute_cost_usd=("total_compute_cost_usd", "mean"),
        )
    )
    baseline_summary.to_csv(table_export / "baseline_comparison.csv", index=False)
    (
        reasoning.groupby(["model", "category", "budget"], as_index=False)
        .agg(
            n_scenarios=("scenario_id", "nunique"),
            mean_utility_ratio=("test_utility_ratio", "mean"),
            mean_regret=("test_regret", "mean"),
            optimal_bid_probability=("test_correct", "mean"),
        )
        .to_csv(table_export / "multi_product_summary.csv", index=False)
    )

    regression_rows = [{"model": model, **slope_ci(group, rng, args.bootstrap_samples)} for model, group in reasoning.groupby("model")]
    pd.DataFrame(regression_rows).to_csv(table_export / "budget_regression.csv", index=False)
    paired_endpoints(reasoning, rng, args.bootstrap_samples).to_csv(table_export / "paired_budget_comparisons.csv", index=False)

    cost = (
        data.groupby(["model", "chat_mode", "decoding", "budget"], as_index=False)
        .agg(
            n=("scenario_id", "size"),
            mean_seconds=("elapsed_seconds", "mean"),
            mean_energy_joules=("energy_joules", "mean"),
            mean_rental_cost_usd=("rental_cost_usd", "mean"),
            mean_energy_cost_usd=("energy_cost_usd", "mean"),
            mean_total_compute_cost_usd=("total_compute_cost_usd", "mean"),
        )
    )
    cost.to_csv(table_export / "compute_cost_calibration.csv", index=False)
    auction_splits = pd.read_csv(source / "data" / "auction_splits.csv")
    social_calibration = (
        auction_splits.groupby(["item", "split"], as_index=False)
        .agg(n_auctions=("auctionid", "nunique"), mean_final_price=("final_price", "mean"), median_final_price=("final_price", "median"))
    )
    social_calibration.to_csv(table_export / "social_value_calibration.csv", index=False)
    print(f"Wrote robustness tables to {table_export}")


if __name__ == "__main__":
    main()
