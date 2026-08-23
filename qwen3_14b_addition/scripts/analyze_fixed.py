#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_dirs, load_config, parse_candidate_json, read_jsonl, shuffled_candidates


def bootstrap_mean(values: pd.DataFrame, column: str, samples: int, seed: int) -> tuple[float, float]:
    # Cluster on scenario so repeated option orders/seeds stay together.
    clusters = values.groupby("scenario_id")[column].mean().dropna()
    if len(clusters) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    arr = clusters.to_numpy(float)
    draws = rng.choice(arr, size=(samples, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def analytic_baselines(scenarios: pd.DataFrame, repeat_seeds: list[int]) -> pd.DataFrame:
    rows = []
    for scenario in scenarios.to_dict("records"):
        scenario["scenario_id"] = str(scenario["scenario_id"]).zfill(12)
        for seed in repeat_seeds:
            options = shuffled_candidates(scenario, seed)
            test_best = max(options, key=lambda x: (x["test_u"], -x["bid"]))
            policies = {
                "calculator_train": max(options, key=lambda x: (x["train_u"], -x["bid"])),
                "median_position": options[len(options) // 2],
                "highest_win_probability": max(options, key=lambda x: (x["train_p"], -x["bid"])),
            }
            for name, choice in policies.items():
                rows.append(
                    {
                        "model": name,
                        "chat_mode": "analytic_baseline",
                        "scenario_id": scenario["scenario_id"],
                        "category": scenario["category"],
                        "value": float(scenario["value"]),
                        "option_count": int(scenario["option_count"]),
                        "repeat_seed": seed,
                        "budget": 0,
                        "valid_output": True,
                        "test_correct": choice["bid"] == test_best["bid"],
                        "test_utility": choice["test_u"],
                        "test_optimal_utility": test_best["test_u"],
                        "test_regret": test_best["test_u"] - choice["test_u"],
                        "test_utility_ratio": choice["test_u"] / test_best["test_u"] if test_best["test_u"] > 0 else math.nan,
                        "elapsed_seconds": 0.0,
                        "total_compute_cost_usd": 0.0,
                    }
                )
            mean_u = float(np.mean([x["test_u"] for x in options]))
            rows.append(
                {
                    "model": "uniform_random_expected",
                    "chat_mode": "analytic_baseline",
                    "scenario_id": scenario["scenario_id"],
                    "category": scenario["category"],
                    "value": float(scenario["value"]),
                    "option_count": int(scenario["option_count"]),
                    "repeat_seed": seed,
                    "budget": 0,
                    "valid_output": True,
                    "test_correct": 1.0 / len(options),
                    "test_utility": mean_u,
                    "test_optimal_utility": test_best["test_u"],
                    "test_regret": test_best["test_u"] - mean_u,
                    "test_utility_ratio": mean_u / test_best["test_u"] if test_best["test_u"] > 0 else math.nan,
                    "elapsed_seconds": 0.0,
                    "total_compute_cost_usd": 0.0,
                }
            )
            rows.append(
                {
                    "model": "heldout_oracle",
                    "chat_mode": "analytic_baseline",
                    "scenario_id": scenario["scenario_id"],
                    "category": scenario["category"],
                    "value": float(scenario["value"]),
                    "option_count": int(scenario["option_count"]),
                    "repeat_seed": seed,
                    "budget": 0,
                    "valid_output": True,
                    "test_correct": True,
                    "test_utility": test_best["test_u"],
                    "test_optimal_utility": test_best["test_u"],
                    "test_regret": 0.0,
                    "test_utility_ratio": 1.0,
                    "elapsed_seconds": 0.0,
                    "total_compute_cost_usd": 0.0,
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    group_cols = ["model", "chat_mode", "decoding", "budget", "category", "option_count"]
    for keys, group in df.groupby(group_cols, dropna=False):
        lo, hi = bootstrap_mean(
            group,
            "test_utility_ratio",
            config["evaluation"]["bootstrap_samples"],
            config["seed"] + abs(hash(keys)) % 100000,
        )
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_rows": len(group),
                "n_scenarios": group["scenario_id"].nunique(),
                "valid_output_rate": float(group["valid_output"].mean()),
                "test_accuracy": float(pd.to_numeric(group["test_correct"]).mean()),
                "mean_test_utility_ratio": float(group["test_utility_ratio"].mean()),
                "utility_ratio_ci_low": lo,
                "utility_ratio_ci_high": hi,
                "mean_test_regret": float(group["test_regret"].mean()),
                "mean_elapsed_seconds": float(group["elapsed_seconds"].mean()),
                "mean_compute_cost_usd": float(group["total_compute_cost_usd"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_frontiers(model_df: pd.DataFrame) -> pd.DataFrame:
    usable = model_df.loc[
        (model_df["chat_mode"] == "reasoning")
        & (model_df["decoding"] == "greedy")
        & (model_df["budget"] >= 0)
    ].copy()
    frontier = (
        usable.groupby(["model", "budget"], as_index=False)
        .agg(
            mean_test_utility_ratio=("test_utility_ratio", "mean"),
            mean_test_utility_usd=("test_utility", "mean"),
            mean_test_regret_usd=("test_regret", "mean"),
            accuracy=("test_correct", "mean"),
            valid_output_rate=("valid_output", "mean"),
            mean_seconds=("elapsed_seconds", "mean"),
            mean_compute_cost_usd=("total_compute_cost_usd", "mean"),
            mean_reasoning_tokens=("reasoning_tokens", "mean"),
        )
        .sort_values(["model", "budget"])
    )
    return frontier


def direct_outcome(a: pd.Series, b: pd.Series) -> dict[str, float]:
    bid_a, bid_b = float(a["chosen_bid"]), float(b["chosen_bid"])
    if not bool(a["valid_output"]):
        bid_a = 0.0
    if not bool(b["valid_output"]):
        bid_b = 0.0
    if bid_a > bid_b:
        p_a = 1.0
    elif bid_b > bid_a:
        p_a = 0.0
    else:
        p_a = 0.5
    p_b = 1.0 - p_a
    value_a, value_b = float(a["value"]), float(b["value"])
    cost_a = float(a["total_compute_cost_usd"])
    cost_b = float(b["total_compute_cost_usd"])
    if math.isnan(cost_a) or math.isnan(cost_b):
        return {}
    private_a = p_a * max(value_a - bid_a, 0.0) - cost_a
    private_b = p_b * max(value_b - bid_b, 0.0) - cost_b
    realized_value = p_a * value_a + p_b * value_b
    efficient_value = max(value_a, value_b)
    social_welfare = realized_value - cost_a - cost_b
    return {
        "private_payoff_a": private_a,
        "private_payoff_b": private_b,
        "allocative_loss_usd": efficient_value - realized_value,
        "compute_cost_usd": cost_a + cost_b,
        "social_welfare_usd": social_welfare,
        "efficient_welfare_usd": efficient_value,
        "total_welfare_loss_usd": efficient_value - social_welfare,
        "total_reasoning_tokens": float(a["reasoning_tokens"]) + float(b["reasoning_tokens"]),
    }


def head_to_head_matrix(df: pd.DataFrame) -> pd.DataFrame:
    data = df.loc[
        (df["chat_mode"] == "reasoning")
        & (df["decoding"] == "greedy")
        & (df["budget"] >= 0)
    ].copy()
    models = sorted(data["model"].unique())
    lookup = {
        (r.model, int(r.budget), int(r.repeat_seed), str(r.scenario_id).zfill(12)): r
        for r in data.itertuples(index=False)
    }
    observations = []
    scenarios = (
        data[["scenario_id", "category", "option_count"]]
        .drop_duplicates()
        .assign(scenario_id=lambda x: x["scenario_id"].astype(str).str.zfill(12))
    )
    for category in sorted(scenarios["category"].unique()):
        for option_count in sorted(scenarios["option_count"].unique()):
            ids = sorted(
                scenarios.loc[
                    (scenarios["category"] == category) & (scenarios["option_count"] == option_count),
                    "scenario_id",
                ].tolist()
            )
            pairs = list(zip(ids[0::2], ids[1::2]))
            for model_a, model_b in itertools.combinations_with_replacement(models, 2):
                budgets_a = sorted(data.loc[data["model"] == model_a, "budget"].unique())
                budgets_b = sorted(data.loc[data["model"] == model_b, "budget"].unique())
                seeds = sorted(set(data.loc[data["model"] == model_a, "repeat_seed"]).intersection(data.loc[data["model"] == model_b, "repeat_seed"]))
                for ba, bb, seed in itertools.product(budgets_a, budgets_b, seeds):
                    for sid_a, sid_b in pairs:
                        ra = lookup.get((model_a, int(ba), int(seed), sid_a))
                        rb = lookup.get((model_b, int(bb), int(seed), sid_b))
                        if ra is None or rb is None:
                            continue
                        outcome = direct_outcome(pd.Series(ra._asdict()), pd.Series(rb._asdict()))
                        if outcome:
                            observations.append(
                                {
                                    "model_a": model_a,
                                    "model_b": model_b,
                                    "budget_a": int(ba),
                                    "budget_b": int(bb),
                                    "category": category,
                                    "option_count": int(option_count),
                                    "repeat_seed": int(seed),
                                    "scenario_a": sid_a,
                                    "scenario_b": sid_b,
                                    **outcome,
                                }
                            )
    if not observations:
        return pd.DataFrame()
    raw = pd.DataFrame(observations)
    return (
        raw.groupby(["model_a", "model_b", "budget_a", "budget_b"], as_index=False)
        .agg(
            n_games=("scenario_a", "size"),
            payoff_a=("private_payoff_a", "mean"),
            payoff_b=("private_payoff_b", "mean"),
            social_welfare_usd=("social_welfare_usd", "mean"),
            welfare_loss_usd=("total_welfare_loss_usd", "mean"),
            allocative_loss_usd=("allocative_loss_usd", "mean"),
            compute_cost_usd=("compute_cost_usd", "mean"),
            total_reasoning_tokens=("total_reasoning_tokens", "mean"),
        )
    )


def solve_games(matrix: pd.DataFrame) -> pd.DataFrame:
    results = []
    if matrix.empty:
        return pd.DataFrame()
    for (model_a, model_b), game in matrix.groupby(["model_a", "model_b"]):
        budgets_a = sorted(game["budget_a"].unique())
        budgets_b = sorted(game["budget_b"].unique())
        cell = {(int(r.budget_a), int(r.budget_b)): r for r in game.itertuples(index=False)}
        equilibria = []
        for ba, bb in itertools.product(budgets_a, budgets_b):
            row = cell.get((ba, bb))
            if row is None:
                continue
            best_a = max(cell[(x, bb)].payoff_a for x in budgets_a if (x, bb) in cell)
            best_b = max(cell[(ba, y)].payoff_b for y in budgets_b if (ba, y) in cell)
            if row.payoff_a >= best_a - 1e-9 and row.payoff_b >= best_b - 1e-9:
                equilibria.append((ba, bb))
        planner_row = max(game.itertuples(index=False), key=lambda r: r.social_welfare_usd)
        for ba, bb in equilibria:
            ne = cell[(ba, bb)]
            planner_tokens = planner_row.total_reasoning_tokens
            results.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "ne_budget_a": ba,
                    "ne_budget_b": bb,
                    "planner_budget_a": int(planner_row.budget_a),
                    "planner_budget_b": int(planner_row.budget_b),
                    "ne_total_tokens": ne.total_reasoning_tokens,
                    "planner_total_tokens": planner_tokens,
                    "PoRC": ne.total_reasoning_tokens / planner_tokens if planner_tokens > 0 else math.nan,
                    "ne_social_welfare_usd": ne.social_welfare_usd,
                    "planner_social_welfare_usd": planner_row.social_welfare_usd,
                    "welfare_gap_usd": planner_row.social_welfare_usd - ne.social_welfare_usd,
                    "ne_welfare_loss_usd": ne.welfare_loss_usd,
                    "planner_welfare_loss_usd": planner_row.welfare_loss_usd,
                    "equilibrium_count": len(equilibria),
                }
            )
        if not equilibria:
            results.append({"model_a": model_a, "model_b": model_b, "equilibrium_count": 0})
    return pd.DataFrame(results)


def plot_frontiers(frontier: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for model, group in frontier.groupby("model"):
        plt.plot(group["budget"], group["mean_test_utility_ratio"], marker="o", label=model)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1, label="held-out oracle")
    plt.xlabel("Reasoning-token budget")
    plt.ylabel("Held-out mean utility ratio")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(args.output_dir)
    records = []
    for path in sorted(paths["raw"].glob("fixed_*.jsonl")):
        records.extend(read_jsonl(path))
    if not records:
        raise RuntimeError("No fixed-budget JSONL files found")
    model_df = pd.DataFrame(records)
    if "decoding" not in model_df:
        model_df["decoding"] = "unknown"
    model_df = model_df.drop_duplicates(
        ["model", "chat_mode", "decoding", "scenario_id", "budget", "repeat_seed"],
        keep="last",
    )
    model_df["scenario_id"] = model_df["scenario_id"].astype(str).str.zfill(12)
    scenarios = pd.read_csv(paths["data"] / "scenarios.csv", dtype={"scenario_id": str})
    scenarios["scenario_id"] = scenarios["scenario_id"].str.zfill(12)
    baselines = analytic_baselines(scenarios, config["evaluation"]["repeat_seeds"])
    baselines["decoding"] = "analytic"
    combined = pd.concat([model_df, baselines], ignore_index=True, sort=False)
    summary = summarize(combined, config)
    summary.to_csv(paths["tables"] / "fixed_budget_summary.csv", index=False)
    baselines.to_csv(paths["tables"] / "analytic_baselines.csv", index=False)
    frontier = make_frontiers(model_df)
    frontier.to_csv(paths["tables"] / "model_frontiers.csv", index=False)
    plot_frontiers(frontier, paths["figures"] / "leakage_free_compute_quality.png")
    matrix = head_to_head_matrix(model_df)
    matrix.to_csv(paths["tables"] / "head_to_head_budget_matrix.csv", index=False)
    games = solve_games(matrix)
    games.to_csv(paths["tables"] / "direct_matrix_equilibria.csv", index=False)
    print(summary.to_string(index=False))
    if matrix.empty:
        print("WARNING: head-to-head matrix omitted because monetary compute cost was not configured.")
    else:
        print("\nDirect payoff-matrix equilibria:\n", games.to_string(index=False))


if __name__ == "__main__":
    main()
