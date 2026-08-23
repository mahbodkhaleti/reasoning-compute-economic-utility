#!/usr/bin/env python3
"""Hierarchical scenario/generation bootstrap and strategic-game propagation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_uncertainty import aggregate_game, solve_grid
from common import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="robustness_extensions/server_outputs")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    output = Path(args.output_dir)
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted((output / "raw").glob("fixed_*_reasoning_stochastic.jsonl")):
        records.extend(read_jsonl(path))
    if not records:
        raise RuntimeError("No stochastic outputs found")
    data = pd.DataFrame(records)
    data["scenario_id"] = data["scenario_id"].astype(str).str.zfill(12)
    data["_confidence_complete"] = data.get("verbal_confidence", pd.Series(index=data.index, dtype=float)).notna().astype(int)
    data = (
        data.sort_values("_confidence_complete")
        .drop_duplicates(["model", "scenario_id", "budget", "repeat_seed"], keep="last")
        .drop(columns="_confidence_complete")
    )
    values = np.arange(25, 1001, 25, dtype=float)
    lambdas = np.linspace(0.0001, 0.03, 60)
    rng = np.random.default_rng(args.seed)
    frontier_rows, game_rows, summary_rows = [], [], []

    for model, group in data.groupby("model"):
        budgets = np.array(sorted(group["budget"].unique()), dtype=int)
        scenarios = sorted(group["scenario_id"].unique())
        seeds = sorted(group["repeat_seed"].unique())
        panel = (
            group.pivot_table(index="scenario_id", columns=["budget", "repeat_seed"], values="test_utility_ratio", aggfunc="first")
            .reindex(index=scenarios, columns=pd.MultiIndex.from_product([budgets, seeds]))
        )
        if panel.isna().any().any():
            raise ValueError(f"{model}: incomplete scenario-budget-seed panel")
        array = panel.to_numpy(float).reshape(len(scenarios), len(budgets), len(seeds))
        point_q = array.mean(axis=(0, 2))
        q_draws = np.empty((args.samples, len(budgets)))
        metric_draws = []
        for b in range(args.samples):
            scenario_index = rng.integers(0, len(scenarios), size=len(scenarios))
            seed_index = rng.integers(0, len(seeds), size=len(scenarios))
            selected = array[scenario_index, :, seed_index]
            q = selected.mean(axis=0)
            q_draws[b] = q
            solved = solve_grid(q, budgets, values, lambdas, epsilon=0.01)
            metrics = aggregate_game(solved)
            game_rows.append({"model": model, "bootstrap": b, **metrics})
            metric_draws.append(metrics)
            for budget, quality in zip(budgets, q):
                frontier_rows.append({"model": model, "bootstrap": b, "budget": budget, "q": quality})
        metric_frame = pd.DataFrame(metric_draws)
        row = {"model": model, "budgets": ",".join(map(str, budgets)), "n_scenarios": len(scenarios), "n_seeds": len(seeds)}
        point_metrics = aggregate_game(solve_grid(point_q, budgets, values, lambdas, epsilon=0.01))
        for name in metric_frame.columns:
            row[f"{name}_point"] = point_metrics[name]
            row[f"{name}_bootstrap_mean"] = metric_frame[name].mean()
            row[f"{name}_ci_low"] = metric_frame[name].quantile(0.025)
            row[f"{name}_ci_high"] = metric_frame[name].quantile(0.975)
        for k, budget in enumerate(budgets):
            row[f"q_{budget}_point"] = point_q[k]
            row[f"q_{budget}_ci_low"] = np.quantile(q_draws[:, k], 0.025)
            row[f"q_{budget}_ci_high"] = np.quantile(q_draws[:, k], 0.975)
        summary_rows.append(row)
        print(f"Completed hierarchical stochastic bootstrap for {model}")

    pd.DataFrame(frontier_rows).to_csv(tables / "stochastic_frontier_bootstrap.csv", index=False)
    pd.DataFrame(game_rows).to_csv(tables / "stochastic_game_uncertainty_bootstrap.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(tables / "stochastic_game_uncertainty_summary.csv", index=False)


if __name__ == "__main__":
    main()
