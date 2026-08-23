#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_dirs, load_config, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(args.output_dir)
    rows = []
    for path in sorted(paths["direct"].glob("direct_*.jsonl")):
        rows.extend(read_jsonl(path))
    if not rows:
        raise RuntimeError("No direct-game outputs found")
    df = pd.DataFrame(rows)
    if "bid_readout_method" not in df:
        raise RuntimeError("Direct-game outputs predate the constrained bid readout; rerun them")
    df = df.loc[df["bid_readout_method"] == "constrained_option_letter_next_token"].copy()
    df = df.drop_duplicates(["model", "mode", "experiment_seed", "round"], keep="last")
    if df.empty:
        raise RuntimeError("No constrained-readout direct-game outputs found")
    df["total_budget"] = df["budget_a"] + df["budget_b"]
    df["both_choice_valid"] = df["choice_valid_a"] & df["choice_valid_b"]
    df["both_bid_valid"] = df["bid_valid_a"] & df["bid_valid_b"]
    agreement_a = np.isclose(df["freeform_bid_a"], df["bid_a"], equal_nan=False)
    agreement_b = np.isclose(df["freeform_bid_b"], df["bid_b"], equal_nan=False)
    df["constrained_freeform_agreement"] = (agreement_a.astype(float) + agreement_b.astype(float)) / 2.0
    df["round_fraction"] = df["round"] / df.groupby(["model", "mode", "experiment_seed"])["round"].transform("max").clip(lower=1)
    df["period"] = np.where(df["round_fraction"] < 0.25, "early", np.where(df["round_fraction"] > 0.75, "late", "middle"))

    summary = (
        df.groupby(["model", "mode"], as_index=False)
        .agg(
            rounds=("round", "size"),
            mean_total_budget=("total_budget", "mean"),
            median_total_budget=("total_budget", "median"),
            choice_valid_rate=("both_choice_valid", "mean"),
            bid_valid_rate=("both_bid_valid", "mean"),
            constrained_freeform_agreement=("constrained_freeform_agreement", "mean"),
            mean_compute_cost_usd=("compute_cost_usd", "mean"),
            mean_allocative_loss_usd=("allocative_loss_usd", "mean"),
            mean_total_welfare_loss_usd=("total_welfare_loss_usd", "mean"),
            mean_social_welfare_usd=("social_welfare_usd", "mean"),
            mean_private_payoff_a=("private_payoff_a", "mean"),
            mean_private_payoff_b=("private_payoff_b", "mean"),
        )
    )
    summary.to_csv(paths["tables"] / "direct_game_summary.csv", index=False)

    dynamics = (
        df.loc[df["period"].isin(["early", "late"])]
        .groupby(["model", "mode", "period"], as_index=False)
        .agg(mean_total_budget=("total_budget", "mean"), mean_welfare_loss_usd=("total_welfare_loss_usd", "mean"))
    )
    dynamics.to_csv(paths["tables"] / "direct_game_dynamics.csv", index=False)

    # Ex-post planner among the actually executed adaptive/fixed-low/fixed-high policies.
    key = ["model", "experiment_seed", "round"]
    planner_rows = []
    for keys, group in df.groupby(key):
        if set(group["mode"]) != {"adaptive", "fixed_low", "fixed_high"}:
            continue
        planner = group.loc[group["social_welfare_usd"].idxmax()]
        adaptive = group.loc[group["mode"] == "adaptive"].iloc[0]
        planner_rows.append(
            {
                "model": keys[0],
                "experiment_seed": keys[1],
                "round": keys[2],
                "adaptive_total_budget": adaptive["total_budget"],
                "planner_policy": planner["mode"],
                "planner_total_budget": planner["total_budget"],
                "adaptive_social_welfare_usd": adaptive["social_welfare_usd"],
                "planner_social_welfare_usd": planner["social_welfare_usd"],
                "welfare_gap_usd": planner["social_welfare_usd"] - adaptive["social_welfare_usd"],
                "observed_overinvestment": adaptive["total_budget"] > planner["total_budget"],
                "compute_ratio": (
                    adaptive["total_budget"] / planner["total_budget"]
                    if planner["total_budget"] > 0
                    else math.nan
                ),
            }
        )
    planner_df = pd.DataFrame(planner_rows)
    planner_df.to_csv(paths["tables"] / "direct_policy_welfare_comparison.csv", index=False)
    if not planner_df.empty:
        planner_summary = (
            planner_df.groupby("model", as_index=False)
            .agg(
                arms_race_fraction=("observed_overinvestment", "mean"),
                mean_compute_ratio=("compute_ratio", "mean"),
                mean_welfare_gap_usd=("welfare_gap_usd", "mean"),
                max_welfare_gap_usd=("welfare_gap_usd", "max"),
            )
        )
        planner_summary.to_csv(paths["tables"] / "direct_policy_welfare_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    adaptive = df.loc[df["mode"] == "adaptive"]
    for model, group in adaptive.groupby("model"):
        trajectory = group.groupby("round")["total_budget"].mean()
        plt.plot(trajectory.index, trajectory.values, label=model)
    plt.xlabel("Repeated-auction round")
    plt.ylabel("Mean endogenous total reasoning budget")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(paths["figures"] / "endogenous_budget_dynamics.png", dpi=200)
    plt.close()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
