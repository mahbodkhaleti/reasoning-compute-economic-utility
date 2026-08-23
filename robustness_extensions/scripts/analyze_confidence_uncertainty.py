#!/usr/bin/env python3
"""Confidence calibration, selective prediction, and adaptive inference analyses.

This layer complements (and does not replace) the paired scenario bootstrap.
It treats correctness of the held-out utility-maximizing bid as the calibration
target and keeps scenario IDs disjoint between calibration and evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import read_jsonl


CONFIDENCE_COLUMNS = {
    "answer_token": "answer_geomean_probability",
    "verbal": "verbal_confidence",
}


def split_name(scenario_id: str) -> str:
    digest = hashlib.sha256(f"confidence-split|{scenario_id}".encode()).hexdigest()
    return "calibration" if int(digest[:8], 16) % 10 < 4 else "evaluation"


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return math.nan
    ranks = pd.Series(score).rank(method="average").to_numpy(float)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def calibration_metrics(frame: pd.DataFrame, bins: int = 10) -> dict[str, float]:
    confidence = frame["confidence"].clip(1e-6, 1 - 1e-6).to_numpy(float)
    target = frame["correct"].to_numpy(float)
    brier = float(np.mean((confidence - target) ** 2))
    nll = float(-np.mean(target * np.log(confidence) + (1 - target) * np.log(1 - confidence)))
    edges = np.linspace(0, 1, bins + 1)
    membership = np.minimum(np.digitize(confidence, edges, right=True), bins) - 1
    membership = np.maximum(membership, 0)
    ece = 0.0
    for b in range(bins):
        mask = membership == b
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - target[mask].mean())
    return {"brier": brier, "nll": nll, "ece_10": float(ece), "auroc": auc_score(target, confidence)}


def scenario_bootstrap_metrics(frame: pd.DataFrame, samples: int, rng: np.random.Generator) -> dict[str, tuple[float, float]]:
    ids = frame["scenario_id"].unique()
    grouped = {sid: frame.loc[frame["scenario_id"] == sid] for sid in ids}
    draws = {name: [] for name in ["brier", "nll", "ece_10", "auroc"]}
    for _ in range(samples):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        draw = pd.concat([grouped[sid] for sid in sampled], ignore_index=True)
        metrics = calibration_metrics(draw)
        for name in draws:
            draws[name].append(metrics[name])
    return {
        name: (float(np.nanquantile(values, 0.025)), float(np.nanquantile(values, 0.975)))
        for name, values in draws.items()
    }


def reliability_rows(frame: pd.DataFrame, bins: int = 10) -> list[dict]:
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for b, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (frame["confidence"] >= low) & (frame["confidence"] <= high if b == bins - 1 else frame["confidence"] < high)
        part = frame.loc[mask]
        rows.append(
            {
                "bin": b,
                "bin_low": low,
                "bin_high": high,
                "n": len(part),
                "mean_confidence": part["confidence"].mean(),
                "empirical_accuracy": part["correct"].mean(),
            }
        )
    return rows


def selective_curve(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0, 1, 21):
        answered = frame["confidence"] >= threshold
        for incorrect_reward in [0.0, -1.0, -20.0]:
            utility = np.where(
                answered,
                np.where(frame["correct"], 1.0, incorrect_reward),
                0.0,
            )
            rows.append(
                {
                    "threshold": threshold,
                    "incorrect_reward": incorrect_reward,
                    "coverage": answered.mean(),
                    "selective_accuracy": frame.loc[answered, "correct"].mean() if answered.any() else math.nan,
                    "selective_risk": 1.0 - frame.loc[answered, "correct"].mean() if answered.any() else math.nan,
                    "mean_decision_utility": utility.mean(),
                }
            )
    return pd.DataFrame(rows)


def calibrated_selective_policy(calibration: pd.DataFrame, evaluation: pd.DataFrame) -> list[dict]:
    rows = []
    for incorrect_reward in [0.0, -1.0, -20.0]:
        curve = selective_curve(calibration)
        candidates = curve.loc[curve["incorrect_reward"] == incorrect_reward]
        chosen = candidates.sort_values(["mean_decision_utility", "coverage", "threshold"], ascending=[False, False, True]).iloc[0]
        threshold = float(chosen["threshold"])
        answered = evaluation["confidence"] >= threshold
        utility = np.where(answered, np.where(evaluation["correct"], 1.0, incorrect_reward), 0.0)
        rows.append(
            {
                "incorrect_reward": incorrect_reward,
                "selected_threshold": threshold,
                "calibration_utility": chosen["mean_decision_utility"],
                "evaluation_coverage": answered.mean(),
                "evaluation_selective_accuracy": evaluation.loc[answered, "correct"].mean() if answered.any() else math.nan,
                "evaluation_mean_decision_utility": utility.mean(),
            }
        )
    return rows


def confidence_weighted_sampling(group: pd.DataFrame) -> list[dict]:
    group = group.sort_values("repeat_seed")
    rows, scores, top_history = [], {}, []
    observations = group.to_dict("records")
    for index, observation in enumerate(observations, start=1):
        bid = observation.get("chosen_bid")
        if pd.notna(bid):
            scores[float(bid)] = scores.get(float(bid), 0.0) + float(observation["confidence"])
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if not ranked:
            continue
        top_bid, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        total_score = sum(scores.values())
        share = top_score / total_score if total_score > 0 else 0.0
        margin = (top_score - second_score) / total_score if total_score > 0 else 0.0
        top_history.append(top_bid)
        converged = index >= 3 and len(set(top_history[-2:])) == 1 and share >= 0.60 and margin >= 0.20
        if converged or index == len(observations):
            selected = next(x for x in observations if pd.notna(x.get("chosen_bid")) and float(x["chosen_bid"]) == top_bid)
            rows.append(
                {
                    "policy": "confidence_weighted_convergent",
                    "samples_used": index,
                    "selected_bid": top_bid,
                    "vote_share": share,
                    "vote_margin": margin,
                    "correct": selected["test_correct"],
                    "utility_ratio": selected["test_utility_ratio"],
                    "regret": selected["test_regret"],
                }
            )
            break
    for count in [1, len(observations)]:
        subset = observations[:count]
        votes: dict[float, float] = {}
        for observation in subset:
            if pd.notna(observation.get("chosen_bid")):
                bid = float(observation["chosen_bid"])
                votes[bid] = votes.get(bid, 0.0) + 1.0
        if votes:
            selected_bid = sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]
            selected = next(x for x in subset if pd.notna(x.get("chosen_bid")) and float(x["chosen_bid"]) == selected_bid)
            rows.append(
                {
                    "policy": f"fixed_{count}_majority",
                    "samples_used": count,
                    "selected_bid": selected_bid,
                    "vote_share": max(votes.values()) / sum(votes.values()),
                    "vote_margin": math.nan,
                    "correct": selected["test_correct"],
                    "utility_ratio": selected["test_utility_ratio"],
                    "regret": selected["test_regret"],
                }
            )
    return rows


def conformal_maxwise_stopping(data: pd.DataFrame, alpha: float = 0.05) -> list[dict]:
    """Maxwise split-conformal abstention over budget-indexed uncertainty paths.

    This mirrors max-over-checkpoints calibration, but the checkpoints are
    independently decoded fixed-budget runs rather than prefixes of one trace.
    A trigger means abstain, not emit the current bid.
    """
    rows = []
    for (model, confidence_type), group in data.groupby(["model", "confidence_type"]):
        wide = group.pivot_table(
            index=["scenario_id", "repeat_seed", "split"],
            columns="budget",
            values=["confidence", "correct"],
            aggfunc="first",
        )
        budgets = sorted(group["budget"].unique())
        max_budget = budgets[-1]
        calibration = wide.loc[wide.index.get_level_values("split") == "calibration"]
        correct_at_max = calibration[("correct", max_budget)] == 1
        maxima = (1.0 - calibration.loc[correct_at_max, "confidence"]).max(axis=1).dropna().sort_values().to_numpy(float)
        if not len(maxima):
            continue
        rank = int(math.ceil((len(maxima) + 1) * (1 - alpha)))
        threshold = math.inf if rank > len(maxima) else float(maxima[rank - 1])
        evaluation = wide.loc[wide.index.get_level_values("split") == "evaluation"]
        for index, observation in evaluation.iterrows():
            trigger_budget = math.nan
            for budget in budgets:
                confidence = observation.get(("confidence", budget), math.nan)
                if pd.notna(confidence) and 1.0 - confidence > threshold:
                    trigger_budget = budget
                    break
            high_correct = bool(observation[("correct", max_budget)])
            rows.append(
                {
                    "model": model,
                    "confidence_type": confidence_type,
                    "scenario_id": index[0],
                    "repeat_seed": index[1],
                    "alpha": alpha,
                    "calibration_threshold": threshold,
                    "triggered_abstention": pd.notna(trigger_budget),
                    "trigger_budget": trigger_budget,
                    "high_budget_correct": high_correct,
                    "premature_abstention": pd.notna(trigger_budget) and high_correct,
                    "detected_high_budget_error": pd.notna(trigger_budget) and not high_correct,
                    "tokens_saved_if_triggered": max_budget - trigger_budget if pd.notna(trigger_budget) else 0,
                }
            )
    return rows


def plot_reliability(frame: pd.DataFrame, output: Path) -> None:
    models = sorted(frame["model"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), sharex=True, sharey=True)
    for ax, model in zip(axes.ravel(), models):
        part = frame.loc[(frame["model"] == model) & (frame["split"] == "evaluation")]
        for confidence_type, marker in [("answer_token", "o"), ("verbal", "s")]:
            typed = part.loc[part["confidence_type"] == confidence_type]
            points = reliability_rows(typed)
            valid = [x for x in points if x["n"] > 0]
            ax.plot([x["mean_confidence"] for x in valid], [x["empirical_accuracy"] for x in valid], marker=marker, label=confidence_type)
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.set_title(model, fontsize=9)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("Mean predicted confidence")
    fig.supylabel("Empirical optimal-bid probability")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="robustness_extensions/server_outputs")
    parser.add_argument("--export-dir", default="robustness_extensions")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    source, export = Path(args.output_dir), Path(args.export_dir)
    tables, raw, figures = export / "tables", export / "raw_results", export / "figures"
    for path in [tables, raw, figures]:
        path.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted((source / "raw").glob("fixed_*_reasoning_stochastic.jsonl")):
        records.extend(read_jsonl(path))
    if not records:
        raise RuntimeError("No stochastic fixed-budget outputs found")
    data = pd.DataFrame(records)
    data["scenario_id"] = data["scenario_id"].astype(str).str.zfill(12)
    data["_confidence_complete"] = data.get("verbal_confidence", pd.Series(index=data.index, dtype=float)).notna().astype(int)
    data = (
        data.sort_values("_confidence_complete")
        .drop_duplicates(["model", "scenario_id", "budget", "repeat_seed"], keep="last")
        .drop(columns="_confidence_complete")
    )
    data["split"] = data["scenario_id"].map(split_name)
    data["correct"] = data["test_correct"].astype(float)
    long_parts = []
    for name, column in CONFIDENCE_COLUMNS.items():
        if column not in data:
            continue
        part = data.copy()
        part["confidence_type"] = name
        part["confidence"] = pd.to_numeric(part[column], errors="coerce")
        long_parts.append(part.loc[part["confidence"].between(0, 1)])
    confidence = pd.concat(long_parts, ignore_index=True)
    confidence.to_csv(raw / "confidence_uncertainty_raw.csv", index=False)

    rng = np.random.default_rng(20260819)
    summary_rows, reliability = [], []
    for keys, group in confidence.groupby(["model", "budget", "confidence_type", "split"]):
        point = calibration_metrics(group)
        intervals = scenario_bootstrap_metrics(group, args.bootstrap_samples, rng)
        row = {"model": keys[0], "budget": keys[1], "confidence_type": keys[2], "split": keys[3], "n": len(group), "n_scenarios": group["scenario_id"].nunique()}
        for name, value in point.items():
            row[name] = value
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = intervals[name]
        summary_rows.append(row)
        reliability.extend({"model": keys[0], "budget": keys[1], "confidence_type": keys[2], "split": keys[3], **x} for x in reliability_rows(group))
    pd.DataFrame(summary_rows).to_csv(tables / "confidence_calibration_summary.csv", index=False)
    reliability_df = pd.DataFrame(reliability)
    reliability_df.to_csv(tables / "confidence_reliability_bins.csv", index=False)
    plot_reliability(confidence, figures / "confidence_reliability_diagram.pdf")

    curve_rows, policy_rows = [], []
    for keys, group in confidence.groupby(["model", "budget", "confidence_type"]):
        calibration = group.loc[group["split"] == "calibration"]
        evaluation = group.loc[group["split"] == "evaluation"]
        curve = selective_curve(evaluation)
        curve.insert(0, "confidence_type", keys[2])
        curve.insert(0, "budget", keys[1])
        curve.insert(0, "model", keys[0])
        curve_rows.append(curve)
        policy_rows.extend({"model": keys[0], "budget": keys[1], "confidence_type": keys[2], **x} for x in calibrated_selective_policy(calibration, evaluation))
    pd.concat(curve_rows, ignore_index=True).to_csv(tables / "selective_risk_coverage.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(tables / "selective_policy_summary.csv", index=False)

    adaptive_rows = []
    for keys, group in confidence.loc[confidence["split"] == "evaluation"].groupby(["model", "budget", "confidence_type", "scenario_id"]):
        adaptive_rows.extend({"model": keys[0], "budget": keys[1], "confidence_type": keys[2], "scenario_id": keys[3], **x} for x in confidence_weighted_sampling(group))
    adaptive = pd.DataFrame(adaptive_rows)
    adaptive.to_csv(tables / "adaptive_sampling_results.csv", index=False)
    (
        adaptive.groupby(["model", "budget", "confidence_type", "policy"], as_index=False)
        .agg(n_scenarios=("scenario_id", "nunique"), mean_samples=("samples_used", "mean"), accuracy=("correct", "mean"), mean_utility_ratio=("utility_ratio", "mean"), mean_regret=("regret", "mean"))
        .to_csv(tables / "adaptive_sampling_summary.csv", index=False)
    )

    conformal = pd.DataFrame(conformal_maxwise_stopping(confidence))
    conformal.to_csv(tables / "conformal_stopping_results.csv", index=False)
    if not conformal.empty:
        conformal_summary = []
        for keys, group in conformal.groupby(["model", "confidence_type", "alpha", "calibration_threshold"]):
            errors = ~group["high_budget_correct"]
            conformal_summary.append(
                {
                    "model": keys[0],
                    "confidence_type": keys[1],
                    "alpha": keys[2],
                    "calibration_threshold": keys[3],
                    "n": len(group),
                    "abstention_rate": group["triggered_abstention"].mean(),
                    "premature_abstention_rate": group["premature_abstention"].mean(),
                    "detection_rate_among_high_budget_errors": group.loc[errors, "detected_high_budget_error"].mean() if errors.any() else math.nan,
                    "mean_tokens_saved": group["tokens_saved_if_triggered"].mean(),
                }
            )
        pd.DataFrame(conformal_summary).to_csv(tables / "conformal_stopping_summary.csv", index=False)
    print(f"Wrote confidence-aware uncertainty outputs to {export}")


if __name__ == "__main__":
    main()
