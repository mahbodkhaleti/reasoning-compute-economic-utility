#!/usr/bin/env python3
"""CPU-only primary analysis and paper-ready output for synthetic runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "phi4_mini_reasoning": "Phi-4-mini-reasoning",
    "qwen3_4b": "Qwen3-4B",
    "deepseek_r1_llama_8b": "DeepSeek-R1-Distill-Llama-8B",
    "qwen3_8b": "Qwen3-8B",
    "qwen3_14b": "Qwen3-14B",
    "granite_3_2_8b_instruct": "Granite-3.2-8B-Instruct",
}
BUDGETS = [0, 64, 128, 256, 384, 512, 768]


def read_frame(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "raw").glob("synthetic_*_deterministic.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No synthetic result rows under {root / 'raw'}")
    frame = pd.DataFrame(rows).drop_duplicates(
        ["model_slug", "scenario_id", "reasoning_budget", "option_order_seed"], keep="last"
    )
    return frame


def paired_bootstrap(panel: pd.DataFrame, samples: int, seed: int) -> dict:
    pivot = panel.pivot(index="scenario_id", columns="reasoning_budget", values="utility_ratio")
    paired = pivot.dropna(subset=[0, 768])
    diffs = (paired[768] - paired[0]).to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(diffs, size=(samples, len(diffs)), replace=True), axis=1)
    return {
        "q0": float(paired[0].mean()),
        "q768": float(paired[768].mean()),
        "difference": float(diffs.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "paired_scenarios": int(len(diffs)),
        "bootstrap_draws": samples,
    }


def latex_rows(frame: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> str:
    lines = []
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column == "model":
                values.append(str(value).replace("_", "\\_"))
            elif pd.isna(value):
                values.append("--")
            else:
                values.append(format(value, formats.get(column, ".3f")))
        lines.append(" & ".join(values) + r" \\")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-ready", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    tables, figures = args.output_dir / "tables", args.output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    args.paper_ready.mkdir(parents=True, exist_ok=True)

    raw = read_frame(args.output_dir)
    expected_hashes = raw["benchmark_sha256"].dropna().unique()
    if len(expected_hashes) != 1:
        raise RuntimeError(f"Mixed benchmark hashes: {expected_hashes}")
    scenario_order = raw.groupby(["model_slug", "scenario_id", "reasoning_budget"], as_index=False).agg(
        utility_ratio=("utility_ratio", "mean"),
        regret=("regret", "mean"),
        exact_optimality=("exact_optimality", "mean"),
    )
    frontier = scenario_order.groupby(["model_slug", "reasoning_budget"], as_index=False).agg(
        q=("utility_ratio", "mean"), mean_regret=("regret", "mean"), exact_optimality=("exact_optimality", "mean")
    ).rename(columns={"model_slug": "model", "reasoning_budget": "budget"})
    frontier["model_label"] = frontier["model"].map(LABELS).fillna(frontier["model"])
    frontier["exact_optimality_percent"] = 100 * frontier["exact_optimality"]
    frontier.to_csv(tables / "synthetic_deterministic_frontier.csv", index=False)
    frontier.to_csv(args.paper_ready / "deterministic_frontier.csv", index=False)

    endpoint_rows, slope_rows = [], []
    for model, part in scenario_order.groupby("model_slug"):
        result = paired_bootstrap(part, args.bootstrap_samples, args.seed)
        result.update(model=model, model_label=LABELS.get(model, model), benchmark_sha256=expected_hashes[0])
        endpoint_rows.append(result)
        model_frontier = frontier.loc[(frontier["model"] == model) & (frontier["budget"] > 0)].sort_values("budget")
        slope = float(np.polyfit(np.log(model_frontier["budget"].to_numpy(float)), model_frontier["q"].to_numpy(float), 1)[0])
        slope_rows.append({"model": model, "model_label": LABELS.get(model, model), "log_budget_slope": slope})
    endpoints = pd.DataFrame(endpoint_rows)
    slopes = pd.DataFrame(slope_rows)
    endpoints.to_csv(tables / "synthetic_endpoint_bootstrap.csv", index=False)
    slopes.to_csv(tables / "synthetic_log_budget_slopes.csv", index=False)
    endpoints.to_csv(args.paper_ready / "endpoint_table.csv", index=False)
    slopes.to_csv(args.paper_ready / "log_budget_slope.csv", index=False)
    frontier[["model", "budget", "exact_optimality_percent"]].to_csv(args.paper_ready / "exact_optimality.csv", index=False)
    frontier[["model", "budget", "mean_regret"]].to_csv(args.paper_ready / "regret.csv", index=False)
    (args.paper_ready / "endpoint_table.tex").write_text(
        latex_rows(endpoints, ["model", "q0", "q768", "difference", "ci_low", "ci_high"], {}), encoding="utf-8"
    )
    (args.paper_ready / "deterministic_frontier.tex").write_text(
        latex_rows(frontier, ["model", "budget", "q", "mean_regret", "exact_optimality_percent"], {"budget": ".0f", "exact_optimality_percent": ".1f"}), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for model, part in frontier.groupby("model"):
        part = part.sort_values("budget")
        ax.plot(part["budget"], part["q"], marker="o", linewidth=1.5, label=LABELS.get(model, model))
    ax.set_xlabel("Assigned reasoning tokens")
    ax.set_ylabel("Mean utility ratio")
    ax.set_title("Synthetic IPV deterministic compute-utility frontiers")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figures / f"synthetic_deterministic_frontier.{suffix}", dpi=300)
        fig.savefig(args.paper_ready / f"deterministic_frontier.{suffix}", dpi=300)
    plt.close(fig)

    summary = [
        "# Synthetic IPV robustness results",
        "",
        f"Benchmark manifest SHA256: `{expected_hashes[0]}`.",
        "",
        "This is a synthetic deterministic-frontier robustness analysis. Strategic results are conditional on the specified normalized contest.",
        "",
    ]
    for row in endpoint_rows:
        direction = "positive" if row["difference"] > 0 else "negative" if row["difference"] < 0 else "zero"
        summary += [
            f"## {row['model_label']}", "",
            f"The 0-to-768 endpoint difference is {row['difference']:.3f} ({direction}); paired 95% CI [{row['ci_low']:.3f}, {row['ci_high']:.3f}].",
            "",
        ]
        matched = raw.loc[raw["model_slug"] == row["model"], "reasoning_protocol_matched"].dropna()
        if not matched.empty and not bool(matched.all()):
            summary += ["This checkpoint uses a materially different reasoning-template protocol and is reported as a protocol sensitivity.", ""]
    (args.paper_ready / "RESULTS_SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps({"models": sorted(raw["model_slug"].unique()), "rows": len(raw), "benchmark_sha256": expected_hashes[0]}, indent=2))


if __name__ == "__main__":
    main()
