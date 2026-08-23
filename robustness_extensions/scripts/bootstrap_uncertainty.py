#!/usr/bin/env python3
"""Paired-scenario bootstrap of q(c) and propagation through the compute game.

This script uses only the existing four-model, five-option raw result files.  It
resamples scenario/value identifiers and carries every budget observation for a
selected scenario into each draw.  Gamma, beta, V and lambda remain design
parameters; their grid is not interpreted as a statistical sample.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_FILES = {
    "Phi-mini-reasoning": ("econml_complete_package_FINAL_v2/raw/phi_5bid_repaired.csv", "repaired_regret", "repaired_correct"),
    "Qwen3-4B": ("econml_complete_package_FINAL_v2/raw/qwen_5bid_raw.csv", "regret", "correct"),
    "DeepSeek-R1-Distill-Llama-8B": ("deepseek_third_model_complete/deepseek_5bid_raw.csv", "regret", "correct"),
    "Qwen3-8B": ("qwen3_8b_econml_complete/raw_results/qwen3_8b_5bid_raw.csv", "regret", "correct"),
}


def load_panel(root: Path, model: str, relpath: str, regret_col: str, accuracy_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(root / relpath)
    frame = frame.rename(columns={"thinking_budget": "budget", regret_col: "regret", accuracy_col: "accuracy"})
    required = {"value", "budget", "utility_ratio", "regret", "accuracy"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{model}: missing columns {sorted(missing)}")
    if frame.duplicated(["value", "budget"]).any():
        raise ValueError(f"{model}: duplicate scenario-budget rows prevent paired bootstrap")
    q_panel = frame.pivot(index="value", columns="budget", values="utility_ratio").sort_index(axis=1)
    r_panel = frame.pivot(index="value", columns="budget", values="regret").reindex(columns=q_panel.columns)
    a_panel = frame.pivot(index="value", columns="budget", values="accuracy").reindex(columns=q_panel.columns)
    if q_panel.isna().any().any() or r_panel.isna().any().any() or a_panel.isna().any().any():
        raise ValueError(f"{model}: incomplete repeated-measures panel")
    return (
        q_panel.index.to_numpy(),
        q_panel.columns.to_numpy(dtype=int),
        q_panel.to_numpy(dtype=float),
        r_panel.to_numpy(dtype=float),
        a_panel.to_numpy(dtype=float),
    )


def percentile_columns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.nanstd(values, axis=0, ddof=1),
        np.nanquantile(values, 0.025, axis=0),
        np.nanquantile(values, 0.975, axis=0),
    )


def solve_grid(q: np.ndarray, budgets: np.ndarray, values: np.ndarray, lambdas: np.ndarray, beta: float = 1.0, gamma: float = 1.0, epsilon: float = 1e-8) -> dict[str, np.ndarray]:
    """Solve the symmetric pure game and planner problem with deterministic ties.

    The lowest budget is selected among exact payoff ties.  If multiple symmetric
    pure equilibria remain, the lowest-budget equilibrium is reported and the
    equilibrium count is retained for auditing.
    """
    grid_v, grid_lam = np.meshgrid(values, lambdas, indexing="ij")
    v = grid_v.ravel()
    lam = grid_lam.ravel()
    qg = np.power(np.maximum(q, 0.0), gamma)
    denom = qg[:, None] + qg[None, :]
    probability = np.divide(qg[:, None], denom, out=np.full_like(denom, 0.5), where=denom > 0)

    private = v[:, None, None] * probability[None, :, :] - lam[:, None, None] * budgets[None, :, None]
    best_response = np.argmax(private, axis=1)
    symmetric = best_response == np.arange(len(budgets))[None, :]
    equilibrium_count = symmetric.sum(axis=1)
    valid = equilibrium_count > 0
    ne_index = np.argmax(symmetric, axis=1)
    ne_index = np.where(valid, ne_index, -1)

    weighted_quality = beta * (probability * q[:, None] + (1.0 - probability) * q[None, :])
    total_compute = budgets[:, None] + budgets[None, :]
    planner_objective = weighted_quality.ravel()[None, :] - lam[:, None] * total_compute.ravel()[None, :]
    planner_index = np.argmax(planner_objective, axis=1)
    opt_i, opt_j = np.unravel_index(planner_index, total_compute.shape)
    opt_total = total_compute[opt_i, opt_j].astype(float)
    w_opt = planner_objective[np.arange(len(v)), planner_index]

    safe_ne = np.maximum(ne_index, 0)
    ne_total = np.where(valid, 2.0 * budgets[safe_ne], np.nan)
    w_ne = np.where(
        valid,
        weighted_quality[safe_ne, safe_ne] - lam * 2.0 * budgets[safe_ne],
        np.nan,
    )
    absolute_loss = w_opt - w_ne
    relative_loss = np.divide(
        absolute_loss,
        np.abs(w_opt),
        out=np.full_like(absolute_loss, np.nan),
        where=np.abs(w_opt) > epsilon,
    )
    porc = np.divide(ne_total, opt_total, out=np.full_like(ne_total, np.nan), where=opt_total > 0)
    return {
        "V": v,
        "lambda": lam,
        "ne_compute": np.where(valid, budgets[safe_ne], np.nan),
        "ne_total": ne_total,
        "opt_c1": budgets[opt_i],
        "opt_c2": budgets[opt_j],
        "opt_total": opt_total,
        "PoRC": porc,
        "arms_race": ne_total > opt_total,
        "W_NE": w_ne,
        "W_OPT": w_opt,
        "absolute_welfare_loss": absolute_loss,
        "relative_welfare_loss": relative_loss,
        "positive_welfare_loss": absolute_loss > 1e-12,
        "equilibrium_count": equilibrium_count,
        "valid_equilibrium": valid,
    }


def aggregate_game(solution: dict[str, np.ndarray]) -> dict[str, float]:
    valid = solution["valid_equilibrium"]
    arms = solution["arms_race"] & valid
    return {
        "arms_race_fraction": float(np.mean(solution["arms_race"][valid])) if valid.any() else math.nan,
        "mean_porc_conditional_arms_race": float(np.mean(solution["PoRC"][arms])) if arms.any() else math.nan,
        "mean_absolute_welfare_loss": float(np.mean(solution["absolute_welfare_loss"][valid])) if valid.any() else math.nan,
        "mean_relative_welfare_loss": float(np.nanmean(solution["relative_welfare_loss"][valid])) if valid.any() else math.nan,
        "positive_welfare_loss_fraction": float(np.mean(solution["positive_welfare_loss"][valid])) if valid.any() else math.nan,
        "valid_grid_fraction": float(np.mean(valid)),
        "multiple_equilibrium_fraction": float(np.mean(solution["equilibrium_count"][valid] > 1)) if valid.any() else math.nan,
    }


def plot_frontiers(summary: pd.DataFrame, output: Path) -> None:
    models = list(MODEL_FILES)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), sharex=True, sharey=True)
    for ax, model in zip(axes.ravel(), models):
        part = summary.loc[summary["model"] == model].sort_values("budget")
        ax.plot(part["budget"], part["q_point"], marker="o", linewidth=1.4)
        ax.fill_between(part["budget"], part["q_ci_low"], part["q_ci_high"], alpha=0.22)
        ax.set_title(model, fontsize=9)
        ax.grid(alpha=0.2)
    fig.supxlabel("Reasoning budget (tokens)")
    fig.supylabel("Mean utility ratio q(c)")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(grid: pd.DataFrame, value_col: str, label: str, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex=True, sharey=True)
    finite = grid[value_col].replace([np.inf, -np.inf], np.nan)
    vmin, vmax = float(finite.quantile(0.02)), float(finite.quantile(0.98))
    image = None
    for ax, model in zip(axes.ravel(), MODEL_FILES):
        part = grid.loc[grid["model"] == model]
        pivot = part.pivot(index="V", columns="lambda", values=value_col)
        image = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(model, fontsize=9)
        ax.set_xticks([0, len(pivot.columns) - 1], [f"{pivot.columns[0]:.4f}", f"{pivot.columns[-1]:.3f}"])
        ax.set_yticks([0, len(pivot.index) - 1], [f"{pivot.index[0]:.0f}", f"{pivot.index[-1]:.0f}"])
    fig.supxlabel("Per-token cost lambda")
    fig.supylabel("Private prize V")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label=label, shrink=0.84)
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.09, top=0.94, wspace=0.12, hspace=0.22)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="robustness_extensions")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = (root / args.output_dir).resolve()
    uncertainty_dir, figures_dir = out / "uncertainty", out / "figures"
    uncertainty_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((out / "scripts" / "config.json").read_text(encoding="utf-8"))
    samples = args.samples or int(config["uncertainty"]["bootstrap_samples"])
    seed = args.seed or int(config["uncertainty"]["bootstrap_seed"])
    epsilon = float(config["uncertainty"]["relative_welfare_epsilon"])
    rng = np.random.default_rng(seed)
    values = np.arange(25, 1001, 25, dtype=float)
    lambdas = np.linspace(0.0001, 0.03, 60)

    q_boot_rows: list[pd.DataFrame] = []
    q_summary_rows: list[dict[str, float | str | int]] = []
    game_boot_rows: list[dict[str, float | str | int]] = []
    game_summary_rows: list[dict[str, float | str]] = []
    grid_rows: list[pd.DataFrame] = []
    welfare_uncertainty_rows: list[pd.DataFrame] = []

    for model, (relpath, regret_col, accuracy_col) in MODEL_FILES.items():
        scenarios, budgets, q_panel, regret_panel, accuracy_panel = load_panel(root, model, relpath, regret_col, accuracy_col)
        draw_indices = rng.integers(0, len(scenarios), size=(samples, len(scenarios)))
        q_boot = q_panel[draw_indices, :].mean(axis=1)
        regret_boot = regret_panel[draw_indices, :].mean(axis=1)
        accuracy_boot = accuracy_panel[draw_indices, :].mean(axis=1)
        q_point, regret_point, accuracy_point = q_panel.mean(axis=0), regret_panel.mean(axis=0), accuracy_panel.mean(axis=0)
        q_sd, q_lo, q_hi = percentile_columns(q_boot)
        r_sd, r_lo, r_hi = percentile_columns(regret_boot)
        a_sd, a_lo, a_hi = percentile_columns(accuracy_boot)
        q_boot_rows.append(
            pd.DataFrame(
                {
                    "model": np.repeat(model, samples * len(budgets)),
                    "bootstrap": np.repeat(np.arange(samples), len(budgets)),
                    "budget": np.tile(budgets, samples),
                    "q": q_boot.ravel(),
                    "mean_regret": regret_boot.ravel(),
                    "mean_accuracy": accuracy_boot.ravel(),
                }
            )
        )
        for k, budget in enumerate(budgets):
            q_summary_rows.append(
                {
                    "model": model,
                    "budget": int(budget),
                    "n_scenarios": int(len(scenarios)),
                    "q_point": q_point[k],
                    "q_bootstrap_sd": q_sd[k],
                    "q_ci_low": q_lo[k],
                    "q_ci_high": q_hi[k],
                    "regret_point": regret_point[k],
                    "regret_bootstrap_sd": r_sd[k],
                    "regret_ci_low": r_lo[k],
                    "regret_ci_high": r_hi[k],
                    "accuracy_point": accuracy_point[k],
                    "accuracy_bootstrap_sd": a_sd[k],
                    "accuracy_ci_low": a_lo[k],
                    "accuracy_ci_high": a_hi[k],
                }
            )

        point_solution = solve_grid(q_point, budgets, values, lambdas, epsilon=epsilon)
        point_grid = pd.DataFrame(point_solution)
        point_grid.insert(0, "model", model)
        grid_rows.append(point_grid)
        point_metrics = aggregate_game(point_solution)

        cell_metrics = {name: np.full((samples, len(point_solution["V"])), np.nan) for name in ["PoRC", "absolute_welfare_loss", "relative_welfare_loss"]}
        metric_names = list(point_metrics)
        metric_draws = {name: np.full(samples, np.nan) for name in metric_names}
        for b in range(samples):
            solved = solve_grid(q_boot[b], budgets, values, lambdas, epsilon=epsilon)
            aggregated = aggregate_game(solved)
            for name in metric_names:
                metric_draws[name][b] = aggregated[name]
            for name in cell_metrics:
                cell_metrics[name][b, :] = solved[name]
            game_boot_rows.append({"model": model, "bootstrap": b, **aggregated})

        summary_row: dict[str, float | str] = {"model": model}
        for name, draws in metric_draws.items():
            summary_row[f"{name}_point"] = point_metrics[name]
            summary_row[f"{name}_bootstrap_mean"] = float(np.nanmean(draws))
            summary_row[f"{name}_ci_low"] = float(np.nanquantile(draws, 0.025))
            summary_row[f"{name}_ci_high"] = float(np.nanquantile(draws, 0.975))
        game_summary_rows.append(summary_row)

        cell_frame = pd.DataFrame({"model": model, "V": point_solution["V"], "lambda": point_solution["lambda"]})
        for name, draws in cell_metrics.items():
            cell_frame[f"{name}_point"] = point_solution[name]
            cell_frame[f"{name}_bootstrap_mean"] = np.nanmean(draws, axis=0)
            cell_frame[f"{name}_ci_low"] = np.nanquantile(draws, 0.025, axis=0)
            cell_frame[f"{name}_ci_high"] = np.nanquantile(draws, 0.975, axis=0)
        welfare_uncertainty_rows.append(cell_frame)
        print(f"Completed {model}: {samples} paired bootstrap samples")

    q_boot_df = pd.concat(q_boot_rows, ignore_index=True)
    q_summary_df = pd.DataFrame(q_summary_rows)
    game_boot_df = pd.DataFrame(game_boot_rows)
    game_summary_df = pd.DataFrame(game_summary_rows)
    grid_df = pd.concat(grid_rows, ignore_index=True)
    welfare_uncertainty_df = pd.concat(welfare_uncertainty_rows, ignore_index=True)
    q_boot_df.to_csv(uncertainty_dir / "q_frontier_bootstrap.csv", index=False)
    q_summary_df.to_csv(uncertainty_dir / "q_frontier_uncertainty_summary.csv", index=False)
    game_boot_df.to_csv(uncertainty_dir / "game_uncertainty_bootstrap.csv", index=False)
    game_summary_df.to_csv(uncertainty_dir / "game_uncertainty_summary.csv", index=False)
    grid_df.to_csv(uncertainty_dir / "welfare_loss_grid.csv", index=False)
    welfare_uncertainty_df.to_csv(uncertainty_dir / "welfare_loss_uncertainty.csv", index=False)
    plot_frontiers(q_summary_df, figures_dir / "model_compute_quality_uncertainty.pdf")
    plot_heatmap(grid_df, "PoRC", "PoRC (compute ratio)", figures_dir / "porc_heatmap.pdf")
    plot_heatmap(grid_df, "absolute_welfare_loss", "Absolute welfare loss", figures_dir / "absolute_welfare_loss_heatmap.pdf")
    plot_heatmap(grid_df, "relative_welfare_loss", "Relative welfare loss", figures_dir / "relative_welfare_loss_heatmap.pdf")


if __name__ == "__main__":
    main()
