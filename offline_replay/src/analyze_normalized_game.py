#!/usr/bin/env python3
"""Enumerate the dimensionless strategic compute game from saved frontiers.

No model is loaded or queried.  The script reads the completed stochastic
constrained-readout files, forms q(c), and solves the game over (kappa, rho,
gamma), where kappa=lambda/V and rho=beta/V.  It also derives the exact
piecewise-constant transition thresholds induced by the discrete compute menu
and verifies the normalized solver against the legacy (V, lambda, beta) solver.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "deepseek_r1_llama_8b": "DeepSeek-8B",
    "phi4_mini_reasoning": "Phi-mini",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
    "qwen3_14b": "Qwen3-14B",
    "granite_3_2_8b_instruct": "Granite-8B",
}
GAMMAS = (0.5, 1.0, 2.0, 4.0)
KAPPA_BOUNDS = (5e-9, 1.2e-3)
RHO_BOUNDS = (1e-3, 40.0)
PHASE_NAMES = {-1: "Undercompute", 0: "Aligned", 1: "Overcompute"}
PHASE_CMAP = ListedColormap(["#3B82B8", "#D9D9D9", "#D97706"])


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_stochastic(output: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted((output / "raw").glob("fixed_*_reasoning_stochastic.jsonl")):
        rows.extend(read_jsonl(path))
    if not rows:
        raise RuntimeError(f"No stochastic outputs found under {output / 'raw'}")
    frame = pd.DataFrame(rows)
    frame["scenario_id"] = frame["scenario_id"].astype(str).str.zfill(12)
    return frame.drop_duplicates(
        ["model", "scenario_id", "budget", "repeat_seed"], keep="last"
    )


def point_frontiers(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    frontiers = {}
    for model, group in frame.groupby("model"):
        means = group.groupby("budget")["test_utility_ratio"].mean().sort_index()
        frontiers[model] = (means.index.to_numpy(int), means.to_numpy(float))
    return frontiers


def game_primitives(q: np.ndarray, budgets: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qg = np.maximum(q, 0.0) ** gamma
    denom = qg[:, None] + qg[None, :]
    probability = np.divide(
        qg[:, None], denom, out=np.full_like(denom, 0.5), where=denom > 0
    )
    selected_quality = probability * q[:, None] + (1.0 - probability) * q[None, :]
    total_compute = budgets[:, None] + budgets[None, :]
    return probability, selected_quality, total_compute


def selected_equilibrium(probability: np.ndarray, budgets: np.ndarray, kappas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the legacy solver: lowest-index symmetric equilibrium is retained."""
    private = probability[None, :, :] - kappas[:, None, None] * budgets[None, :, None]
    best_response = np.argmax(private, axis=1)
    symmetric = best_response == np.arange(len(budgets))[None, :]
    count = symmetric.sum(axis=1)
    index = np.where(count > 0, np.argmax(symmetric, axis=1), -1)
    return index, count


def solve_normalized(
    q: np.ndarray,
    budgets: np.ndarray,
    kappas: np.ndarray,
    rhos: np.ndarray,
    gamma: float,
) -> pd.DataFrame:
    probability, quality, total_compute = game_primitives(q, budgets, gamma)
    ne_index_by_kappa, equilibrium_count = selected_equilibrium(probability, budgets, kappas)
    grid_kappa, grid_rho = np.meshgrid(kappas, rhos, indexing="xy")
    flat_kappa, flat_rho = grid_kappa.ravel(), grid_rho.ravel()

    ne_index = np.tile(ne_index_by_kappa, len(rhos))
    valid = ne_index >= 0
    safe_ne = np.maximum(ne_index, 0)
    ne_total = np.where(valid, 2.0 * budgets[safe_ne], np.nan)
    q_ne = quality[safe_ne, safe_ne]
    w_ne = np.where(valid, flat_rho * q_ne - flat_kappa * ne_total, np.nan)

    planner_objective = (
        flat_rho[:, None] * quality.ravel()[None, :]
        - flat_kappa[:, None] * total_compute.ravel()[None, :]
    )
    planner_index = np.argmax(planner_objective, axis=1)
    opt_i, opt_j = np.unravel_index(planner_index, total_compute.shape)
    opt_total = total_compute[opt_i, opt_j].astype(float)
    w_opt = planner_objective[np.arange(len(flat_kappa)), planner_index]

    phase = np.sign(ne_total - opt_total)
    phase = np.where(valid, phase, np.nan)
    return pd.DataFrame(
        {
            "kappa": flat_kappa,
            "rho": flat_rho,
            "gamma": gamma,
            "ne_total": ne_total,
            "planner_total": opt_total,
            "phase_code": phase,
            "phase": [PHASE_NAMES.get(int(x), "No symmetric pure NE") if np.isfinite(x) else "No symmetric pure NE" for x in phase],
            "normalized_welfare_ne": w_ne,
            "normalized_welfare_opt": w_opt,
            "normalized_welfare_loss": w_opt - w_ne,
            "equilibrium_count": np.tile(equilibrium_count, len(rhos)),
        }
    )


def _critical_values(values: np.ndarray, costs: np.ndarray) -> list[float]:
    candidates: set[float] = set()
    for a in range(len(values)):
        for b in range(a):
            delta_cost = float(costs[a] - costs[b])
            if delta_cost == 0:
                continue
            threshold = float((values[a] - values[b]) / delta_cost)
            if np.isfinite(threshold) and threshold > 0:
                candidates.add(threshold)
    return sorted(candidates)


def _interval_states(boundaries: list[float], state_function) -> list[object]:
    points = [0.5 * boundaries[0]] if boundaries else [1.0]
    points += [np.sqrt(a * b) for a, b in zip(boundaries[:-1], boundaries[1:])]
    if boundaries:
        points.append(2.0 * boundaries[-1])
    return [state_function(point) for point in points]


def exact_transition_rows(model: str, q: np.ndarray, budgets: np.ndarray, gamma: float) -> list[dict]:
    probability, quality, total_compute = game_primitives(q, budgets, gamma)
    rows: list[dict] = []

    # A private transition can occur only where two actions tie against a fixed
    # opposing action: kappa=(Delta P)/(Delta c).
    private_boundaries: set[float] = set()
    for opponent in range(len(budgets)):
        private_boundaries.update(_critical_values(probability[:, opponent], budgets))
    private_sorted = sorted(private_boundaries)

    def ne_state(kappa: float):
        index, count = selected_equilibrium(probability, budgets, np.array([kappa]))
        return int(budgets[index[0]]) if count[0] > 0 else None

    ne_states = _interval_states(private_sorted, ne_state)
    for boundary, before, after in zip(private_sorted, ne_states[:-1], ne_states[1:]):
        if before != after:
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "gamma": gamma,
                    "boundary_type": "private_NE_kappa",
                    "threshold": boundary,
                    "compute_before": before,
                    "compute_after": after,
                    "inequality": "kappa < DeltaP/DeltaC favors the higher-payoff deviation",
                }
            )

    # The planner depends on r=lambda/beta=kappa/rho.  For a fixed total
    # compute, only its highest-quality allocation can be optimal.
    totals = np.unique(total_compute)
    best_quality = np.array([quality[total_compute == total].max() for total in totals])
    planner_sorted = _critical_values(best_quality, totals)

    def planner_state(ratio: float):
        objective = quality.ravel() - ratio * total_compute.ravel()
        index = int(np.argmax(objective))
        return int(total_compute.ravel()[index])

    planner_states = _interval_states(planner_sorted, planner_state)
    for boundary, before, after in zip(planner_sorted, planner_states[:-1], planner_states[1:]):
        if before != after:
            rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "gamma": gamma,
                    "boundary_type": "planner_kappa_over_rho",
                    "threshold": boundary,
                    "compute_before": before,
                    "compute_after": after,
                    "inequality": "kappa/rho < DeltaQ/DeltaC favors the higher-compute allocation",
                }
            )
    return rows


def plot_phase_panels(
    phases: pd.DataFrame,
    models: list[str],
    gammas: list[float],
    output: Path,
    show_legacy_grid: bool,
) -> None:
    single_gamma = len(gammas) == 1
    if single_gamma:
        columns = min(2, len(models))
        rows = int(np.ceil(len(models) / columns))
        fig, axes = plt.subplots(
            rows, columns, figsize=(3.6 * columns, 2.85 * rows),
            squeeze=False, sharex=True, sharey=True,
        )
        panel_specs = [
            (axes.ravel()[index], model, gammas[0], index // columns, index % columns)
            for index, model in enumerate(models)
        ]
        for ax in axes.ravel()[len(models):]:
            ax.set_visible(False)
    else:
        fig, axes = plt.subplots(
            len(models), len(gammas),
            figsize=(3.15 * len(gammas), 2.45 * len(models)),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        panel_specs = [
            (axes[row_index, col_index], model, gamma, row_index, col_index)
            for row_index, model in enumerate(models)
            for col_index, gamma in enumerate(gammas)
        ]
    legacy_v = np.arange(25, 1001, 25, dtype=float)
    legacy_lambda = np.linspace(0.0001, 0.03, 60)
    legacy_V, legacy_lam = np.meshgrid(legacy_v, legacy_lambda, indexing="ij")
    legacy_kappa = (legacy_lam / legacy_V).ravel()
    legacy_rho = (1.0 / legacy_V).ravel()

    for ax, model, gamma, row_index, col_index in panel_specs:
        part = phases.loc[(phases["model"] == model) & (phases["gamma"] == gamma)]
        pivot = part.pivot(index="rho", columns="kappa", values="phase_code")
        ax.pcolormesh(
            pivot.columns.to_numpy(float),
            pivot.index.to_numpy(float),
            pivot.to_numpy(float),
            cmap=PHASE_CMAP,
            vmin=-1,
            vmax=1,
            shading="nearest",
            rasterized=True,
        )
        if show_legacy_grid:
            ax.scatter(legacy_kappa, legacy_rho, s=0.4, c="black", alpha=0.10, rasterized=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(which="major", color="white", linewidth=0.35, alpha=0.45)
        if single_gamma:
            ax.set_title(MODEL_LABELS[model], fontsize=9)
        else:
            if row_index == 0:
                ax.set_title(rf"$\gamma={gamma:g}$", fontsize=9)
            if col_index == 0:
                ax.text(
                    -0.27, 0.5, MODEL_LABELS[model], transform=ax.transAxes,
                    rotation=90, va="center", ha="center", fontsize=9,
                )
    fig.supxlabel(r"Relative compute cost $\kappa=\lambda/V$")
    fig.supylabel(r"Relative social quality value $\rho=\beta/V$")
    legend = [
        Patch(facecolor=PHASE_CMAP(0.0), label="Undercompute"),
        Patch(facecolor=PHASE_CMAP(0.5), label="Aligned"),
        Patch(facecolor=PHASE_CMAP(1.0), label="Overcompute"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False, fontsize=8)
    if single_gamma:
        fig.suptitle(r"Normalized compute-allocation regions ($\gamma=1$)", y=0.965, fontsize=10)
        fig.subplots_adjust(left=0.11, right=0.99, bottom=0.11, top=0.87, wspace=0.08, hspace=0.16)
    else:
        fig.subplots_adjust(left=0.11, right=0.99, bottom=0.09, top=0.91, wspace=0.08, hspace=0.13)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)


def validate_normalization(frontiers, solve_grid) -> dict:
    max_welfare_error = 0.0
    comparisons = 0
    for model, (budgets, q) in frontiers.items():
        for gamma in GAMMAS:
            for value in (25.0, 250.0, 1000.0):
                for lam in (0.0001, 0.01, 0.03):
                    for beta in (1.0, 100.0, 1000.0):
                        legacy = solve_grid(
                            q, budgets, np.array([value]), np.array([lam]),
                            beta=beta, gamma=gamma, epsilon=1e-12,
                        )
                        normalized = solve_normalized(
                            q, budgets, np.array([lam / value]),
                            np.array([beta / value]), gamma,
                        ).iloc[0]
                        if legacy["ne_total"][0] != normalized["ne_total"]:
                            raise AssertionError("Nash allocation is not normalization invariant")
                        if legacy["opt_total"][0] != normalized["planner_total"]:
                            raise AssertionError("Planner allocation is not normalization invariant")
                        error = abs(
                            legacy["absolute_welfare_loss"][0] / value
                            - normalized["normalized_welfare_loss"]
                        )
                        max_welfare_error = max(max_welfare_error, float(error))
                        comparisons += 1
    return {
        "status": "passed",
        "comparisons": comparisons,
        "max_absolute_error_in_delta_W_over_V": max_welfare_error,
        "allocation_result": "NE and planner totals exactly match",
        "normalization": "kappa=lambda/V; rho=beta/V",
        "reporting_caveat": (
            "The legacy relative-loss omission rule |W_OPT|<=0.01 is not scale "
            "invariant. It is not used to define normalized phase regions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="robustness_extensions/server_outputs")
    parser.add_argument("--grid-points", type=int, default=181)
    args = parser.parse_args()
    output = Path(args.output_dir)
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from game_core import solve_grid

    frontiers = point_frontiers(load_stochastic(output))
    kappas = np.geomspace(*KAPPA_BOUNDS, args.grid_points)
    rhos = np.geomspace(*RHO_BOUNDS, args.grid_points)
    phase_frames: list[pd.DataFrame] = []
    boundary_rows: list[dict] = []
    for model, (budgets, q) in frontiers.items():
        for gamma in GAMMAS:
            phase = solve_normalized(q, budgets, kappas, rhos, gamma)
            phase.insert(0, "model_label", MODEL_LABELS[model])
            phase.insert(0, "model", model)
            phase_frames.append(phase)
            boundary_rows.extend(exact_transition_rows(model, q, budgets, gamma))
    phases = pd.concat(phase_frames, ignore_index=True)
    boundaries = pd.DataFrame(boundary_rows)
    validation = validate_normalization(frontiers, solve_grid)

    phases.to_csv(tables / "normalized_phase_grid.csv.gz", index=False, compression="gzip")
    boundaries.to_csv(tables / "normalized_transition_boundaries.csv", index=False)
    (tables / "normalization_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    models = list(MODEL_LABELS)
    plot_phase_panels(
        phases, models, [1.0], figures / "normalized_phase_gamma1.pdf", True
    )
    plot_phase_panels(
        phases, models, list(GAMMAS), figures / "normalized_phase_all_gamma.pdf", False
    )
    plot_phase_panels(phases, models, [1.0], figures / "normalized_phase_gamma1.png", True)
    plot_phase_panels(phases, models, list(GAMMAS), figures / "normalized_phase_all_gamma.png", False)
    print(json.dumps(validation, indent=2))
    print(f"Wrote normalized game outputs to {output}")


if __name__ == "__main__":
    main()
