"""Exact reduced-form game enumeration shared by offline analyses."""
from __future__ import annotations

import math

import numpy as np


def solve_grid(
    q: np.ndarray,
    budgets: np.ndarray,
    values: np.ndarray,
    lambdas: np.ndarray,
    beta: float = 1.0,
    gamma: float = 1.0,
    epsilon: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Solve every discrete game cell using the paper's deterministic ties."""
    grid_v, grid_lam = np.meshgrid(values, lambdas, indexing="ij")
    v = grid_v.ravel()
    lam = grid_lam.ravel()
    qg = np.power(np.maximum(q, 0.0), gamma)
    denom = qg[:, None] + qg[None, :]
    probability = np.divide(
        qg[:, None], denom, out=np.full_like(denom, 0.5), where=denom > 0
    )

    private = (
        v[:, None, None] * probability[None, :, :]
        - lam[:, None, None] * budgets[None, :, None]
    )
    best_response = np.argmax(private, axis=1)
    symmetric = best_response == np.arange(len(budgets))[None, :]
    equilibrium_count = symmetric.sum(axis=1)
    valid = equilibrium_count > 0
    ne_index = np.where(valid, np.argmax(symmetric, axis=1), -1)

    weighted_quality = beta * (
        probability * q[:, None] + (1.0 - probability) * q[None, :]
    )
    total_compute = budgets[:, None] + budgets[None, :]
    planner_objective = (
        weighted_quality.ravel()[None, :]
        - lam[:, None] * total_compute.ravel()[None, :]
    )
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
    porc = np.divide(
        ne_total, opt_total, out=np.full_like(ne_total, np.nan), where=opt_total > 0
    )
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
