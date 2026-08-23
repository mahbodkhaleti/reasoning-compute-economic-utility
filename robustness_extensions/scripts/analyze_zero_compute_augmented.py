#!/usr/bin/env python3
"""Offline zero-compute-augmented sensitivity analysis.

This script never loads or queries a language model.  It combines the stored
deterministic reasoning-mode c=0 observations (averaged over option-order
replications within scenario) with the stored five-seed stochastic positive-
budget observations.  The resulting C_H={0,128,384,768} frontier is a hybrid
sensitivity, not a protocol-matched stochastic four-budget experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from analyze_normalized_game import (
    GAMMAS,
    KAPPA_BOUNDS,
    MODEL_LABELS,
    PHASE_CMAP,
    RHO_BOUNDS,
    game_primitives,
    selected_equilibrium,
    solve_normalized,
)


HYBRID_BUDGETS = np.array([0, 128, 384, 768], dtype=int)
PRIMARY_BUDGETS = np.array([128, 384, 768], dtype=int)
ZERO_ORDER_SEEDS = [11, 23, 37]
STOCHASTIC_SEEDS = [1, 7, 21, 42, 99]
STOCHASTIC_ORDER_SEED = 314159
EXPECTED_SCENARIOS = 216


def _merge_numerically_identical(values: list[float]) -> list[float]:
    """Merge algebraically coincident thresholds split by floating roundoff."""
    merged: list[list[float]] = []
    for value in sorted(values):
        if merged and np.isclose(value, np.mean(merged[-1]), rtol=1e-11, atol=1e-15):
            merged[-1].append(value)
        else:
            merged.append([value])
    return [float(np.mean(group)) for group in merged]


def stable_transition_rows(model: str, q: np.ndarray, budgets: np.ndarray, gamma: float) -> list[dict]:
    """Exact discrete transitions with roundoff-equivalent boundaries merged."""
    probability, quality, total_compute = game_primitives(q, budgets, gamma)
    rows: list[dict] = []
    private_candidates: list[float] = []
    for opponent in range(len(budgets)):
        for a in range(len(budgets)):
            for b in range(a):
                dc = float(budgets[a] - budgets[b])
                value = float((probability[a, opponent] - probability[b, opponent]) / dc)
                if np.isfinite(value) and value > 0:
                    private_candidates.append(value)
    private = _merge_numerically_identical(private_candidates)

    def ne_state(kappa: float):
        index, count = selected_equilibrium(probability, budgets, np.array([kappa]))
        return int(budgets[index[0]]) if count[0] > 0 else None

    private_points = ([0.5 * private[0]] if private else [1.0])
    private_points += [np.sqrt(a * b) for a, b in zip(private[:-1], private[1:])]
    if private:
        private_points.append(2.0 * private[-1])
    private_states = [ne_state(point) for point in private_points]
    for boundary, before, after in zip(private, private_states[:-1], private_states[1:]):
        if before != after:
            rows.append(
                {
                    "model": model, "model_label": MODEL_LABELS[model], "gamma": gamma,
                    "boundary_type": "private_NE_kappa", "threshold": boundary,
                    "compute_before": before, "compute_after": after,
                    "inequality": "kappa < DeltaP/DeltaC favors the higher-payoff deviation",
                }
            )

    totals = np.unique(total_compute)
    best_quality = np.array([quality[total_compute == total].max() for total in totals])
    planner_candidates: list[float] = []
    for a in range(len(totals)):
        for b in range(a):
            value = float((best_quality[a] - best_quality[b]) / (totals[a] - totals[b]))
            if np.isfinite(value) and value > 0:
                planner_candidates.append(value)
    planner = _merge_numerically_identical(planner_candidates)

    def planner_state(ratio: float):
        index = int(np.argmax(quality.ravel() - ratio * total_compute.ravel()))
        return int(total_compute.ravel()[index])

    planner_points = ([0.5 * planner[0]] if planner else [1.0])
    planner_points += [np.sqrt(a * b) for a, b in zip(planner[:-1], planner[1:])]
    if planner:
        planner_points.append(2.0 * planner[-1])
    planner_states = [planner_state(point) for point in planner_points]
    for boundary, before, after in zip(planner, planner_states[:-1], planner_states[1:]):
        if before != after:
            rows.append(
                {
                    "model": model, "model_label": MODEL_LABELS[model], "gamma": gamma,
                    "boundary_type": "planner_kappa_over_rho", "threshold": boundary,
                    "compute_before": before, "compute_after": after,
                    "inequality": "kappa/rho < DeltaQ/DeltaC favors the higher-compute allocation",
                }
            )
    return rows


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(output: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = output / "raw"
    zero_rows: list[dict] = []
    positive_rows: list[dict] = []
    zero_files = sorted(raw.glob("fixed_*_reasoning_greedy.jsonl"))
    positive_files = sorted(raw.glob("fixed_*_reasoning_stochastic.jsonl"))
    for path in zero_files:
        zero_rows.extend(read_jsonl(path))
    for path in positive_files:
        positive_rows.extend(read_jsonl(path))
    if not zero_rows or not positive_rows:
        raise RuntimeError(f"Required saved outputs are missing under {raw}")

    zero = pd.DataFrame(zero_rows)
    positive = pd.DataFrame(positive_rows)
    for frame in (zero, positive):
        frame["scenario_id"] = frame["scenario_id"].astype(str).str.zfill(12)

    zero = zero.loc[zero["budget"] == 0].copy()
    positive = positive.loc[positive["budget"].isin(PRIMARY_BUDGETS)].copy()
    zero_key = ["model", "scenario_id", "budget", "repeat_seed"]
    positive_key = ["model", "scenario_id", "budget", "repeat_seed"]
    zero_duplicates = int(zero.duplicated(zero_key, keep=False).sum())
    positive_duplicates = int(positive.duplicated(positive_key, keep=False).sum())
    if zero_duplicates or positive_duplicates:
        raise ValueError(
            f"Duplicate keys detected: zero={zero_duplicates}, positive={positive_duplicates}"
        )

    provenance = {
        "zero_files": [str(path) for path in zero_files],
        "positive_files": [str(path) for path in positive_files],
        "zero_source": "reasoning-mode deterministic greedy c=0, averaged over option-order replications within scenario",
        "positive_source": "five-seed stochastic reasoning observations at c in {128,384,768}",
        "protocol_warning": (
            "The zero and positive-budget observations use different presentation/decoding protocols; "
            "this is a hybrid sensitivity, not a fully stochastic four-budget frontier."
        ),
        "duplicate_zero_keys": zero_duplicates,
        "duplicate_positive_keys": positive_duplicates,
    }
    return zero, positive, provenance


def build_panels(
    zero: pd.DataFrame, positive: pd.DataFrame
) -> tuple[dict[str, dict], list[dict]]:
    expected_models = set(MODEL_LABELS)
    if set(zero["model"].unique()) != expected_models:
        raise ValueError("Deterministic zero-budget data do not contain exactly all four models")
    if set(positive["model"].unique()) != expected_models:
        raise ValueError("Stochastic data do not contain exactly all four models")

    panels: dict[str, dict] = {}
    frontier_rows: list[dict] = []
    for model in MODEL_LABELS:
        z = zero.loc[zero["model"] == model]
        p = positive.loc[positive["model"] == model]
        scenarios = sorted(z["scenario_id"].unique())
        if len(scenarios) != EXPECTED_SCENARIOS:
            raise ValueError(f"{model}: expected 216 c=0 scenarios, found {len(scenarios)}")
        if scenarios != sorted(p["scenario_id"].unique()):
            raise ValueError(f"{model}: deterministic and stochastic scenario IDs differ")
        if sorted(z["repeat_seed"].unique()) != ZERO_ORDER_SEEDS:
            raise ValueError(f"{model}: unexpected deterministic option-order seeds")
        if sorted(p["repeat_seed"].unique()) != STOCHASTIC_SEEDS:
            raise ValueError(f"{model}: unexpected stochastic generation repeat seeds")
        if sorted(p["budget"].unique()) != PRIMARY_BUDGETS.tolist():
            raise ValueError(f"{model}: unexpected stochastic budgets")
        if set(p["option_order_seed"].unique()) != {STOCHASTIC_ORDER_SEED}:
            raise ValueError(f"{model}: stochastic option order is not fixed at 314159")

        zero_panel = (
            z.pivot(index="scenario_id", columns="repeat_seed", values="test_utility_ratio")
            .reindex(index=scenarios, columns=ZERO_ORDER_SEEDS)
        )
        positive_panel = (
            p.pivot(index="scenario_id", columns=["budget", "repeat_seed"], values="test_utility_ratio")
            .reindex(
                index=scenarios,
                columns=pd.MultiIndex.from_product([PRIMARY_BUDGETS, STOCHASTIC_SEEDS]),
            )
        )
        if zero_panel.isna().any().any() or positive_panel.isna().any().any():
            raise ValueError(f"{model}: incomplete hybrid input panel")
        zero_array = zero_panel.to_numpy(float)
        positive_array = positive_panel.to_numpy(float).reshape(
            EXPECTED_SCENARIOS, len(PRIMARY_BUDGETS), len(STOCHASTIC_SEEDS)
        )
        if not np.isfinite(zero_array).all() or not np.isfinite(positive_array).all():
            raise ValueError(f"{model}: non-finite q observation")

        zero_scenario_mean = zero_array.mean(axis=1)
        point_primary = positive_array.mean(axis=(0, 2))
        point_hybrid = np.r_[zero_scenario_mean.mean(), point_primary]
        panels[model] = {
            "scenarios": scenarios,
            "zero_by_scenario_order": zero_array,
            "zero_by_scenario": zero_scenario_mean,
            "positive": positive_array,
            "point_primary": point_primary,
            "point_hybrid": point_hybrid,
        }
        for budget, quality in zip(HYBRID_BUDGETS, point_hybrid):
            frontier_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "budget": budget,
                    "q": quality,
                    "source_protocol": (
                        "deterministic_reasoning_greedy_order_average"
                        if budget == 0
                        else "stochastic_reasoning_five_seed_fixed_order"
                    ),
                    "n_scenarios": EXPECTED_SCENARIOS,
                    "n_replicates": len(ZERO_ORDER_SEEDS) if budget == 0 else len(STOCHASTIC_SEEDS),
                }
            )
    return panels, frontier_rows


def transitions(model: str, q: np.ndarray, budgets: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for gamma in GAMMAS:
        rows.extend(stable_transition_rows(model, q, budgets, gamma))
    return rows


def phase_set_exact(q: np.ndarray, budgets: np.ndarray, gamma: float) -> set[str]:
    """Find open phase regions by exact threshold-cell decomposition.

    NE changes only at vertical kappa thresholds and the planner changes only
    at diagonals kappa/rho=r.  Sampling one interior point in every cell is
    therefore exhaustive over the declared positive (kappa,rho) domain.
    """
    rows = stable_transition_rows("qwen3_4b", q, budgets, gamma)
    ne_t = [r["threshold"] for r in rows if r["boundary_type"] == "private_NE_kappa"]
    pl_t = [r["threshold"] for r in rows if r["boundary_type"] == "planner_kappa_over_rho"]
    k_breaks = {KAPPA_BOUNDS[0], KAPPA_BOUNDS[1]}
    k_breaks.update(t for t in ne_t if KAPPA_BOUNDS[0] < t < KAPPA_BOUNDS[1])
    for threshold in pl_t:
        k_breaks.update(
            x for x in (threshold * RHO_BOUNDS[0], threshold * RHO_BOUNDS[1])
            if KAPPA_BOUNDS[0] < x < KAPPA_BOUNDS[1]
        )
    k_sorted = sorted(k_breaks)
    phases: set[str] = set()
    for lo, hi in zip(k_sorted[:-1], k_sorted[1:]):
        kappa = float(np.sqrt(lo * hi))
        rho_breaks = {RHO_BOUNDS[0], RHO_BOUNDS[1]}
        rho_breaks.update(
            kappa / threshold for threshold in pl_t
            if RHO_BOUNDS[0] < kappa / threshold < RHO_BOUNDS[1]
        )
        rho_sorted = sorted(rho_breaks)
        for rlo, rhi in zip(rho_sorted[:-1], rho_sorted[1:]):
            rho = float(np.sqrt(rlo * rhi))
            phases.add(
                str(solve_normalized(q, budgets, np.array([kappa]), np.array([rho]), gamma).iloc[0]["phase"])
            )
    return phases


def transition_text(frame: pd.DataFrame, boundary_type: str) -> str:
    part = frame.loc[frame["boundary_type"] == boundary_type].sort_values("threshold")
    if part.empty:
        return "none"
    def compute_text(value: float) -> str:
        return "no symmetric pure NE" if pd.isna(value) else str(int(value))

    return "; ".join(
        f"{compute_text(row.compute_before)}->{compute_text(row.compute_after)} @ {row.threshold:.8g}"
        for row in part.itertuples()
    )


def enumerate_phases(
    panels: dict[str, dict], kappas: np.ndarray, rhos: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phase_frames: list[pd.DataFrame] = []
    transition_rows: list[dict] = []
    comparison_rows: list[dict] = []
    for model, panel in panels.items():
        menu_data = {
            "primary_stochastic": (PRIMARY_BUDGETS, panel["point_primary"]),
            "hybrid_zero_compute": (HYBRID_BUDGETS, panel["point_hybrid"]),
        }
        menu_phases: dict[tuple[str, float], pd.DataFrame] = {}
        for menu, (budgets, q) in menu_data.items():
            for gamma in GAMMAS:
                phase = solve_normalized(q, budgets, kappas, rhos, gamma)
                phase.insert(0, "menu", menu)
                phase.insert(0, "model_label", MODEL_LABELS[model])
                phase.insert(0, "model", model)
                phase_frames.append(phase)
                menu_phases[(menu, gamma)] = phase
                for row in stable_transition_rows(model, q, budgets, gamma):
                    transition_rows.append({"menu": menu, **row})

        all_transitions = pd.DataFrame(transition_rows)
        for gamma in GAMMAS:
            primary = menu_phases[("primary_stochastic", gamma)]
            hybrid = menu_phases[("hybrid_zero_compute", gamma)]
            primary_t = all_transitions.loc[
                (all_transitions["model"] == model)
                & (all_transitions["gamma"] == gamma)
                & (all_transitions["menu"] == "primary_stochastic")
            ]
            hybrid_t = all_transitions.loc[
                (all_transitions["model"] == model)
                & (all_transitions["gamma"] == gamma)
                & (all_transitions["menu"] == "hybrid_zero_compute")
            ]
            pset = phase_set_exact(panel["point_primary"], PRIMARY_BUDGETS, gamma)
            hset = phase_set_exact(panel["point_hybrid"], HYBRID_BUDGETS, gamma)
            core_phases = {"Overcompute", "Aligned", "Undercompute"}
            comparison_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "gamma": gamma,
                    "primary_nash_transitions": transition_text(primary_t, "private_NE_kappa"),
                    "zero_augmented_nash_transitions": transition_text(hybrid_t, "private_NE_kappa"),
                    "primary_planner_transitions": transition_text(primary_t, "planner_kappa_over_rho"),
                    "zero_augmented_planner_transitions": transition_text(hybrid_t, "planner_kappa_over_rho"),
                    "primary_open_region_phases": ", ".join(sorted(pset)),
                    "zero_augmented_open_region_phases": ", ".join(sorted(hset)),
                    "qualitative_phase_structure_preserved": (
                        pset.intersection(core_phases) == hset.intersection(core_phases)
                        and core_phases.issubset(pset)
                        and core_phases.issubset(hset)
                    ),
                    "primary_no_symmetric_pure_ne_region": "No symmetric pure NE" in pset,
                    "zero_augmented_no_symmetric_pure_ne_region": "No symmetric pure NE" in hset,
                    "grid_dependent_disagreement_fraction_diagnostic": float(
                        np.mean(primary["phase_code"].to_numpy() != hybrid["phase_code"].to_numpy())
                    ),
                }
            )
    return (
        pd.concat(phase_frames, ignore_index=True),
        pd.DataFrame(transition_rows),
        pd.DataFrame(comparison_rows),
    )


def bootstrap(
    panels: dict[str, dict], samples: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hybrid hierarchical bootstrap using only saved observations.

    Scenarios are sampled jointly across all four budgets.  At c=0 the
    deterministic option-order average for that scenario is used.  One stored
    stochastic repeat seed is sampled per selected scenario and used jointly
    across all three positive budgets, matching the primary bootstrap logic.
    """
    rng = np.random.default_rng(seed)
    frontier_rows: list[dict] = []
    threshold_rows: list[dict] = []
    phase_rows: list[dict] = []
    for model, panel in panels.items():
        zero = panel["zero_by_scenario"]
        positive = panel["positive"]
        for draw in range(samples):
            scenario_index = rng.integers(0, EXPECTED_SCENARIOS, size=EXPECTED_SCENARIOS)
            seed_index = rng.integers(0, len(STOCHASTIC_SEEDS), size=EXPECTED_SCENARIOS)
            q = np.r_[
                zero[scenario_index].mean(),
                positive[scenario_index, :, seed_index].mean(axis=0),
            ]
            for budget, quality in zip(HYBRID_BUDGETS, q):
                frontier_rows.append(
                    {"model": model, "bootstrap": draw, "budget": budget, "q": quality}
                )
            for gamma in GAMMAS:
                draw_transitions = stable_transition_rows(model, q, HYBRID_BUDGETS, gamma)
                for row in draw_transitions:
                    threshold_rows.append({"bootstrap": draw, **row})
                phases = phase_set_exact(q, HYBRID_BUDGETS, gamma)
                for phase in (
                    "Overcompute", "Aligned", "Undercompute", "No symmetric pure NE"
                ):
                    phase_rows.append(
                        {
                            "model": model,
                            "bootstrap": draw,
                            "gamma": gamma,
                            "phase": phase,
                            "open_region_present": phase in phases,
                        }
                    )
        print(f"Completed offline hybrid bootstrap for {model}")

    frontier = pd.DataFrame(frontier_rows)
    thresholds = pd.DataFrame(threshold_rows)
    phase_presence = pd.DataFrame(phase_rows)
    summary_rows: list[dict] = []
    for (model, budget), group in frontier.groupby(["model", "budget"]):
        values = group["q"].to_numpy(float)
        summary_rows.append(
            {
                "summary_type": "frontier_q",
                "model": model,
                "gamma": np.nan,
                "boundary_type": "q",
                "compute_before": budget,
                "compute_after": budget,
                "phase": "",
                "point_or_fraction": panels[model]["point_hybrid"][np.where(HYBRID_BUDGETS == budget)[0][0]],
                "bootstrap_mean": values.mean(),
                "ci_low": np.quantile(values, 0.025),
                "ci_high": np.quantile(values, 0.975),
                "n_bootstrap": samples,
            }
        )
    for keys, group in thresholds.groupby(
        ["model", "gamma", "boundary_type", "compute_before", "compute_after"], dropna=False
    ):
        model, gamma, kind, before, after = keys
        values = group["threshold"].to_numpy(float)
        summary_rows.append(
            {
                "summary_type": "transition_threshold_conditional_on_presence",
                "model": model,
                "gamma": gamma,
                "boundary_type": kind,
                "compute_before": before,
                "compute_after": after,
                "phase": "",
                "point_or_fraction": len(group) / samples,
                "bootstrap_mean": values.mean(),
                "ci_low": np.quantile(values, 0.025),
                "ci_high": np.quantile(values, 0.975),
                "n_bootstrap": samples,
            }
        )
    for (model, gamma, phase), group in phase_presence.groupby(["model", "gamma", "phase"]):
        fraction = float(group["open_region_present"].mean())
        summary_rows.append(
            {
                "summary_type": "exact_open_region_presence_fraction",
                "model": model,
                "gamma": gamma,
                "boundary_type": "",
                "compute_before": np.nan,
                "compute_after": np.nan,
                "phase": phase,
                "point_or_fraction": fraction,
                "bootstrap_mean": fraction,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_bootstrap": samples,
            }
        )
    return frontier, thresholds, pd.DataFrame(summary_rows)


def validate_allocations(phases: pd.DataFrame, panels: dict[str, dict]) -> dict:
    checks = 0
    max_ne_gap = 0.0
    max_planner_gap = 0.0
    hybrid = phases.loc[phases["menu"] == "hybrid_zero_compute"]
    allowed_totals = set((HYBRID_BUDGETS[:, None] + HYBRID_BUDGETS[None, :]).ravel())
    if not set(hybrid["ne_total"].dropna().astype(int)).issubset({0, 256, 768, 1536}):
        raise AssertionError("Hybrid NE allocation is outside the symmetric action menu")
    if not set(hybrid["planner_total"].astype(int)).issubset(allowed_totals):
        raise AssertionError("Hybrid planner allocation is outside C_H x C_H")

    # Exhaustively check all rendered grid allocations against direct payoff and
    # welfare enumeration.  This also checks the stated tie rules in practice.
    for model, panel in panels.items():
        q = panel["point_hybrid"]
        for gamma in GAMMAS:
            probability, quality, total = game_primitives(q, HYBRID_BUDGETS, gamma)
            part = hybrid.loc[(hybrid["model"] == model) & (hybrid["gamma"] == gamma)]
            for row in part.itertuples():
                if not np.isfinite(row.ne_total):
                    # The existing game deliberately reports absence of a
                    # symmetric pure equilibrium rather than substituting a
                    # mixed or asymmetric solution.
                    continue
                kappa, rho = float(row.kappa), float(row.rho)
                ne_action = int(row.ne_total // 2)
                opponent = int(np.where(HYBRID_BUDGETS == ne_action)[0][0])
                private = probability[:, opponent] - kappa * HYBRID_BUDGETS
                chosen = int(np.argmax(private))
                max_ne_gap = max(max_ne_gap, float(private.max() - private[opponent]))
                if chosen != opponent:
                    raise AssertionError("Reported Nash action is not the lowest-budget mutual best response")
                objective = rho * quality.ravel() - kappa * total.ravel()
                planner_index = int(np.argmax(objective))
                max_planner_gap = max(
                    max_planner_gap,
                    float(objective.max() - row.normalized_welfare_opt),
                )
                if int(total.ravel()[planner_index]) != int(row.planner_total):
                    raise AssertionError("Reported planner allocation is not an exhaustive welfare maximizer")
                checks += 1
    return {
        "allocation_checks": checks,
        "max_nash_optimality_gap": max_ne_gap,
        "max_planner_optimality_gap": max_planner_gap,
    }


def validate_thresholds(transitions_frame: pd.DataFrame, panels: dict[str, dict]) -> dict:
    checked = 0
    for row in transitions_frame.loc[transitions_frame["menu"] == "hybrid_zero_compute"].itertuples():
        q = panels[row.model]["point_hybrid"]
        threshold = float(row.threshold)
        lo, hi = threshold * (1 - 1e-8), threshold * (1 + 1e-8)
        if row.boundary_type == "private_NE_kappa":
            probability, _, _ = game_primitives(q, HYBRID_BUDGETS, row.gamma)
            i_lo, n_lo = selected_equilibrium(probability, HYBRID_BUDGETS, np.array([lo]))
            i_hi, n_hi = selected_equilibrium(probability, HYBRID_BUDGETS, np.array([hi]))
            before = int(HYBRID_BUDGETS[i_lo[0]]) if n_lo[0] else None
            after = int(HYBRID_BUDGETS[i_hi[0]]) if n_hi[0] else None
        else:
            _, quality, total = game_primitives(q, HYBRID_BUDGETS, row.gamma)
            before = int(total.ravel()[np.argmax(quality.ravel() - lo * total.ravel())])
            after = int(total.ravel()[np.argmax(quality.ravel() - hi * total.ravel())])
        expected_before = None if pd.isna(row.compute_before) else int(row.compute_before)
        expected_after = None if pd.isna(row.compute_after) else int(row.compute_after)
        if before != expected_before or after != expected_after:
            raise AssertionError(
                f"Threshold-side mismatch for {row.model}, gamma={row.gamma}, {row.boundary_type}"
            )
        checked += 1
    return {"thresholds_checked_on_both_sides": checked, "status": "passed"}


def validate_rendering(
    phases: pd.DataFrame, transitions_frame: pd.DataFrame, panels: dict[str, dict]
) -> dict:
    """Compare stored pixels with scalar enumeration away from boundaries."""
    checked = 0
    for (model, menu, gamma), part in phases.groupby(["model", "menu", "gamma"]):
        budgets = PRIMARY_BUDGETS if menu == "primary_stochastic" else HYBRID_BUDGETS
        q = panels[model]["point_primary"] if menu == "primary_stochastic" else panels[model]["point_hybrid"]
        boundary = transitions_frame.loc[
            (transitions_frame["model"] == model)
            & (transitions_frame["menu"] == menu)
            & (transitions_frame["gamma"] == gamma)
        ]
        ne_t = boundary.loc[boundary["boundary_type"] == "private_NE_kappa", "threshold"].to_numpy(float)
        pl_t = boundary.loc[boundary["boundary_type"] == "planner_kappa_over_rho", "threshold"].to_numpy(float)
        sample = part.iloc[::97]
        for row in sample.itertuples():
            if any(np.isclose(row.kappa, ne_t, rtol=1e-8, atol=0.0)):
                continue
            if any(np.isclose(row.kappa / row.rho, pl_t, rtol=1e-8, atol=0.0)):
                continue
            scalar = solve_normalized(
                q, budgets, np.array([row.kappa]), np.array([row.rho]), gamma
            ).iloc[0]
            same_phase = (
                (pd.isna(row.phase_code) and pd.isna(scalar["phase_code"]))
                or row.phase_code == scalar["phase_code"]
            )
            if not same_phase or row.ne_total != scalar["ne_total"] or row.planner_total != scalar["planner_total"]:
                raise AssertionError("Rendered phase cell disagrees with scalar exact enumeration")
            checked += 1
    return {
        "phase_pixels_checked_away_from_boundaries": checked,
        "phase_rendering_agrees_with_exact_enumeration": True,
    }


def plot_comparison(phases: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12.2, 5.35), sharex=True, sharey=True)
    menus = ["primary_stochastic", "hybrid_zero_compute"]
    row_labels = [r"Primary $\{128,384,768\}$", r"Hybrid $\{0,128,384,768\}$"]
    for row_index, (menu, row_label) in enumerate(zip(menus, row_labels)):
        for col_index, model in enumerate(MODEL_LABELS):
            ax = axes[row_index, col_index]
            part = phases.loc[
                (phases["menu"] == menu)
                & (phases["model"] == model)
                & (phases["gamma"] == 1.0)
            ]
            pivot = part.pivot(index="rho", columns="kappa", values="phase_code")
            ax.pcolormesh(
                pivot.columns.to_numpy(float), pivot.index.to_numpy(float), pivot.to_numpy(float),
                cmap=PHASE_CMAP, vmin=-1, vmax=1, shading="nearest", rasterized=True,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(which="major", color="white", linewidth=0.35, alpha=0.45)
            if row_index == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=9)
            if col_index == 0:
                ax.text(-0.28, 0.5, row_label, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=8.5)
    fig.supxlabel(r"Relative compute cost $\kappa=\lambda/V$")
    fig.supylabel(r"Relative social quality value $\rho=\beta/V$")
    legend = [
        Patch(facecolor=PHASE_CMAP(0.0), label="Undercompute"),
        Patch(facecolor=PHASE_CMAP(0.5), label="Aligned"),
        Patch(facecolor=PHASE_CMAP(1.0), label="Overcompute"),
    ]
    fig.legend(
        handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=3, frameon=False, fontsize=8,
    )
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.11, top=0.91, wspace=0.08, hspace=0.13)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_hybrid_all_gamma(phases: pd.DataFrame, output: Path) -> None:
    hybrid = phases.loc[phases["menu"] == "hybrid_zero_compute"]
    fig, axes = plt.subplots(4, 4, figsize=(12.3, 9.4), sharex=True, sharey=True)
    for row_index, model in enumerate(MODEL_LABELS):
        for col_index, gamma in enumerate(GAMMAS):
            ax = axes[row_index, col_index]
            part = hybrid.loc[(hybrid["model"] == model) & (hybrid["gamma"] == gamma)]
            pivot = part.pivot(index="rho", columns="kappa", values="phase_code")
            ax.pcolormesh(
                pivot.columns.to_numpy(float), pivot.index.to_numpy(float), pivot.to_numpy(float),
                cmap=PHASE_CMAP, vmin=-1, vmax=1, shading="nearest", rasterized=True,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(which="major", color="white", linewidth=0.35, alpha=0.45)
            if row_index == 0:
                ax.set_title(rf"$\gamma={gamma:g}$", fontsize=9)
            if col_index == 0:
                ax.text(-0.27, 0.5, MODEL_LABELS[model], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=9)
    fig.supxlabel(r"Relative compute cost $\kappa=\lambda/V$")
    fig.supylabel(r"Relative social quality value $\rho=\beta/V$")
    fig.legend(
        handles=[Patch(facecolor=PHASE_CMAP(0.0), label="Undercompute"),
                 Patch(facecolor=PHASE_CMAP(0.5), label="Aligned"),
                 Patch(facecolor=PHASE_CMAP(1.0), label="Overcompute")],
        loc="upper center", ncol=3, frameon=False, fontsize=8,
    )
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.08, top=0.92, wspace=0.08, hspace=0.12)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="robustness_extensions/server_outputs")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--grid-points", type=int, default=181)
    args = parser.parse_args()
    output = Path(args.output_dir)
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    primary_paths = [
        tables / "normalized_phase_grid.csv.gz",
        tables / "normalized_transition_boundaries.csv",
        tables / "stochastic_game_uncertainty_summary.csv",
        tables / "stochastic_frontier_bootstrap.csv",
        figures / "normalized_phase_gamma1.pdf",
        figures / "normalized_phase_all_gamma.pdf",
    ]
    hashes_before = {str(path): sha256(path) for path in primary_paths if path.exists()}
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("Inference-related modules were already imported; aborting offline analysis")

    zero, positive, provenance = load_inputs(output)
    panels, frontier_rows = build_panels(zero, positive)
    kappas = np.geomspace(*KAPPA_BOUNDS, args.grid_points)
    rhos = np.geomspace(*RHO_BOUNDS, args.grid_points)
    phases, transition_frame, comparison = enumerate_phases(panels, kappas, rhos)
    frontier_boot, threshold_boot, bootstrap_summary = bootstrap(
        panels, args.samples, args.seed
    )

    frontier_path = tables / "hybrid_zero_compute_frontier.csv"
    transitions_path = tables / "hybrid_zero_compute_transition_thresholds.csv"
    comparison_path = tables / "hybrid_zero_compute_transition_comparison.csv"
    phase_path = tables / "hybrid_zero_compute_phase_grid.csv.gz"
    frontier_boot_path = tables / "hybrid_zero_compute_bootstrap_frontiers.csv.gz"
    threshold_boot_path = tables / "hybrid_zero_compute_bootstrap_thresholds.csv.gz"
    summary_path = tables / "hybrid_zero_compute_bootstrap_summary.csv"
    validation_path = tables / "hybrid_zero_compute_validation.json"

    pd.DataFrame(frontier_rows).to_csv(frontier_path, index=False)
    transition_frame.to_csv(transitions_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    phases.to_csv(phase_path, index=False, compression="gzip")
    frontier_boot.to_csv(frontier_boot_path, index=False, compression="gzip")
    threshold_boot.to_csv(threshold_boot_path, index=False, compression="gzip")
    bootstrap_summary.to_csv(summary_path, index=False)
    plot_comparison(phases, figures / "hybrid_zero_compute_phase_comparison_gamma1.pdf")
    plot_hybrid_all_gamma(phases, figures / "hybrid_zero_compute_phase_all_gamma.pdf")

    allocation_validation = validate_allocations(phases, panels)
    threshold_validation = validate_thresholds(transition_frame, panels)
    rendering_validation = validate_rendering(phases, transition_frame, panels)
    hashes_after = {str(path): sha256(path) for path in primary_paths if path.exists()}
    primary_unchanged = hashes_before == hashes_after
    if not primary_unchanged:
        raise AssertionError("An existing primary output changed")
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise AssertionError("An inference-related module was imported")
    validation = {
        "status": "passed",
        "new_inference_executed": False,
        "inference_modules_imported": False,
        "models": list(MODEL_LABELS),
        "all_four_models_present": True,
        "scenarios_per_model": EXPECTED_SCENARIOS,
        "all_216_zero_scenarios_present": True,
        "zero_option_order_seeds": ZERO_ORDER_SEEDS,
        "positive_budgets": PRIMARY_BUDGETS.tolist(),
        "stochastic_repeat_seeds": STOCHASTIC_SEEDS,
        "stochastic_option_order_seed": STOCHASTIC_ORDER_SEED,
        "all_required_stochastic_cells_present": True,
        "duplicate_key_check": "passed",
        "finite_q_check": "passed",
        "hybrid_menu": HYBRID_BUDGETS.tolist(),
        "primary_outputs_hash_unchanged": primary_unchanged,
        "primary_output_sha256": hashes_after,
        "phase_rendering": "generated by the same exact enumerator used for validation",
        "protocol_warning": provenance["protocol_warning"],
        "bootstrap_estimand": (
            "scenario resampling jointly over all four budgets; deterministic c=0 order-average; "
            "one stored stochastic repeat seed sampled per selected scenario jointly over positive budgets"
        ),
        **allocation_validation,
        **threshold_validation,
        **rendering_validation,
        "provenance": provenance,
    }
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))
    print(f"Wrote zero-compute-augmented outputs to {output}")


if __name__ == "__main__":
    main()
