#!/usr/bin/env python3
"""Offline robustness analyses for the EconML workshop submission.

The script uses only saved constrained-readout outputs.  It does not call a
model.  It produces (i) beta and gamma comparative statics for the primary
stochastic frontier, (ii) a six-budget deterministic menu sensitivity, and
(iii) an auction-level bootstrap of the held-out closing-price CDF that
re-scores fixed stored choices.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "deepseek_r1_llama_8b": "DeepSeek-8B",
    "phi4_mini_reasoning": "Phi-mini",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
}
BETA_VALUES = (1.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0)
GAMMA_VALUES = (0.5, 1.0, 2.0, 4.0)
BUDGETS_DETERMINISTIC = (0, 64, 128, 256, 512, 768)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_saved_outputs(output: Path, pattern: str) -> pd.DataFrame:
    records: list[dict] = []
    for path in sorted((output / "raw").glob(pattern)):
        records.extend(read_jsonl(path))
    if not records:
        raise RuntimeError(f"No records matched {pattern!r} under {output / 'raw'}")
    frame = pd.DataFrame(records)
    frame["scenario_id"] = frame["scenario_id"].astype(str).str.zfill(12)
    frame = frame.drop_duplicates(
        ["model", "scenario_id", "budget", "repeat_seed"], keep="last"
    )
    return frame


def add_compute_metrics(solution: dict[str, np.ndarray], metrics: dict[str, float]) -> dict[str, float]:
    valid = solution["valid_equilibrium"]
    return {
        **metrics,
        "mean_equilibrium_total_compute": float(np.nanmean(solution["ne_total"][valid])),
        "mean_planner_total_compute": float(np.nanmean(solution["opt_total"][valid])),
        "median_equilibrium_total_compute": float(np.nanmedian(solution["ne_total"][valid])),
        "median_planner_total_compute": float(np.nanmedian(solution["opt_total"][valid])),
    }


def point_frontiers(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for model, group in frame.groupby("model"):
        means = group.groupby("budget")["test_utility_ratio"].mean().sort_index()
        result[model] = (means.index.to_numpy(int), means.to_numpy(float))
    return result


def strategic_sensitivities(
    stochastic: pd.DataFrame,
    solve_grid,
    aggregate_game,
    values: np.ndarray,
    lambdas: np.ndarray,
    epsilon: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    beta_rows, gamma_rows = [], []
    for model, (budgets, q) in point_frontiers(stochastic).items():
        for beta in BETA_VALUES:
            solved = solve_grid(
                q, budgets, values, lambdas, beta=beta, gamma=1.0, epsilon=epsilon
            )
            beta_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "beta": beta,
                    "gamma": 1.0,
                    **add_compute_metrics(solved, aggregate_game(solved)),
                }
            )
        for gamma in GAMMA_VALUES:
            solved = solve_grid(
                q, budgets, values, lambdas, beta=1.0, gamma=gamma, epsilon=epsilon
            )
            gamma_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "beta": 1.0,
                    "gamma": gamma,
                    **add_compute_metrics(solved, aggregate_game(solved)),
                }
            )
    return pd.DataFrame(beta_rows), pd.DataFrame(gamma_rows)


def deterministic_menu_sensitivity(
    greedy: pd.DataFrame,
    stochastic: pd.DataFrame,
    solve_grid,
    aggregate_game,
    values: np.ndarray,
    lambdas: np.ndarray,
    epsilon: float,
    samples: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, draw_rows = [], []
    primary = point_frontiers(stochastic)
    for model, group in greedy.groupby("model"):
        group = group.loc[group["budget"].isin(BUDGETS_DETERMINISTIC)].copy()
        panel = group.pivot_table(
            index="scenario_id", columns="budget", values="test_utility_ratio", aggfunc="mean"
        ).reindex(columns=BUDGETS_DETERMINISTIC)
        if panel.isna().any().any():
            raise ValueError(f"{model}: incomplete deterministic six-budget panel")
        budgets = panel.columns.to_numpy(int)
        q_point = panel.mean(axis=0).to_numpy(float)
        six_solution = solve_grid(
            q_point, budgets, values, lambdas, beta=1.0, gamma=1.0, epsilon=epsilon
        )
        six_metrics = add_compute_metrics(six_solution, aggregate_game(six_solution))
        boot = []
        array = panel.to_numpy(float)
        for draw in range(samples):
            indices = rng.integers(0, len(array), size=len(array))
            q = array[indices].mean(axis=0)
            solved = solve_grid(
                q, budgets, values, lambdas, beta=1.0, gamma=1.0, epsilon=epsilon
            )
            metrics = add_compute_metrics(solved, aggregate_game(solved))
            boot.append(metrics)
            draw_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "bootstrap": draw,
                    **metrics,
                }
            )
        boot_frame = pd.DataFrame(boot)
        row = {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "analysis": "deterministic_six_budget",
            "budgets": ",".join(map(str, budgets)),
        }
        for name, value in six_metrics.items():
            row[f"{name}_point"] = value
            row[f"{name}_ci_low"] = boot_frame[name].quantile(0.025)
            row[f"{name}_ci_high"] = boot_frame[name].quantile(0.975)
        rows.append(row)

        primary_budgets, primary_q = primary[model]
        primary_solution = solve_grid(
            primary_q,
            primary_budgets,
            values,
            lambdas,
            beta=1.0,
            gamma=1.0,
            epsilon=epsilon,
        )
        primary_metrics = add_compute_metrics(primary_solution, aggregate_game(primary_solution))
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "analysis": "stochastic_three_budget_primary",
                "budgets": ",".join(map(str, primary_budgets)),
                **{f"{name}_point": value for name, value in primary_metrics.items()},
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(draw_rows)


def parse_candidates(text: str) -> tuple[np.ndarray, np.ndarray]:
    candidates = json.loads(text)
    return (
        np.array([float(row["bid"]) for row in candidates]),
        np.array([float(row.get("test_u", np.nan)) for row in candidates]),
    )


def auction_cdf_bootstrap(
    greedy: pd.DataFrame,
    scenarios: pd.DataFrame,
    auctions: pd.DataFrame,
    samples: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rescore fixed choices under auction-resampled held-out CDFs.

    This targets held-out CDF estimation only.  It does not change the displayed
    construction CDF, prompts, candidate sets, or model choices.
    """
    scenario_info = {}
    for row in scenarios.to_dict("records"):
        bids, _ = parse_candidates(row["candidates_json"])
        scenario_info[str(row["scenario_id"]).zfill(12)] = {
            "category": row["category"],
            "value": float(row["value"]),
            "bids": bids,
        }
    test_prices = {
        category: group["final_price"].to_numpy(float)
        for category, group in auctions.loc[auctions["split"] == "test"].groupby("item")
    }
    categories = sorted(test_prices)
    resampled_prices = {
        category: prices[rng.integers(0, len(prices), size=(samples, len(prices)))]
        for category, prices in test_prices.items()
    }

    # Average the three option-order choices within each scenario and budget.
    chosen = (
        greedy.loc[greedy["budget"].isin(BUDGETS_DETERMINISTIC)]
        .groupby(["model", "scenario_id", "budget"])["chosen_bid"]
        .apply(list)
        .reset_index()
    )
    accum: dict[tuple[str, int], list[np.ndarray]] = {}
    for row in chosen.to_dict("records"):
        info = scenario_info[row["scenario_id"]]
        prices = resampled_prices[info["category"]]
        bids = info["bids"]
        value = info["value"]
        probabilities = (prices[:, :, None] <= bids[None, None, :]).mean(axis=1)
        utilities = probabilities * (value - bids)[None, :]
        optimum = utilities.max(axis=1)
        ratios = []
        for selected_bid in row["chosen_bid"]:
            index = int(np.argmin(np.abs(bids - float(selected_bid))))
            ratios.append(
                np.divide(
                    utilities[:, index],
                    optimum,
                    out=np.zeros(samples),
                    where=optimum > 0,
                )
            )
        accum.setdefault((row["model"], int(row["budget"])), []).append(
            np.mean(ratios, axis=0)
        )

    q_draws = {
        key: np.mean(np.stack(values_for_scenarios), axis=0)
        for key, values_for_scenarios in accum.items()
    }
    draw_rows, summary_rows = [], []
    for model in sorted(greedy["model"].unique()):
        for draw in range(samples):
            draw_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "bootstrap": draw,
                    **{
                        f"q_{budget}": q_draws[(model, budget)][draw]
                        for budget in BUDGETS_DETERMINISTIC
                    },
                    "endpoint_difference": q_draws[(model, 768)][draw]
                    - q_draws[(model, 0)][draw],
                }
            )
        difference = q_draws[(model, 768)] - q_draws[(model, 0)]
        summary_rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "bootstrap_unit": "held-out auction within product category",
                "n_bootstrap": samples,
                "q_0_bootstrap_mean": q_draws[(model, 0)].mean(),
                "q_0_ci_low": np.quantile(q_draws[(model, 0)], 0.025),
                "q_0_ci_high": np.quantile(q_draws[(model, 0)], 0.975),
                "q_768_bootstrap_mean": q_draws[(model, 768)].mean(),
                "q_768_ci_low": np.quantile(q_draws[(model, 768)], 0.025),
                "q_768_ci_high": np.quantile(q_draws[(model, 768)], 0.975),
                "endpoint_difference_bootstrap_mean": difference.mean(),
                "endpoint_difference_ci_low": np.quantile(difference, 0.025),
                "endpoint_difference_ci_high": np.quantile(difference, 0.975),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(draw_rows)


def plot_strategic_sensitivity(beta: pd.DataFrame, gamma: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    for model, group in beta.groupby("model"):
        axes[0].plot(
            group["beta"], group["arms_race_fraction"], marker="o", ms=3,
            label=MODEL_LABELS[model],
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"Planner quality scale $\beta$")
    axes[0].set_ylabel("Excess-compute share")
    axes[0].grid(alpha=0.25)
    for model, group in gamma.groupby("model"):
        axes[1].plot(
            group["gamma"], group["arms_race_fraction"], marker="o", ms=3,
            label=MODEL_LABELS[model],
        )
    axes[1].set_xlabel(r"Contest sensitivity $\gamma$")
    axes[1].set_ylabel("Excess-compute share")
    axes[1].grid(alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="robustness_extensions/server_outputs")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--auction-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    output = Path(args.output_dir)
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bootstrap_uncertainty import aggregate_game, solve_grid

    greedy = load_saved_outputs(output, "fixed_*_reasoning_greedy.jsonl")
    stochastic = load_saved_outputs(output, "fixed_*_reasoning_stochastic.jsonl")
    values = np.arange(25, 1001, 25, dtype=float)
    lambdas = np.linspace(0.0001, 0.03, 60)
    epsilon = 0.01
    rng = np.random.default_rng(args.seed)

    beta, gamma = strategic_sensitivities(
        stochastic, solve_grid, aggregate_game, values, lambdas, epsilon
    )
    menu, menu_draws = deterministic_menu_sensitivity(
        greedy,
        stochastic,
        solve_grid,
        aggregate_game,
        values,
        lambdas,
        epsilon,
        args.samples,
        rng,
    )
    scenarios = pd.read_csv(output / "data" / "scenarios.csv", dtype={"scenario_id": str})
    scenarios["scenario_id"] = scenarios["scenario_id"].str.zfill(12)
    auctions = pd.read_csv(output / "data" / "auction_splits.csv")
    auction_summary, auction_draws = auction_cdf_bootstrap(
        greedy, scenarios, auctions, args.auction_samples, rng
    )

    beta.to_csv(tables / "game_beta_sensitivity_submission.csv", index=False)
    gamma.to_csv(tables / "game_gamma_sensitivity_submission.csv", index=False)
    menu.to_csv(tables / "compute_menu_robustness_submission.csv", index=False)
    menu_draws.to_csv(tables / "compute_menu_robustness_bootstrap.csv", index=False)
    auction_summary.to_csv(tables / "auction_cdf_bootstrap_summary.csv", index=False)
    auction_draws.to_csv(tables / "auction_cdf_bootstrap_draws.csv", index=False)
    plot_strategic_sensitivity(beta, gamma, figures / "strategic_parameter_sensitivity.pdf")
    print(f"Wrote submission robustness outputs to {output}")


if __name__ == "__main__":
    main()
