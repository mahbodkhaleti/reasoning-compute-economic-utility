#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    ModelHarness,
    append_jsonl,
    auction_prompt,
    budget_choice_prompt,
    ensure_dirs,
    find_model,
    load_config,
    meter_fields,
    parse_compute_budget,
    parse_final_bid,
    read_jsonl,
    score_bid,
    shuffled_candidates,
    stable_int,
)


def combine_costs(*fields: dict) -> dict[str, float]:
    keys = ["elapsed_seconds", "energy_joules", "rental_cost_usd", "energy_cost_usd", "total_compute_cost_usd"]
    result = {}
    for key in keys:
        values = [float(x[key]) for x in fields if key in x and not math.isnan(float(x[key]))]
        result[key] = float(sum(values)) if values else math.nan
    return result


def history_text(history: list[dict], length: int) -> str:
    if not history:
        return "No previous rounds."
    lines = []
    for item in history[-length:]:
        lines.append(
            f"round {item['round']}: opponent budget {item['opponent_budget']}, "
            f"opponent bid ${item['opponent_bid']:.2f}, you {'won' if item['won'] else 'did not win'}"
        )
    return "\n".join(lines)


def choose_budget(
    harness: ModelHarness,
    mode: str,
    allowed: list[int],
    value: float,
    frontier_text: str,
    opponent_history: str,
    gpu_hour_usd: float,
    seconds_per_token: float,
    seed: int,
    config: dict,
) -> tuple[int, bool, str, dict]:
    if mode == "fixed_low":
        return int(config["direct_game"]["fixed_low_budget"]), True, "fixed_low", {}
    if mode == "fixed_high":
        return int(config["direct_game"]["fixed_high_budget"]), True, "fixed_high", {}
    prompt = budget_choice_prompt(value, allowed, frontier_text, opponent_history, gpu_hour_usd, seconds_per_token)
    text, tokens, measured = harness.generate_text(
        prompt,
        max_tokens=config["direct_game"]["choice_tokens"],
        seed=seed,
        chat_mode="no_think",
        sample=True,
    )
    selected = parse_compute_budget(text, allowed)
    valid = selected is not None
    if selected is None:
        selected = min(allowed)
    fields = meter_fields(measured, config)
    fields["choice_tokens"] = tokens
    return selected, valid, text, fields


def auction_payoffs(value_a: float, bid_a: float | None, cost_a: float, value_b: float, bid_b: float | None, cost_b: float, tie_seed: int) -> dict:
    effective_a = 0.0 if bid_a is None else bid_a
    effective_b = 0.0 if bid_b is None else bid_b
    if effective_a > effective_b:
        winner = "A"
    elif effective_b > effective_a:
        winner = "B"
    else:
        winner = "A" if random.Random(tie_seed).random() < 0.5 else "B"
    winner_value = value_a if winner == "A" else value_b
    winner_bid = effective_a if winner == "A" else effective_b
    private_a = (max(value_a - effective_a, 0.0) if winner == "A" else 0.0) - cost_a
    private_b = (max(value_b - effective_b, 0.0) if winner == "B" else 0.0) - cost_b
    efficient_value = max(value_a, value_b)
    allocative_loss = efficient_value - winner_value
    compute_cost = cost_a + cost_b
    social_welfare = winner_value - compute_cost  # payment is a transfer; seller value normalized to zero.
    return {
        "winner": winner,
        "winner_value": winner_value,
        "winner_bid": winner_bid,
        "private_payoff_a": private_a,
        "private_payoff_b": private_b,
        "efficient_welfare_usd": efficient_value,
        "social_welfare_usd": social_welfare,
        "allocative_loss_usd": allocative_loss,
        "compute_cost_usd": compute_cost,
        "total_welfare_loss_usd": efficient_value - social_welfare,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["adaptive", "fixed_low", "fixed_high"], required=True)
    parser.add_argument("--gpu-hour-usd", type=float, required=True)
    parser.add_argument("--electricity-usd-per-kwh", type=float, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    args = parser.parse_args()

    if args.gpu_hour_usd <= 0:
        raise ValueError("--gpu-hour-usd must be positive so payoffs and welfare use observed dollar costs")
    config = load_config(args.config)
    config["cost"]["gpu_hour_usd"] = args.gpu_hour_usd
    if args.electricity_usd_per_kwh is not None:
        config["cost"]["electricity_usd_per_kwh"] = args.electricity_usd_per_kwh
    paths = ensure_dirs(args.output_dir)
    spec = find_model(config, args.model)
    scenarios = pd.read_csv(paths["data"] / "scenarios.csv", dtype={"scenario_id": str})
    scenarios["scenario_id"] = scenarios["scenario_id"].str.zfill(12)
    frontier = pd.read_csv(paths["tables"] / "model_frontiers.csv")
    frontier = frontier.loc[frontier["model"] == args.model].sort_values("budget").copy()
    if frontier.empty:
        raise RuntimeError(f"No frontier for {args.model}; run analyze_fixed.py first")
    # Fixed-budget inference may have been collected before a rental price was
    # supplied.  Reconstruct the price used in the budget-choice prompt from the
    # measured runtime rather than displaying a stale zero-dollar cost.
    frontier["mean_compute_cost_usd"] = (
        frontier["mean_seconds"].astype(float) * args.gpu_hour_usd / 3600.0
    )
    allowed = [int(x) for x in frontier["budget"] if int(x) >= 0]
    frontier_text = "\n".join(
        f"budget {int(r.budget)}: held-out utility ratio {r.mean_test_utility_ratio:.4f}, "
        f"mean regret ${r.mean_test_regret_usd:.4f}, mean measured cost ${r.mean_compute_cost_usd:.8f}"
        for r in frontier.itertuples(index=False)
    )
    positive = frontier.loc[frontier["mean_reasoning_tokens"] > 0]
    seconds_per_token = float((positive["mean_seconds"] / positive["mean_reasoning_tokens"]).median())
    rounds = args.rounds or config["direct_game"]["rounds"]
    output_path = paths["direct"] / f"direct_{args.model}_{args.mode}.jsonl"
    prior = [
        row
        for row in read_jsonl(output_path)
        if row.get("bid_readout_method") == "constrained_option_letter_next_token"
        and math.isclose(float(row.get("gpu_hour_usd", math.nan)), args.gpu_hour_usd)
    ]
    completed = {(int(x["experiment_seed"]), int(x["round"])) for x in prior}
    harness = ModelHarness(spec, config)

    for experiment_seed in config["direct_game"]["experiment_seeds"]:
        history_a = []
        history_b = []
        previous = sorted(
            [x for x in prior if int(x["experiment_seed"]) == experiment_seed],
            key=lambda x: int(x["round"]),
        )
        for x in previous:
            history_a.append(
                {
                    "round": x["round"],
                    "opponent_budget": x["budget_b"],
                    "opponent_bid": x["bid_b"] or 0.0,
                    "won": x["winner"] == "A",
                }
            )
            history_b.append(
                {
                    "round": x["round"],
                    "opponent_budget": x["budget_a"],
                    "opponent_bid": x["bid_a"] or 0.0,
                    "won": x["winner"] == "B",
                }
            )
        categories = sorted(scenarios["category"].unique())
        for round_index in range(rounds):
            round_rng = np.random.default_rng(stable_int(config["seed"], experiment_seed, round_index, "direct-scenarios"))
            category = categories[round_index % len(categories)]
            pool = scenarios.loc[scenarios["category"] == category]
            selected_indices = round_rng.choice(pool.index.to_numpy(), size=2, replace=False)
            if (experiment_seed, round_index) in completed:
                continue
            sa = pool.loc[selected_indices[0]].to_dict()
            sb = pool.loc[selected_indices[1]].to_dict()
            value_a, value_b = float(sa["value"]), float(sb["value"])
            seed_a = stable_int(config["seed"], args.model, args.mode, experiment_seed, round_index, "A")
            seed_b = stable_int(config["seed"], args.model, args.mode, experiment_seed, round_index, "B")
            budget_a, choice_valid_a, choice_text_a, choice_cost_a = choose_budget(
                harness, args.mode, allowed, value_a, frontier_text,
                history_text(history_a, config["direct_game"]["history_length"]),
                args.gpu_hour_usd, seconds_per_token, seed_a, config,
            )
            budget_b, choice_valid_b, choice_text_b, choice_cost_b = choose_budget(
                harness, args.mode, allowed, value_b, frontier_text,
                history_text(history_b, config["direct_game"]["history_length"]),
                args.gpu_hour_usd, seconds_per_token, seed_b, config,
            )
            options_a = shuffled_candidates(sa, experiment_seed + round_index, "A")
            options_b = shuffled_candidates(sb, experiment_seed + round_index, "B")
            prompt_a = auction_prompt(value_a, options_a, history_text(history_a, config["direct_game"]["history_length"]))
            prompt_b = auction_prompt(value_b, options_b, history_text(history_b, config["direct_game"]["history_length"]))
            out_a = harness.auction_decision(prompt_a, budget_a, stable_int(seed_a, "bid"), "reasoning")
            out_b = harness.auction_decision(prompt_b, budget_b, stable_int(seed_b, "bid"), "reasoning")
            freeform_bid_a = parse_final_bid(out_a["final_text"], options_a)
            freeform_bid_b = parse_final_bid(out_b["final_text"], options_b)
            constrained_a = harness.constrained_bid_readout(
                prompt_a, out_a["reasoning_text"], options_a, "reasoning"
            )
            constrained_b = harness.constrained_bid_readout(
                prompt_b, out_b["reasoning_text"], options_b, "reasoning"
            )
            bid_a = constrained_a["chosen_bid"]
            bid_b = constrained_b["chosen_bid"]
            score_a = score_bid(bid_a, options_a)
            score_b = score_bid(bid_b, options_b)
            bid_cost_a = meter_fields(out_a["meter"], config)
            bid_cost_b = meter_fields(out_b["meter"], config)
            readout_cost_a = meter_fields(constrained_a["meter"], config)
            readout_cost_b = meter_fields(constrained_b["meter"], config)
            total_cost_a = combine_costs(choice_cost_a, bid_cost_a, readout_cost_a)
            total_cost_b = combine_costs(choice_cost_b, bid_cost_b, readout_cost_b)
            payoff = auction_payoffs(
                value_a, bid_a, total_cost_a["total_compute_cost_usd"],
                value_b, bid_b, total_cost_b["total_compute_cost_usd"],
                stable_int(experiment_seed, round_index, "tie"),
            )
            row = {
                "model": args.model,
                "hf_id": spec["hf_id"],
                "mode": args.mode,
                "experiment_seed": experiment_seed,
                "round": round_index,
                "category": category,
                "scenario_a": str(sa["scenario_id"]).zfill(12),
                "scenario_b": str(sb["scenario_id"]).zfill(12),
                "value_a": value_a,
                "value_b": value_b,
                "budget_a": budget_a,
                "budget_b": budget_b,
                "choice_valid_a": choice_valid_a,
                "choice_valid_b": choice_valid_b,
                "choice_text_a": choice_text_a,
                "choice_text_b": choice_text_b,
                "bid_a": bid_a,
                "bid_b": bid_b,
                "bid_readout_method": "constrained_option_letter_next_token",
                "freeform_bid_a": freeform_bid_a,
                "freeform_bid_b": freeform_bid_b,
                "constrained_option_label_a": constrained_a["option_label"],
                "constrained_option_label_b": constrained_b["option_label"],
                "constrained_option_probability_a": constrained_a["option_probability"],
                "constrained_option_probability_b": constrained_b["option_probability"],
                "constrained_option_probabilities_a": constrained_a["option_probabilities"],
                "constrained_option_probabilities_b": constrained_b["option_probabilities"],
                "bid_valid_a": score_a["valid_output"],
                "bid_valid_b": score_b["valid_output"],
                "reasoning_tokens_a": out_a["reasoning_tokens"],
                "reasoning_tokens_b": out_b["reasoning_tokens"],
                "reasoning_text_a": out_a["reasoning_text"],
                "reasoning_text_b": out_b["reasoning_text"],
                "final_text_a": out_a["final_text"],
                "final_text_b": out_b["final_text"],
                "constrained_readout_elapsed_a_seconds": readout_cost_a["elapsed_seconds"],
                "constrained_readout_elapsed_b_seconds": readout_cost_b["elapsed_seconds"],
                "constrained_readout_energy_a_joules": readout_cost_a["energy_joules"],
                "constrained_readout_energy_b_joules": readout_cost_b["energy_joules"],
                "compute_cost_a_usd": total_cost_a["total_compute_cost_usd"],
                "compute_cost_b_usd": total_cost_b["total_compute_cost_usd"],
                "elapsed_a_seconds": total_cost_a["elapsed_seconds"],
                "elapsed_b_seconds": total_cost_b["elapsed_seconds"],
                "energy_a_joules": total_cost_a["energy_joules"],
                "energy_b_joules": total_cost_b["energy_joules"],
                "gpu_hour_usd": args.gpu_hour_usd,
                "electricity_usd_per_kwh": config["cost"].get("electricity_usd_per_kwh"),
                **payoff,
            }
            append_jsonl(output_path, row)
            completed.add((experiment_seed, round_index))
            history_a.append({"round": round_index, "opponent_budget": budget_b, "opponent_bid": bid_b or 0.0, "won": payoff["winner"] == "A"})
            history_b.append({"round": round_index, "opponent_budget": budget_a, "opponent_bid": bid_a or 0.0, "won": payoff["winner"] == "B"})
            print(
                f"{args.model} {args.mode} seed={experiment_seed} round={round_index} "
                f"budgets=({budget_a},{budget_b}) bids=({bid_a},{bid_b}) "
                f"welfare_loss=${payoff['total_welfare_loss_usd']:.6f}"
            )


if __name__ == "__main__":
    main()
