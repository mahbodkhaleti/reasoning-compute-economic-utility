#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path

import pandas as pd
import torch

from common import (
    ModelHarness,
    append_jsonl,
    auction_prompt,
    ensure_dirs,
    find_model,
    load_config,
    meter_fields,
    parse_budget_list,
    parse_final_bid,
    read_jsonl,
    score_bid,
    shuffled_candidates,
    stable_int,
    verbal_confidence_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model", required=True, help="Model slug from config.json")
    parser.add_argument("--chat-mode", choices=["reasoning", "no_think"], default="reasoning")
    parser.add_argument("--decoding", choices=["greedy", "stochastic"], default="greedy")
    parser.add_argument("--budgets", default=None, help="Comma list; use 'unconstrained' for -1")
    parser.add_argument("--repeat-seeds", default=None, help="Comma-separated seeds; defaults to config")
    parser.add_argument(
        "--option-order-seed",
        type=int,
        default=None,
        help="Hold option order fixed at this seed while generation seeds vary",
    )
    parser.add_argument("--gpu-hour-usd", type=float, default=None)
    parser.add_argument("--electricity-usd-per-kwh", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Debug only: cap scenarios")
    parser.add_argument("--query-verbal-confidence", action="store_true")
    parser.add_argument(
        "--constrained-option-readout",
        action="store_true",
        help="Use candidate-letter logits for an always-valid, uniform final bid readout",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.gpu_hour_usd is not None:
        config["cost"]["gpu_hour_usd"] = args.gpu_hour_usd
    if args.electricity_usd_per_kwh is not None:
        config["cost"]["electricity_usd_per_kwh"] = args.electricity_usd_per_kwh
    paths = ensure_dirs(args.output_dir)
    scenarios = pd.read_csv(paths["data"] / "scenarios.csv").to_dict("records")
    if args.limit:
        scenarios = scenarios[: args.limit]
    budgets = parse_budget_list(args.budgets, config["evaluation"]["budgets"])
    repeat_seeds = (
        [int(x.strip()) for x in args.repeat_seeds.split(",")]
        if args.repeat_seeds
        else list(config["evaluation"]["repeat_seeds"])
    )
    spec = find_model(config, args.model)
    output_path = paths["raw"] / f"fixed_{args.model}_{args.chat_mode}_{args.decoding}.jsonl"
    completed = {
        (x["scenario_id"], int(x["budget"]), int(x["repeat_seed"]))
        for x in read_jsonl(output_path)
        if (not args.query_verbal_confidence or x.get("verbal_confidence") is not None)
        and (
            not args.constrained_option_readout
            or x.get("readout_method")
            in {
                "constrained_option_letter_next_token",
                "saved_trace_constrained_option_letter_next_token",
            }
        )
    }
    harness = ModelHarness(spec, config)
    print(f"Loaded {spec['hf_id']} on {torch.cuda.get_device_name(0)}")

    for scenario in scenarios:
        scenario["scenario_id"] = str(scenario["scenario_id"]).zfill(12)
        for budget in budgets:
            for repeat_seed in repeat_seeds:
                key = (scenario["scenario_id"], int(budget), int(repeat_seed))
                if key in completed:
                    continue
                order_seed = repeat_seed if args.option_order_seed is None else args.option_order_seed
                candidates = shuffled_candidates(scenario, order_seed)
                prompt = auction_prompt(float(scenario["value"]), candidates)
                generation_seed = stable_int(config["seed"], args.model, args.chat_mode, *key)
                generated = harness.auction_decision(
                    prompt,
                    budget,
                    generation_seed,
                    args.chat_mode,
                    sample=args.decoding == "stochastic",
                )
                freeform_final_text = generated["final_text"]
                freeform_chosen = parse_final_bid(freeform_final_text, candidates)
                readout_fields = {}
                if args.constrained_option_readout:
                    constrained = harness.constrained_bid_readout(
                        prompt,
                        generated["reasoning_text"],
                        candidates,
                        args.chat_mode,
                    )
                    chosen = constrained["chosen_bid"]
                    generated["final_text"] = (
                        f"FINAL OPTION: {constrained['option_label']}; FINAL BID: {chosen:.2f}"
                    )
                    probability = constrained["option_probability"]
                    generated["answer_mean_logprob"] = math.log(max(probability, 1e-300))
                    generated["answer_min_logprob"] = generated["answer_mean_logprob"]
                    generated["answer_geomean_probability"] = probability
                    readout_fields = {
                        "readout_method": "constrained_option_letter_next_token",
                        "freeform_final_text": freeform_final_text,
                        "freeform_chosen_bid": freeform_chosen,
                        "constrained_option_label": constrained["option_label"],
                        "constrained_option_probability": probability,
                        "constrained_option_probabilities": constrained["option_probabilities"],
                        "constrained_option_token_ids": constrained["option_token_ids"],
                        "constrained_readout_context_tokens": constrained["context_tokens"],
                        **{
                            f"constrained_readout_{key}": value
                            for key, value in meter_fields(constrained["meter"], config).items()
                        },
                    }
                else:
                    chosen = freeform_chosen
                score = score_bid(chosen, candidates)
                confidence = {}
                if args.query_verbal_confidence:
                    queried = harness.confidence_query(
                        verbal_confidence_prompt(prompt, chosen),
                        stable_int(generation_seed, "verbal-confidence"),
                    )
                    confidence = {
                        "verbal_confidence": queried["verbal_confidence"],
                        "verbal_confidence_text": queried["verbal_confidence_text"],
                        "verbal_confidence_tokens": queried["verbal_confidence_tokens"],
                        **{
                            f"verbal_confidence_{key}": value
                            for key, value in meter_fields(queried["verbal_confidence_meter"], config).items()
                        },
                    }
                row = {
                    "model": args.model,
                    "hf_id": spec["hf_id"],
                    "chat_mode": args.chat_mode,
                    "decoding": args.decoding,
                    "scenario_id": scenario["scenario_id"],
                    "category": scenario["category"],
                    "value": float(scenario["value"]),
                    "option_count": int(scenario["option_count"]),
                    "candidate_set_index": int(scenario["candidate_set_index"]),
                    "repeat_seed": int(repeat_seed),
                    "option_order_seed": int(order_seed),
                    "generation_seed": int(generation_seed),
                    "budget": int(budget),
                    "option_order": [c["bid"] for c in candidates],
                    "reasoning_tokens": generated["reasoning_tokens"],
                    "answer_tokens": generated["answer_tokens"],
                    "answer_mean_logprob": generated.get("answer_mean_logprob"),
                    "answer_min_logprob": generated.get("answer_min_logprob"),
                    "answer_geomean_probability": generated.get("answer_geomean_probability"),
                    "reasoning_text": generated["reasoning_text"],
                    "final_text": generated["final_text"],
                    **score,
                    **meter_fields(generated["meter"], config),
                    **readout_fields,
                    **confidence,
                }
                append_jsonl(output_path, row)
                completed.add(key)
                print(
                    f"{args.model} {args.chat_mode} scenario={scenario['scenario_id']} "
                    f"budget={budget} seed={repeat_seed} valid={score['valid_output']} "
                    f"test_ratio={score['test_utility_ratio']:.4f}"
                )

    metadata = {
        "model": spec,
        "chat_mode": args.chat_mode,
        "budgets": budgets,
        "repeat_seeds": repeat_seeds,
        "decoding": args.decoding,
        "option_order_seed": args.option_order_seed,
        "query_verbal_confidence": args.query_verbal_confidence,
        "constrained_option_readout": args.constrained_option_readout,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "accelerate": importlib.metadata.version("accelerate"),
        "python": platform.python_version(),
        "model_revision": harness.model_revision,
        "tokenizer_revision": harness.tokenizer_revision,
        "model_class": type(harness.model).__name__,
        "tokenizer_class": type(harness.tokenizer).__name__,
        "dtype": str(next(harness.model.parameters()).dtype),
        "gpu_hour_usd": config["cost"].get("gpu_hour_usd"),
        "electricity_usd_per_kwh": config["cost"].get("electricity_usd_per_kwh"),
    }
    (paths["raw"] / f"fixed_{args.model}_{args.chat_mode}_{args.decoding}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
