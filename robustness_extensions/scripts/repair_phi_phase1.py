#!/usr/bin/env python3
"""Repair Phi Phase-1 free-form readout with constrained option scoring.

The original reasoning traces are retained.  For every row, this script
reconstructs the randomized auction prompt, appends the saved reasoning trace,
and compares the next-token logits for the displayed option letters.  This
provides one uniform, always-valid readout without regenerating the reasoning
trajectory.  The repaired data are written to a separate checkpointed JSONL;
the source file is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from common import (
    ModelHarness,
    append_jsonl,
    auction_prompt,
    find_model,
    load_config,
    parse_candidate_json,
    read_jsonl,
    score_bid,
)


def load_scenarios(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            str(row["scenario_id"]).zfill(12): row
            for row in csv.DictReader(handle)
        }


def ordered_candidates(scenario: dict[str, Any], option_order: list[float]) -> list[dict[str, float]]:
    candidates = parse_candidate_json(scenario["candidates_json"])
    ordered: list[dict[str, float]] = []
    for raw_bid in option_order:
        bid = float(raw_bid)
        match = next((item for item in candidates if abs(item["bid"] - bid) < 1e-6), None)
        if match is None:
            raise ValueError(f"Bid {bid} is absent from scenario {scenario['scenario_id']}")
        ordered.append(match)
    return ordered


def single_token_labels(tokenizer: Any, count: int) -> tuple[list[str], list[int]]:
    labels = [chr(65 + index) for index in range(count)]
    token_ids: list[int] = []
    for label in labels:
        variants = [f" {label}", label]
        encoded = [tokenizer.encode(text, add_special_tokens=False) for text in variants]
        one_token = next((ids for ids in encoded if len(ids) == 1), None)
        if one_token is None:
            raise RuntimeError(
                f"Option label {label!r} is not a single token under this tokenizer: {encoded}"
            )
        token_ids.append(int(one_token[0]))
    if len(set(token_ids)) != len(token_ids):
        raise RuntimeError(f"Option labels do not have distinct token IDs: {token_ids}")
    return labels, token_ids


def constrained_readout(
    harness: ModelHarness,
    prompt: str,
    reasoning_text: str,
    option_count: int,
) -> dict[str, Any]:
    tokenizer = harness.tokenizer
    prefix = harness._chat_text(prompt, "reasoning")
    prefix_ids = tokenizer(prefix, return_tensors="pt")["input_ids"].to(harness.input_device)
    if reasoning_text:
        reasoning_ids = tokenizer(
            reasoning_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(harness.input_device)
    else:
        reasoning_ids = torch.empty(
            (1, 0), dtype=prefix_ids.dtype, device=harness.input_device
        )
    suffix_ids = tokenizer(
        "\n\nUse the reasoning above and select the best displayed option. "
        "Return one option letter only. FINAL OPTION:",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(harness.input_device)
    input_ids = torch.cat([prefix_ids, reasoning_ids, suffix_ids], dim=1)
    labels, label_token_ids = single_token_labels(tokenizer, option_count)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        output = harness.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        )
    elapsed = time.perf_counter() - start
    logits = output.logits[0, -1].float()
    candidate_logits = torch.stack([logits[token_id] for token_id in label_token_ids])
    probabilities = torch.softmax(candidate_logits, dim=0)
    selected_index = int(torch.argmax(candidate_logits).item())
    return {
        "selected_index": selected_index,
        "selected_label": labels[selected_index],
        "selected_probability": float(probabilities[selected_index].item()),
        "label_probabilities": {
            label: float(probabilities[index].item()) for index, label in enumerate(labels)
        },
        "label_token_ids": dict(zip(labels, label_token_ids)),
        "repair_context_tokens": int(input_ids.numel()),
        "repair_readout_seconds": elapsed,
        "repair_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="phi4_mini_reasoning")
    args = parser.parse_args()

    config = load_config(args.config)
    spec = find_model(config, args.model)
    source_rows = read_jsonl(args.input)
    if not source_rows:
        raise SystemExit(f"No source rows found in {args.input}")
    scenarios = load_scenarios(Path(args.scenarios))
    output_path = Path(args.output)
    completed = {
        (str(row["scenario_id"]).zfill(12), int(row["budget"]), int(row["repeat_seed"]))
        for row in read_jsonl(output_path)
    }

    harness = ModelHarness(spec, config)
    print(f"Loaded {spec['hf_id']} for constrained Phi readout")
    for source in source_rows:
        scenario_id = str(source["scenario_id"]).zfill(12)
        key = (scenario_id, int(source["budget"]), int(source["repeat_seed"]))
        if key in completed:
            continue
        scenario = scenarios.get(scenario_id)
        if scenario is None:
            raise KeyError(f"Missing scenario {scenario_id}")
        option_order = [float(value) for value in source["option_order"]]
        candidates = ordered_candidates(scenario, option_order)
        prompt = auction_prompt(float(source["value"]), candidates)
        repaired = constrained_readout(
            harness,
            prompt,
            str(source.get("reasoning_text", "")),
            len(candidates),
        )
        chosen_bid = float(candidates[repaired["selected_index"]]["bid"])

        row = dict(source)
        row.update(
            {
                "original_valid_output": bool(source.get("valid_output")),
                "original_chosen_bid": source.get("chosen_bid"),
                "original_final_text": source.get("final_text", ""),
                "readout_method": "saved_trace_constrained_option_letter_next_token",
                "constrained_option_label": repaired["selected_label"],
                "constrained_option_probability": repaired["selected_probability"],
                "constrained_option_probabilities": repaired["label_probabilities"],
                "constrained_option_token_ids": repaired["label_token_ids"],
                "repair_context_tokens": repaired["repair_context_tokens"],
                "repair_readout_seconds": repaired["repair_readout_seconds"],
                "repair_peak_memory_bytes": repaired["repair_peak_memory_bytes"],
                "answer_mean_logprob": math.log(
                    max(repaired["selected_probability"], 1e-300)
                ),
                "answer_min_logprob": math.log(
                    max(repaired["selected_probability"], 1e-300)
                ),
                "answer_geomean_probability": repaired["selected_probability"],
                "chosen_bid": chosen_bid,
                "final_text": (
                    f"FINAL OPTION: {repaired['selected_label']}; FINAL BID: {chosen_bid:.2f}"
                ),
                **score_bid(chosen_bid, candidates),
            }
        )
        append_jsonl(output_path, row)
        completed.add(key)
        print(
            f"repaired scenario={scenario_id} budget={source['budget']} "
            f"seed={source['repeat_seed']} option={repaired['selected_label']} "
            f"bid={chosen_bid:.2f} ratio={row['test_utility_ratio']:.4f}"
        )

    repaired_rows = read_jsonl(output_path)
    unique = {
        (str(row["scenario_id"]).zfill(12), int(row["budget"]), int(row["repeat_seed"]))
        for row in repaired_rows
    }
    if len(unique) != len(source_rows) or len(repaired_rows) != len(source_rows):
        raise RuntimeError(
            f"Repair is incomplete: source={len(source_rows)}, output={len(repaired_rows)}, "
            f"unique={len(unique)}"
        )
    if not all(row.get("valid_output") for row in repaired_rows):
        raise RuntimeError("Repair completed but at least one output remains invalid")
    if not all(math.isfinite(float(row["test_utility_ratio"])) for row in repaired_rows):
        raise RuntimeError("Repair produced a non-finite utility ratio")
    summary_rows: list[dict[str, Any]] = []
    for budget in sorted({int(row["budget"]) for row in repaired_rows}):
        group = [row for row in repaired_rows if int(row["budget"]) == budget]
        originally_valid = [row for row in group if row.get("original_valid_output")]
        agreements = [
            abs(float(row["chosen_bid"]) - float(row["original_chosen_bid"])) < 1e-6
            for row in originally_valid
            if row.get("original_chosen_bid") is not None
        ]
        summary_rows.append(
            {
                "budget": budget,
                "n": len(group),
                "valid_rate": sum(bool(row["valid_output"]) for row in group) / len(group),
                "mean_utility_ratio": sum(float(row["test_utility_ratio"]) for row in group) / len(group),
                "mean_regret": sum(float(row["test_regret"]) for row in group) / len(group),
                "accuracy": sum(bool(row["test_correct"]) for row in group) / len(group),
                "originally_valid_n": len(originally_valid),
                "agreement_with_original_valid": (
                    sum(agreements) / len(agreements) if agreements else math.nan
                ),
            }
        )
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "source": str(Path(args.input).resolve()),
                "output": str(output_path.resolve()),
                "model": spec,
                "rows": len(repaired_rows),
                "method": "saved_trace_constrained_option_letter_next_token",
                "scope": "readout repair only; original reasoning traces are unchanged",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Repair complete: {len(repaired_rows)} valid rows written to {output_path}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
