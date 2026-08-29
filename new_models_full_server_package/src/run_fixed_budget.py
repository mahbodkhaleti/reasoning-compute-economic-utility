#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from common import (
    ModelHarness,
    append_jsonl,
    auction_prompt,
    ensure_dirs,
    find_model,
    load_config,
    meter_fields,
    parse_budget_list,
    parse_candidate_json,
    parse_final_bid,
    read_jsonl,
    score_bid,
    shuffled_candidates,
    stable_int,
    verbal_confidence_prompt,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_driver_version() -> str | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        value = pynvml.nvmlSystemGetDriverVersion()
        return value.decode() if isinstance(value, bytes) else str(value)
    except Exception:
        return None


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def cell_is_complete(row: dict, query_confidence: bool, constrained_readout: bool) -> bool:
    return (
        (not query_confidence or row.get("verbal_confidence") is not None)
        and (
            not constrained_readout
            or row.get("readout_method")
            in {
                "constrained_option_letter_next_token",
                "saved_trace_constrained_option_letter_next_token",
            }
        )
    )


def compact_jsonl(path: Path) -> list[dict]:
    """Atomically retain the last row for each resume key."""
    rows, corrupt = [], []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("row is not a JSON object")
            rows.append(value)
        except Exception as exc:
            corrupt.append({"line": number, "error": f"{type(exc).__name__}: {exc}", "text": line[:500]})
    if corrupt:
        write_json_atomic(path.with_suffix(".corrupt.json"), {"corrupt_rows": corrupt})
    by_key = {
        (str(row["scenario_id"]).zfill(12), int(row["budget"]), int(row["repeat_seed"])): row
        for row in rows
    }
    compacted = list(by_key.values())
    if corrupt or len(compacted) != len(rows):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in compacted:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    return compacted


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
    parser.add_argument("--gpu-count-billed", type=int, default=None)
    parser.add_argument("--hardware-name", default=None)
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
    if args.gpu_count_billed is not None:
        config["cost"]["gpu_count_billed"] = args.gpu_count_billed
    if args.hardware_name is not None:
        config["cost"]["hardware_name"] = args.hardware_name
    if args.electricity_usd_per_kwh is not None:
        config["cost"]["electricity_usd_per_kwh"] = args.electricity_usd_per_kwh
    paths = ensure_dirs(args.output_dir)
    scenario_sha256 = sha256(paths["data"] / "scenarios.csv")
    split_sha256 = sha256(paths["data"] / "auction_splits.csv")
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
    manifest_path = paths["raw"] / f"fixed_{args.model}_{args.chat_mode}_{args.decoding}_run_manifest.json"
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    if manifest_path.exists():
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest = {
            "model_slug": args.model,
            "hf_id": spec["hf_id"],
            "requested_model_revision": spec.get("revision"),
            "requested_tokenizer_revision": spec.get("tokenizer_revision", spec.get("revision")),
            "chat_mode": args.chat_mode,
            "decoding": args.decoding,
            "budgets": budgets,
            "repeat_seeds": repeat_seeds,
            "option_order_seed": args.option_order_seed,
            "query_verbal_confidence": args.query_verbal_confidence,
            "constrained_option_readout": args.constrained_option_readout,
            "gpu_hour_usd": config["cost"].get("gpu_hour_usd"),
            "gpu_count_billed": config["cost"].get("gpu_count_billed", 1),
            "hardware_name": config["cost"].get("hardware_name"),
            "scenario_manifest_sha256": scenario_sha256,
            "auction_split_sha256": split_sha256,
        }
        mismatches = {
            key: {"recorded": run_manifest.get(key), "requested": value}
            for key, value in expected_manifest.items()
            if run_manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Existing run manifest does not match this invocation; use a separate "
                f"output directory. Mismatches: {json.dumps(mismatches, sort_keys=True)}"
            )
        run_manifest.setdefault("resume_events_utc", []).append(now)
    else:
        config_path = Path(args.config).resolve()
        run_manifest = {
            "status": "initialized_before_model_load",
            "first_started_at_utc": now,
            "model_slug": args.model,
            "hf_id": spec["hf_id"],
            "requested_model_revision": spec.get("revision"),
            "requested_tokenizer_revision": spec.get("tokenizer_revision", spec.get("revision")),
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "repository_git_revision": git_revision(Path(__file__).resolve().parents[2]),
            "chat_mode": args.chat_mode,
            "decoding": args.decoding,
            "budgets": budgets,
            "repeat_seeds": repeat_seeds,
            "option_order_seed": args.option_order_seed,
            "query_verbal_confidence": args.query_verbal_confidence,
            "constrained_option_readout": args.constrained_option_readout,
            "gpu_hour_usd": config["cost"].get("gpu_hour_usd"),
            "gpu_count_billed": config["cost"].get("gpu_count_billed", 1),
            "hardware_name": config["cost"].get("hardware_name"),
            "scenario_manifest_sha256": scenario_sha256,
            "auction_split_sha256": split_sha256,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "accelerate": importlib.metadata.version("accelerate"),
            "gpu_driver": gpu_driver_version(),
            "configured_device_map": spec.get("device_map", "auto"),
            "reasoning_protocol": spec.get("reasoning_protocol"),
            "reasoning_protocol_matched": bool(spec.get("reasoning_protocol_matched", True)),
        }
    write_json_atomic(manifest_path, run_manifest)
    existing_rows = compact_jsonl(output_path) if output_path.exists() else []
    completed = {
        (str(x["scenario_id"]).zfill(12), int(x["budget"]), int(x["repeat_seed"]))
        for x in existing_rows
        if cell_is_complete(x, args.query_verbal_confidence, args.constrained_option_readout)
    }
    required = {
        (str(scenario["scenario_id"]).zfill(12), int(budget), int(repeat_seed))
        for scenario in scenarios
        for budget in budgets
        for repeat_seed in repeat_seeds
    }
    if required <= completed:
        run_manifest["status"] = "already_complete_no_model_loaded"
        run_manifest["last_checked_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        run_manifest["completed_cells"] = len(required)
        write_json_atomic(manifest_path, run_manifest)
        print(f"{output_path.name} already contains all {len(required)} required cells; skipping model load")
        return
    harness = ModelHarness(spec, config)
    expected_model_revision = spec.get("revision")
    expected_tokenizer_revision = spec.get("tokenizer_revision", expected_model_revision)
    if harness.model_revision != expected_model_revision:
        raise RuntimeError(
            "Resolved model revision does not match the pinned revision: "
            f"expected {expected_model_revision}, got {harness.model_revision}"
        )
    if harness.tokenizer_revision != expected_tokenizer_revision:
        raise RuntimeError(
            "Resolved tokenizer revision does not match the pinned revision: "
            f"expected {expected_tokenizer_revision}, got {harness.tokenizer_revision}"
        )
    run_manifest.update(
        {
            "status": "model_loaded_before_inference",
            "model_loaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "resolved_model_revision": harness.model_revision,
            "resolved_tokenizer_revision": harness.tokenizer_revision,
            "model_class": type(harness.model).__name__,
            "tokenizer_class": type(harness.tokenizer).__name__,
            "dtype": str(next(harness.model.parameters()).dtype),
            "resolved_device_map": getattr(harness.model, "hf_device_map", None),
            "resolved_generation_config": {
                key: getattr(harness.model.generation_config, key, None)
                for key in (
                    "do_sample",
                    "temperature",
                    "top_p",
                    "top_k",
                    "eos_token_id",
                    "pad_token_id",
                )
            },
            "gpu_devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    )
    write_json_atomic(manifest_path, run_manifest)
    print(f"Loaded {spec['hf_id']} on {torch.cuda.get_device_name(0)}")
    progress = tqdm(total=len(required), initial=len(required & completed), desc=f"{args.model}:{args.chat_mode}:{args.decoding}")

    for scenario in scenarios:
        scenario["scenario_id"] = str(scenario["scenario_id"]).zfill(12)
        canonical_candidates = [
            {"candidate_id": f"c{index}", **candidate}
            for index, candidate in enumerate(parse_candidate_json(scenario["candidates_json"]))
        ]
        candidate_id_by_bid = {candidate["bid"]: candidate["candidate_id"] for candidate in canonical_candidates}
        for budget in budgets:
            for repeat_seed in repeat_seeds:
                key = (scenario["scenario_id"], int(budget), int(repeat_seed))
                if key in completed:
                    continue
                order_seed = repeat_seed if args.option_order_seed is None else args.option_order_seed
                candidates = shuffled_candidates(scenario, order_seed)
                prompt = auction_prompt(float(scenario["value"]), candidates)
                generation_seed = stable_int(config["seed"], args.model, args.chat_mode, *key)
                try:
                    generated = harness.auction_decision(
                        prompt,
                        budget,
                        generation_seed,
                        args.chat_mode,
                        sample=args.decoding == "stochastic",
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    run_manifest.update(status="cuda_oom", failed_cell=list(key), error=str(exc))
                    write_json_atomic(manifest_path, run_manifest)
                    raise RuntimeError(f"CUDA OOM at cell {key}; saved work is resumable") from exc
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
                        "freeform_valid": freeform_chosen is not None,
                        "constrained_freeform_agreement": bool(
                            freeform_chosen is not None and math.isclose(freeform_chosen, chosen)
                        ),
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
                meter_parts = [generated["meter"]]
                if args.constrained_option_readout:
                    meter_parts.append(constrained["meter"])
                if args.query_verbal_confidence:
                    meter_parts.append(queried["verbal_confidence_meter"])
                finite_energy = [part.energy_joules for part in meter_parts if math.isfinite(part.energy_joules)]
                row = {
                    "record_schema_version": "auction-reasoning-cell-v2",
                    "benchmark_name": "historical_ebay_auction_decision",
                    "benchmark_version": "historical-auction-paper-v1",
                    "scenario_manifest_sha256": scenario_sha256,
                    "auction_split_sha256": split_sha256,
                    "model": args.model,
                    "hf_id": spec["hf_id"],
                    "requested_model_revision": spec.get("revision"),
                    "requested_tokenizer_revision": spec.get("tokenizer_revision", spec.get("revision")),
                    "resolved_model_revision": harness.model_revision,
                    "resolved_tokenizer_revision": harness.tokenizer_revision,
                    "reasoning_protocol": spec.get("reasoning_protocol"),
                    "reasoning_protocol_matched": bool(spec.get("reasoning_protocol_matched", True)),
                    "chat_template_argument": spec.get("chat_template_argument"),
                    "dtype": str(next(harness.model.parameters()).dtype),
                    "device_map": spec.get("device_map", "auto"),
                    "resolved_device_map": run_manifest.get("resolved_device_map"),
                    "gpu_devices": run_manifest.get("gpu_devices"),
                    "gpu_driver": run_manifest.get("gpu_driver"),
                    "hardware_name": config["cost"].get("hardware_name"),
                    "gpu_hour_usd": config["cost"].get("gpu_hour_usd"),
                    "gpu_count_billed": config["cost"].get("gpu_count_billed", 1),
                    "total_pipeline_elapsed_seconds": sum(part.seconds for part in meter_parts),
                    "total_pipeline_energy_joules": sum(finite_energy) if finite_energy else math.nan,
                    "total_pipeline_peak_memory_bytes": max(part.peak_memory_bytes for part in meter_parts),
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
                    "canonical_candidate_set": canonical_candidates,
                    "displayed_candidates": [
                        {"display_label": chr(65 + index), "candidate_id": candidate_id_by_bid[candidate["bid"]], **candidate}
                        for index, candidate in enumerate(candidates)
                    ],
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
                if cell_is_complete(row, args.query_verbal_confidence, args.constrained_option_readout):
                    completed.add(key)
                    progress.update(1)
                print(
                    f"{args.model} {args.chat_mode} scenario={scenario['scenario_id']} "
                    f"budget={budget} seed={repeat_seed} valid={score['valid_output']} "
                    f"test_ratio={score['test_utility_ratio']:.4f}"
                )

    progress.close()
    final_rows = compact_jsonl(output_path)
    final_completed = {
        (str(row["scenario_id"]).zfill(12), int(row["budget"]), int(row["repeat_seed"]))
        for row in final_rows
        if cell_is_complete(row, args.query_verbal_confidence, args.constrained_option_readout)
    }
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
        "revision_recorded_before_inference": True,
        "run_manifest": manifest_path.name,
        "model_class": type(harness.model).__name__,
        "tokenizer_class": type(harness.tokenizer).__name__,
        "dtype": str(next(harness.model.parameters()).dtype),
        "gpu_hour_usd": config["cost"].get("gpu_hour_usd"),
        "electricity_usd_per_kwh": config["cost"].get("electricity_usd_per_kwh"),
    }
    (paths["raw"] / f"fixed_{args.model}_{args.chat_mode}_{args.decoding}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    is_complete = required <= final_completed
    run_manifest["status"] = "complete" if is_complete else "incomplete_requires_resume"
    run_manifest["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat() if is_complete else None
    run_manifest["completed_cells"] = len(required.intersection(final_completed))
    write_json_atomic(manifest_path, run_manifest)
    if not is_complete:
        raise RuntimeError(
            f"{len(required - final_completed)} cells lack a complete readout; rerun the same command to resume"
        )


if __name__ == "__main__":
    main()
