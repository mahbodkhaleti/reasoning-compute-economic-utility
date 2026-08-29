#!/usr/bin/env python3
"""Checkpoint-resumable inference for the canonical synthetic IPV benchmark."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
from pathlib import Path

import torch
from tqdm.auto import tqdm

from common import ModelHarness, append_jsonl, auction_prompt, load_config, meter_fields, parse_final_bid, score_bid, stable_int


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_rows(path: Path) -> tuple[list[dict], list[dict]]:
    rows, corrupt = [], []
    if not path.exists():
        return rows, corrupt
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("row is not an object")
            rows.append(row)
        except Exception as exc:
            corrupt.append({"line": number, "error": f"{type(exc).__name__}: {exc}", "text": line[:500]})
    return rows, corrupt


def compact(path: Path, rows: list[dict], force: bool = False) -> list[dict]:
    by_key = {}
    for row in rows:
        try:
            key = (row["scenario_id"], int(row["reasoning_budget"]), int(row["option_order_seed"]))
        except (KeyError, TypeError, ValueError):
            continue
        by_key[key] = row
    compacted = list(by_key.values())
    if force or compacted != rows:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in compacted:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    return compacted


def load_benchmark(data_dir: Path) -> tuple[list[dict], str, str]:
    manifest = data_dir / "synthetic_ipv_manifest.jsonl"
    hash_file = data_dir / "manifest.sha256"
    expected = hash_file.read_text(encoding="ascii").split()[0]
    actual = sha256(manifest)
    if actual != expected:
        raise RuntimeError(f"Synthetic benchmark checksum mismatch: expected {expected}, got {actual}")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 216 or len({row["scenario_id"] for row in rows}) != 216:
        raise RuntimeError("Synthetic benchmark must contain exactly 216 unique scenarios")
    versions = {row["benchmark_version"] for row in rows}
    if len(versions) != 1:
        raise RuntimeError(f"Mixed benchmark versions: {versions}")
    return rows, actual, versions.pop()


def candidates_for(row: dict, order_seed: int) -> list[dict]:
    candidates = [
        {
            "candidate_id": item["candidate_id"],
            "bid": float(item["bid"]),
            "train_p": float(item["win_probability"]),
            "test_p": float(item["win_probability"]),
            "train_u": float(item["utility"]),
            "test_u": float(item["utility"]),
        }
        for item in row["candidates"]
    ]
    random.Random(stable_int(row["scenario_id"], order_seed, "option-order")).shuffle(candidates)
    return candidates


def driver_version() -> str | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        value = pynvml.nvmlSystemGetDriverVersion()
        return value.decode() if isinstance(value, bytes) else str(value)
    except Exception:
        return None


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def complete(row: dict, budget: int) -> bool:
    return (
        row.get("readout_method") == "constrained_option_letter_next_token"
        and row.get("selected_bid") is not None
        and row.get("utility_ratio") is not None
        and (budget == 0 or int(row.get("realized_reasoning_tokens", -1)) == budget)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--budgets", default="0,64,128,256,384,512,768")
    parser.add_argument("--option-order-seeds", default="11,23,37")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    spec = next((item for item in config["models"] if item["slug"] == args.model), None)
    if spec is None:
        raise SystemExit(f"Unknown model {args.model!r}")
    budgets = [int(item) for item in args.budgets.split(",")]
    order_seeds = [int(item) for item in args.option_order_seeds.split(",")]
    scenarios, benchmark_hash, benchmark_version = load_benchmark(args.data_dir)
    if args.limit:
        scenarios = scenarios[: args.limit]
    expected = {(row["scenario_id"], budget, seed) for row in scenarios for budget in budgets for seed in order_seeds}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output = raw_dir / f"synthetic_{args.model}_deterministic.jsonl"
    run_manifest_path = raw_dir / f"synthetic_{args.model}_run_manifest.json"
    existing, corrupt = load_rows(output)
    if corrupt:
        atomic_json(output.with_suffix(".corrupt.json"), corrupt)
    existing = compact(output, existing, force=bool(corrupt))
    completed = {
        (row["scenario_id"], int(row["reasoning_budget"]), int(row["option_order_seed"]))
        for row in existing
        if complete(row, int(row.get("reasoning_budget", -1)))
    }
    invocation = {
        "model_slug": args.model,
        "requested_model_id": spec["hf_id"],
        "requested_model_revision": spec["revision"],
        "requested_tokenizer_revision": spec.get("tokenizer_revision", spec["revision"]),
        "benchmark_version": benchmark_version,
        "benchmark_sha256": benchmark_hash,
        "budgets": budgets,
        "option_order_seeds": order_seeds,
        "expected_cells": len(expected),
        "scenario_limit": args.limit,
        "gpu_hour_usd": config["cost"].get("gpu_hour_usd"),
        "gpu_count_billed": config["cost"].get("gpu_count_billed", 1),
        "hardware_name": config["cost"].get("hardware_name"),
    }
    if run_manifest_path.exists():
        recorded = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        mismatches = {key: (recorded.get(key), value) for key, value in invocation.items() if recorded.get(key) != value}
        if mismatches:
            raise RuntimeError(f"Existing run manifest differs from invocation: {mismatches}")
        run_manifest = recorded
        run_manifest.setdefault("resume_events_utc", []).append(dt.datetime.now(dt.timezone.utc).isoformat())
    else:
        run_manifest = {
            **invocation,
            "status": "initialized_before_model_load",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config_sha256": sha256(args.config),
            "runner_sha256": sha256(Path(__file__)),
            "repository_git_revision": git_revision(Path(__file__).resolve().parents[2]),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_build_version": torch.version.cuda,
            "transformers_version": importlib.metadata.version("transformers"),
            "accelerate_version": importlib.metadata.version("accelerate"),
        }
    atomic_json(run_manifest_path, run_manifest)
    if expected <= completed:
        run_manifest.update(status="already_complete_no_model_loaded", completed_cells=len(expected))
        atomic_json(run_manifest_path, run_manifest)
        print(f"All {len(expected)} cells complete; model load skipped")
        return

    harness = ModelHarness(spec, config)
    model_revision = harness.model_revision
    tokenizer_revision = harness.tokenizer_revision
    if model_revision != spec["revision"] or tokenizer_revision != spec.get("tokenizer_revision", spec["revision"]):
        raise RuntimeError(
            f"Resolved revisions differ: model={model_revision}, tokenizer={tokenizer_revision}, requested={spec['revision']}"
        )
    gpu_devices = [
        {"index": index, "name": torch.cuda.get_device_name(index), "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory}
        for index in range(torch.cuda.device_count())
    ]
    run_manifest.update(
        status="model_loaded_before_inference",
        resolved_model_id=getattr(harness.model.config, "_name_or_path", spec["hf_id"]),
        resolved_model_revision=model_revision,
        resolved_tokenizer_revision=tokenizer_revision,
        reasoning_protocol=spec.get("reasoning_protocol"),
        reasoning_protocol_matched=bool(spec.get("reasoning_protocol_matched", True)),
        chat_template_argument=spec.get("chat_template_argument"),
        dtype=str(next(harness.model.parameters()).dtype),
        device_map=spec.get("device_map", "auto"),
        resolved_device_map=getattr(harness.model, "hf_device_map", None),
        gpu_devices=gpu_devices,
        gpu_driver=driver_version(),
    )
    atomic_json(run_manifest_path, run_manifest)
    progress = tqdm(total=len(expected), initial=len(expected & completed), desc=f"{args.model}:synthetic")

    for scenario in scenarios:
        for budget in budgets:
            for order_seed in order_seeds:
                key = (scenario["scenario_id"], budget, order_seed)
                if key in completed:
                    continue
                displayed = candidates_for(scenario, order_seed)
                prompt = auction_prompt(float(scenario["private_value"]), displayed)
                generation_seed = stable_int(config["seed"], args.model, "synthetic", *key)
                try:
                    generated = harness.auction_decision(prompt, budget, generation_seed, "reasoning", sample=False)
                    freeform_text = generated["final_text"]
                    freeform_bid = parse_final_bid(freeform_text, displayed)
                    constrained = harness.constrained_bid_readout(prompt, generated["reasoning_text"], displayed, "reasoning")
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    run_manifest.update(status="cuda_oom", failed_cell=list(key), error=str(exc))
                    atomic_json(run_manifest_path, run_manifest)
                    raise RuntimeError(f"CUDA OOM at cell {key}; saved work is resumable") from exc
                selected_bid = constrained["chosen_bid"]
                scores = score_bid(selected_bid, displayed)
                freeform_valid = freeform_bid is not None
                probability = constrained["option_probability"]
                meter_parts = [generated["meter"], constrained["meter"]]
                finite_energy = [part.energy_joules for part in meter_parts if math.isfinite(part.energy_joules)]
                row = {
                    "record_schema_version": "auction-reasoning-cell-v2",
                    "benchmark_name": "synthetic_ipv_first_price",
                    "benchmark_version": benchmark_version,
                    "benchmark_sha256": benchmark_hash,
                    "scoring_version": "synthetic-ipv-exact-discrete-v1",
                    "model_slug": args.model,
                    "requested_model_id": spec["hf_id"],
                    "resolved_model_id": getattr(harness.model.config, "_name_or_path", spec["hf_id"]),
                    "requested_model_revision": spec["revision"],
                    "resolved_model_revision": model_revision,
                    "requested_tokenizer_revision": spec.get("tokenizer_revision", spec["revision"]),
                    "resolved_tokenizer_revision": tokenizer_revision,
                    "reasoning_protocol": spec.get("reasoning_protocol"),
                    "reasoning_protocol_matched": bool(spec.get("reasoning_protocol_matched", True)),
                    "scenario_id": scenario["scenario_id"],
                    "private_value": float(scenario["private_value"]),
                    "candidate_style": scenario["candidate_style"],
                    "option_count": int(scenario["option_count"]),
                    "canonical_candidate_set": scenario["candidates"],
                    "displayed_candidates": [
                        {"display_label": chr(65 + index), "candidate_id": item["candidate_id"], "bid": item["bid"], "win_probability": item["train_p"]}
                        for index, item in enumerate(displayed)
                    ],
                    "reasoning_budget": budget,
                    "realized_reasoning_tokens": int(generated["reasoning_tokens"]),
                    "answer_tokens": int(generated.get("answer_tokens", 0)),
                    "decoding": {"do_sample": False, "temperature": None, "top_p": None, "use_cache": True},
                    "generation_seed": generation_seed,
                    "option_order_seed": order_seed,
                    "raw_reasoning": generated["reasoning_text"],
                    "freeform_response": freeform_text,
                    "freeform_parsed_bid": freeform_bid,
                    "freeform_valid": freeform_valid,
                    "constrained_response": f"FINAL OPTION: {constrained['option_label']}",
                    "selected_option_label": constrained["option_label"],
                    "selected_bid": selected_bid,
                    "constrained_freeform_agreement": bool(freeform_valid and math.isclose(freeform_bid, selected_bid)),
                    "answer_log_probability": math.log(max(probability, 1e-300)),
                    "constrained_answer_probability": probability,
                    "constrained_option_probabilities": constrained["option_probabilities"],
                    "readout_method": "constrained_option_letter_next_token",
                    "utility": scores["test_utility"],
                    "optimal_bid": scores["test_optimal_bid"],
                    "optimal_utility": scores["test_optimal_utility"],
                    "utility_ratio": scores["test_utility_ratio"],
                    "regret": scores["test_regret"],
                    "exact_optimality": scores["test_correct"],
                    "python_version": platform.python_version(),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "transformers_version": importlib.metadata.version("transformers"),
                    "gpu_devices": gpu_devices,
                    "gpu_driver": driver_version(),
                    "hardware_name": config["cost"].get("hardware_name"),
                    "gpu_hour_usd": config["cost"].get("gpu_hour_usd"),
                    "gpu_count_billed": config["cost"].get("gpu_count_billed", 1),
                    "total_pipeline_elapsed_seconds": sum(part.seconds for part in meter_parts),
                    "total_pipeline_energy_joules": sum(finite_energy) if finite_energy else math.nan,
                    "total_pipeline_peak_memory_bytes": max(part.peak_memory_bytes for part in meter_parts),
                    "dtype": str(next(harness.model.parameters()).dtype),
                    "device_map": spec.get("device_map", "auto"),
                    "resolved_device_map": getattr(harness.model, "hf_device_map", None),
                    **meter_fields(generated["meter"], config),
                    **{f"constrained_readout_{name}": value for name, value in meter_fields(constrained["meter"], config).items()},
                }
                append_jsonl(output, row)
                completed.add(key)
                progress.update(1)
                run_manifest.update(status="running", completed_cells=len(expected & completed), last_completed_cell=list(key))
                atomic_json(run_manifest_path, run_manifest)
                print(f"{args.model}: {len(expected & completed)}/{len(expected)} {key} q={row['utility_ratio']:.4f}", flush=True)

    progress.close()
    final, corrupt = load_rows(output)
    final = compact(output, final, force=bool(corrupt))
    final_keys = {
        (row["scenario_id"], int(row["reasoning_budget"]), int(row["option_order_seed"]))
        for row in final if complete(row, int(row.get("reasoning_budget", -1)))
    }
    missing = sorted(expected - final_keys)
    run_manifest.update(
        status="complete" if not missing and not corrupt else "incomplete_requires_resume",
        completed_cells=len(expected & final_keys),
        missing_cells=missing[:100],
        corrupt_rows=len(corrupt),
        completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat() if not missing and not corrupt else None,
    )
    atomic_json(run_manifest_path, run_manifest)
    if missing or corrupt:
        raise RuntimeError(f"Run incomplete: missing={len(missing)}, corrupt={len(corrupt)}")


if __name__ == "__main__":
    main()
