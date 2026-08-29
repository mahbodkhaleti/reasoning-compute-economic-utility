#!/usr/bin/env python3
"""Validate Qwen3-14B inputs or completed outputs without loading a model."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


MODEL = "qwen3_14b"
HF_ID = "Qwen/Qwen3-14B"
REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
SCENARIOS = 216
DETERMINISTIC_BUDGETS = (0, 64, 128, 256, 512, 768, -1)
DETERMINISTIC_SEEDS = (11, 23, 37)
STOCHASTIC_BUDGETS = (128, 384, 768)
STOCHASTIC_SEEDS = (1, 7, 21, 42, 99)
STOCHASTIC_ORDER_SEED = 314159


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_rows(
    path: Path,
    budgets: tuple[int, ...],
    seeds: tuple[int, ...],
    errors: list[str],
    option_order_seed: int | None = None,
    require_confidence: bool = False,
) -> None:
    if not path.exists():
        errors.append(f"Missing {path.name}")
        return
    rows = read_jsonl(path)
    keys = [
        (str(row.get("scenario_id", "")).zfill(12), int(row["budget"]), int(row["repeat_seed"]))
        for row in rows
    ]
    expected = SCENARIOS * len(budgets) * len(seeds)
    require(len(rows) == expected, f"{path.name}: {len(rows)} rows, expected {expected}", errors)
    require(len(keys) == len(set(keys)), f"{path.name}: duplicate scenario/budget/seed keys", errors)
    require(set(row["budget"] for row in rows) == set(budgets), f"{path.name}: wrong budgets", errors)
    require(set(row["repeat_seed"] for row in rows) == set(seeds), f"{path.name}: wrong seeds", errors)
    require({row.get("model") for row in rows} == {MODEL}, f"{path.name}: wrong model slug", errors)
    require({row.get("hf_id") for row in rows} == {HF_ID}, f"{path.name}: wrong Hugging Face ID", errors)
    for field in ("requested_model_revision", "requested_tokenizer_revision", "resolved_model_revision", "resolved_tokenizer_revision"):
        require({row.get(field) for row in rows} == {REVISION}, f"{path.name}: {field} is not pinned to {REVISION}", errors)
    require(
        all(row.get("readout_method") == "constrained_option_letter_next_token" for row in rows),
        f"{path.name}: primary constrained readout is missing",
        errors,
    )
    require(all(bool(row.get("valid_output")) for row in rows), f"{path.name}: invalid constrained output", errors)
    require(
        all(math.isfinite(float(row.get("test_utility_ratio", math.nan))) for row in rows),
        f"{path.name}: non-finite held-out utility ratio",
        errors,
    )
    if option_order_seed is not None:
        require(
            {int(row.get("option_order_seed", -1)) for row in rows} == {option_order_seed},
            f"{path.name}: stochastic option order is not fixed at {option_order_seed}",
            errors,
        )
    if require_confidence:
        require(
            all(row.get("verbal_confidence") is not None for row in rows),
            f"{path.name}: missing verbal-confidence readout",
            errors,
        )
        require(
            all(math.isfinite(float(row.get("answer_geomean_probability", math.nan))) for row in rows),
            f"{path.name}: missing answer-token confidence",
            errors,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("inputs", "complete"), default="complete")
    args = parser.parse_args()
    output = Path(args.output_dir)
    errors: list[str] = []

    scenarios_path = output / "data" / "scenarios.csv"
    require(scenarios_path.exists(), "Missing data/scenarios.csv", errors)
    if scenarios_path.exists():
        scenarios = pd.read_csv(scenarios_path, dtype={"scenario_id": str})
        require(len(scenarios) == SCENARIOS, f"Expected {SCENARIOS} scenarios, found {len(scenarios)}", errors)
        require(scenarios["scenario_id"].nunique() == SCENARIOS, "Scenario IDs are not unique", errors)
        require(set(scenarios["option_count"]) == {5, 7}, "Expected both 5- and 7-option tasks", errors)
    require((output / "data" / "auction_splits.csv").exists(), "Missing data/auction_splits.csv", errors)

    if args.stage == "complete":
        raw = output / "raw"
        validate_rows(
            raw / "fixed_qwen3_14b_reasoning_greedy.jsonl",
            DETERMINISTIC_BUDGETS,
            DETERMINISTIC_SEEDS,
            errors,
        )
        validate_rows(
            raw / "fixed_qwen3_14b_no_think_greedy.jsonl",
            (0,),
            DETERMINISTIC_SEEDS,
            errors,
        )
        validate_rows(
            raw / "fixed_qwen3_14b_reasoning_stochastic.jsonl",
            STOCHASTIC_BUDGETS,
            STOCHASTIC_SEEDS,
            errors,
            option_order_seed=STOCHASTIC_ORDER_SEED,
            require_confidence=True,
        )
        for name in (
            "fixed_budget_summary.csv",
            "model_frontiers.csv",
            "paired_budget_comparisons.csv",
            "seed_robustness_summary.csv",
            "freeform_sensitivity_summary.csv",
            "confidence_calibration_summary.csv",
            "stochastic_game_uncertainty_summary.csv",
            "game_beta_sensitivity_submission.csv",
            "game_gamma_sensitivity_submission.csv",
            "compute_menu_robustness_submission.csv",
            "auction_cdf_bootstrap_summary.csv",
            "normalized_transition_boundaries.csv",
            "hybrid_zero_compute_frontier.csv",
            "hybrid_zero_compute_transition_thresholds.csv",
            "hybrid_zero_compute_bootstrap_summary.csv",
        ):
            require((output / "tables" / name).exists(), f"Missing analysis table {name}", errors)
        for suffix in ("reasoning_greedy", "no_think_greedy", "reasoning_stochastic"):
            manifest = raw / f"fixed_qwen3_14b_{suffix}_run_manifest.json"
            metadata = raw / f"fixed_qwen3_14b_{suffix}_metadata.json"
            require(manifest.exists(), f"Missing run manifest {manifest.name}", errors)
            require(metadata.exists(), f"Missing metadata {metadata.name}", errors)
            if manifest.exists():
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                require(payload.get("status") in {"complete", "already_complete_no_model_loaded"}, f"{manifest.name}: incomplete status", errors)
                require(payload.get("requested_model_revision") == REVISION, f"{manifest.name}: wrong requested revision", errors)
                require(payload.get("resolved_model_revision") == REVISION, f"{manifest.name}: wrong resolved revision", errors)
                require(payload.get("resolved_tokenizer_revision") == REVISION, f"{manifest.name}: wrong resolved tokenizer revision", errors)
                require(bool(payload.get("config_sha256")), f"{manifest.name}: missing config hash", errors)
                require(bool(payload.get("runner_sha256")), f"{manifest.name}: missing runner hash", errors)
                generation = payload.get("resolved_generation_config", {})
                require(generation.get("top_k") == 20, f"{manifest.name}: expected pinned Qwen top_k=20", errors)
            if metadata.exists():
                payload = json.loads(metadata.read_text(encoding="utf-8"))
                require(payload.get("model_revision") == REVISION, f"{metadata.name}: wrong model revision", errors)
                require(payload.get("tokenizer_revision") == REVISION, f"{metadata.name}: wrong tokenizer revision", errors)
                require(payload.get("revision_recorded_before_inference") is True, f"{metadata.name}: revision timing flag missing", errors)

    report = {
        "status": "PASS" if not errors else "FAIL",
        "stage": args.stage,
        "model": MODEL,
        "hf_id": HF_ID,
        "revision": REVISION,
        "validation_executed_llm_inference": False,
        "completed_outputs_require_prior_llm_inference": args.stage == "complete",
        "errors": errors,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / f"qwen3_14b_{args.stage}_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
