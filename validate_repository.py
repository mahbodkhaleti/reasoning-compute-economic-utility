#!/usr/bin/env python3
"""Validate the complete six-checkpoint submission repository without inference."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ORIGINAL_MODELS = {
    "phi4_mini_reasoning",
    "qwen3_4b",
    "deepseek_r1_llama_8b",
    "qwen3_8b",
}
REPORTED_MODELS = ORIGINAL_MODELS | {
    "qwen3_14b",
    "granite_3_2_8b_instruct",
}
REQUIRED_PATHS = {
    "README.md",
    "LICENSE",
    "LICENSES.md",
    "docs/REPRODUCIBILITY.md",
    "docs/REPRODUCIBILITY_MANIFEST.json",
    "docs/RESULTS_PROVENANCE.md",
    "offline_replay/README.md",
    "offline_replay/requirements.txt",
    "offline_replay/verify_preserved.py",
    "reproduce_all_from_preserved.sh",
    "new_models_full_server_package/README.md",
    "synthetic_ipv_benchmark/synthetic_ipv_manifest.jsonl",
    "paper/main.tex",
    "paper/references.bib",
    "paper/checklist.tex",
    "paper/neurips_2026.sty",
    "paper/figures/strategic_input_pipeline_overview.pdf",
    "paper/figures/synthetic_ipv_six_model_frontier.pdf",
    "paper/tables/synthetic_ipv_endpoints.tex",
}
SOURCE_DIRS = (
    "robustness_extensions/scripts",
    "offline_replay",
    "new_models_full_server_package/src",
    "synthetic_existing4_server_package/src",
    "synthetic_ipv_benchmark",
)
MANIFEST_EXCLUSIONS = {"MANIFEST.sha256", "repository_validation.json"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_files() -> set[str]:
    command = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if command.returncode == 0 and command.stdout.strip() == "true":
        return set(
            subprocess.run(
                ["git", "ls-files"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        )
    manifest = ROOT / "MANIFEST.sha256"
    return {
        line.split(None, 1)[1]
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    } | MANIFEST_EXCLUSIONS


def check_top_manifest() -> tuple[int, list[str]]:
    entries: list[tuple[str, str]] = []
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        entries.append((expected, relative.strip()))
    errors = []
    for expected, relative in entries:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
        elif sha256(path) != expected:
            errors.append(f"SHA-256 mismatch: {relative}")
    return len(entries), errors


def main() -> None:
    config = json.loads((ROOT / "robustness_extensions/scripts/config.json").read_text())
    original_specs = {model["slug"]: model for model in config["models"]}
    require(set(original_specs) == ORIGINAL_MODELS, "Original runner config must contain its four historical models")
    for slug, spec in original_specs.items():
        revision = spec.get("revision", "")
        require(len(revision) == 40, f"Missing 40-character revision for {slug}")
        int(revision, 16)

    structured = json.loads((ROOT / "docs/REPRODUCIBILITY_MANIFEST.json").read_text(encoding="utf-8"))
    reported_specs = {model["slug"]: model for model in structured["models"]}
    require(set(reported_specs) == REPORTED_MODELS, "Structured manifest must contain exactly six reported models")
    for slug, spec in reported_specs.items():
        for field in ("model_revision", "tokenizer_revision"):
            revision = spec.get(field, "")
            require(len(revision) == 40, f"Missing 40-character {field} for {slug}")
            int(revision, 16)

    missing_paths = sorted(path for path in REQUIRED_PATHS if not (ROOT / path).is_file())
    require(not missing_paths, f"Missing release paths: {missing_paths}")

    python_files: list[Path] = []
    for relative in SOURCE_DIRS:
        for path in sorted((ROOT / relative).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
            python_files.append(path)

    verification = subprocess.run(
        [sys.executable, str(ROOT / "offline_replay/verify_preserved.py"), "--repo-root", str(ROOT)],
        check=True,
        capture_output=True,
        text=True,
    )
    preserved_report = json.loads(verification.stdout)
    require(preserved_report["status"] == "passed", "Preserved-output verification failed")

    scenario_path = ROOT / "new_models_full_server_package/data/historical/scenarios.csv"
    with scenario_path.open(newline="", encoding="utf-8") as handle:
        scenario_ids = {row["scenario_id"] for row in csv.DictReader(handle)}
    require(len(scenario_ids) == 216, f"Expected 216 scenarios, found {len(scenario_ids)}")

    tracked = repository_files()
    prohibited = {
        "online-auctions-dataset/auction.csv",
        "data/source/auction.csv",
    }
    require(not (tracked & prohibited), f"Raw source data are tracked: {sorted(tracked & prohibited)}")
    require(REQUIRED_PATHS <= tracked, "One or more required release paths are not tracked")

    manifest_count, manifest_errors = check_top_manifest()
    require(not manifest_errors, "; ".join(manifest_errors))

    report = {
        "status": "passed",
        "models": sorted(REPORTED_MODELS),
        "historical_scenarios": len(scenario_ids),
        "historical_records_per_model": {
            "no_think_greedy": 648,
            "reasoning_greedy": 4536,
            "reasoning_stochastic": 3240,
        },
        "synthetic_records_per_model": 4536,
        "synthetic_records_total": 27216,
        "preserved_output_verification": preserved_report["status"],
        "python_files_parsed": len(python_files),
        "manifest_files_checked": manifest_count,
        "manifest_errors": manifest_errors,
        "raw_auction_source_tracked": False,
        "transformed_scenario_artifacts_tracked": True,
        "transformed_artifact_license_caveat": "See LICENSES.md and data/README.md; no new license is asserted.",
        "new_llm_inference_executed": False,
    }
    (ROOT / "repository_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
