#!/usr/bin/env python3
"""Validate the canonical synthetic IPV benchmark and its checksum."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import Counter
from pathlib import Path

from generate_scenarios import SCENARIO_COUNT, VERSION, generate, win_probability


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root: Path, check_regeneration: bool = True) -> dict:
    manifest = root / "synthetic_ipv_manifest.jsonl"
    expected_hash = (root / "manifest.sha256").read_text(encoding="ascii").split()[0]
    actual_hash = digest(manifest)
    errors: list[str] = []
    if actual_hash != expected_hash:
        errors.append(f"checksum mismatch: expected {expected_hash}, got {actual_hash}")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    ids = [row.get("scenario_id") for row in rows]
    if len(rows) != SCENARIO_COUNT:
        errors.append(f"expected {SCENARIO_COUNT} rows, got {len(rows)}")
    if len(set(ids)) != len(ids):
        errors.append("scenario IDs are not unique")
    factors = Counter()
    for row in rows:
        sid = row.get("scenario_id", "<missing>")
        if row.get("benchmark_version") != VERSION:
            errors.append(f"{sid}: version mismatch")
        k = row.get("option_count")
        style = row.get("candidate_style")
        factors[(k, style)] += 1
        if k not in {5, 7} or style not in {"structured", "unstructured"}:
            errors.append(f"{sid}: invalid K/style")
        candidates = row.get("candidates", [])
        if len(candidates) != k:
            errors.append(f"{sid}: candidate count mismatch")
            continue
        value = row.get("private_value")
        bids = [item.get("bid") for item in candidates]
        if len(set(bids)) != len(bids) or any(not isinstance(b, int) or b < 0 or b > value for b in bids):
            errors.append(f"{sid}: invalid or duplicate bid")
        if row.get("candidate_generation_optimum_used") is not False:
            errors.append(f"{sid}: candidate generation provenance flag invalid")
        for position, item in enumerate(candidates):
            expected_label = chr(65 + position)
            if item.get("canonical_label") != expected_label or item.get("candidate_id") != f"c{position}":
                errors.append(f"{sid}: option-label mapping is inconsistent")
            p = item.get("win_probability")
            u = item.get("utility")
            expected_p = win_probability(item["bid"])
            expected_u = (value - item["bid"]) * expected_p
            if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
                errors.append(f"{sid}: invalid probability")
            elif not math.isclose(p, expected_p, abs_tol=1e-15):
                errors.append(f"{sid}: probability mismatch for bid {item['bid']}")
            if not isinstance(u, (int, float)) or not math.isfinite(u) or not math.isclose(u, expected_u, abs_tol=1e-12):
                errors.append(f"{sid}: utility mismatch for bid {item['bid']}")
        optimum = max(candidates, key=lambda item: (item["utility"], -item["bid"]))
        if optimum["utility"] <= 0:
            errors.append(f"{sid}: non-positive utility-ratio denominator")
        for key, expected in (
            ("optimal_candidate_id", optimum["candidate_id"]),
            ("optimal_bid", optimum["bid"]),
            ("optimal_utility", optimum["utility"]),
        ):
            value_found = row.get(key)
            if isinstance(expected, float):
                if not math.isclose(value_found, expected, abs_tol=1e-12):
                    errors.append(f"{sid}: {key} mismatch")
            elif value_found != expected:
                errors.append(f"{sid}: {key} mismatch")
    expected_factors = {(5, "structured"): 54, (5, "unstructured"): 54, (7, "structured"): 54, (7, "unstructured"): 54}
    if factors != expected_factors:
        errors.append(f"factor balance mismatch: {dict(factors)}")
    regeneration_hash = None
    if check_regeneration:
        with tempfile.TemporaryDirectory() as temporary:
            regenerated, regeneration_hash = generate(Path(temporary))
            if regenerated.read_bytes() != manifest.read_bytes():
                errors.append("deterministic regeneration is not byte-identical")
    report = {
        "valid": not errors,
        "benchmark_version": VERSION,
        "scenario_count": len(rows),
        "unique_scenario_ids": len(set(ids)),
        "manifest_sha256": actual_hash,
        "regenerated_sha256": regeneration_hash,
        "factor_counts": {f"K{k}_{style}": count for (k, style), count in sorted(factors.items())},
        "errors": errors,
    }
    if errors:
        raise SystemExit(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--skip-regeneration", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.benchmark_dir, not args.skip_regeneration), indent=2))


if __name__ == "__main__":
    main()
