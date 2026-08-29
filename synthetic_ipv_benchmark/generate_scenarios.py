#!/usr/bin/env python3
"""Generate the canonical synthetic IPV first-price decision benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from fractions import Fraction
from pathlib import Path

VERSION = "synthetic-ipv-fpsb-v1.0.0"
SEED = 20260824
SCENARIO_COUNT = 216
V_MAX = 99
N_BIDDERS = 3
STYLES = ((5, "structured"), (5, "unstructured"), (7, "structured"), (7, "unstructured"))


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**31 - 1)


def win_probability(bid: int) -> float:
    """Exact probability with integer values and uniform tie-breaking.

    Opponents draw v in {0,...,99} and bid 2v/3. Comparisons use integer
    scores 3*b and 2*v, so no floating-point equality decision is involved.
    """
    total = Fraction(0, 1)
    focal = 3 * bid
    for v1 in range(V_MAX + 1):
        for v2 in range(V_MAX + 1):
            scores = (focal, 2 * v1, 2 * v2)
            top = max(scores)
            if focal == top:
                total += Fraction(1, sum(score == top for score in scores))
    return float(total / ((V_MAX + 1) ** 2))


WIN_PROBABILITIES = {bid: win_probability(bid) for bid in range(V_MAX + 1)}


def candidate_bids(value: int, count: int, style: str, seed: int) -> list[int]:
    rng = random.Random(seed)
    feasible = list(range(value + 1))
    if style == "unstructured":
        return sorted(rng.sample(feasible, count))
    max_step = max(1, value // (count - 1))
    step = rng.randint(1, min(5, max_step))
    max_start = value - step * (count - 1)
    start = rng.randint(0, max_start)
    return [start + step * index for index in range(count)]


def make_scenario(index: int) -> dict:
    count, style = STYLES[index % len(STYLES)]
    # A deterministic stratification covers the support while assigning every
    # adjacent block of four values to all K/style combinations.
    value = 8 + round((index + 0.5) * (V_MAX - 8) / SCENARIO_COUNT)
    attempt = 0
    while True:
        cseed = stable_seed(VERSION, SEED, index, count, style, attempt)
        bids = candidate_bids(value, count, style, cseed)
        candidates = [
            {
                "candidate_id": f"c{position}",
                "canonical_label": chr(65 + position),
                "bid": bid,
                "win_probability": WIN_PROBABILITIES[bid],
                "utility": (value - bid) * WIN_PROBABILITIES[bid],
            }
            for position, bid in enumerate(bids)
        ]
        optimum = max(candidates, key=lambda item: (item["utility"], -item["bid"]))
        if optimum["utility"] > 0:
            break
        attempt += 1
    return {
        "benchmark_name": "Synthetic IPV first-price auction robustness benchmark",
        "benchmark_version": VERSION,
        "scenario_id": f"syn-ipv-{index + 1:04d}",
        "environment": {
            "value_distribution": "discrete_uniform_integers",
            "value_support": [0, V_MAX],
            "n_bidders": N_BIDDERS,
            "opponent_count": N_BIDDERS - 1,
            "opponent_strategy": "b(v)=2v/3",
            "auction_format": "first_price_sealed_bid",
            "tie_rule": "uniform_random_among_highest_bidders",
        },
        "private_value": value,
        "option_count": count,
        "candidate_style": style,
        "candidate_seed": cseed,
        "generation_attempt": attempt,
        "candidate_generation_optimum_used": False,
        "candidates": candidates,
        "optimal_candidate_id": optimum["candidate_id"],
        "optimal_bid": optimum["bid"],
        "optimal_utility": optimum["utility"],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(output_dir: Path) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "synthetic_ipv_manifest.jsonl"
    rows = [make_scenario(index) for index in range(SCENARIO_COUNT)]
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    digest = sha256(manifest)
    (output_dir / "manifest.sha256").write_text(
        f"{digest}  {manifest.name}\n", encoding="ascii"
    )
    factor_counts = Counter((row["option_count"], row["candidate_style"]) for row in rows)
    metadata = {
        "benchmark_version": VERSION,
        "generation_seed": SEED,
        "scenario_count": len(rows),
        "manifest_file": manifest.name,
        "manifest_sha256": digest,
        "factor_counts": {f"K{k}_{style}": count for (k, style), count in sorted(factor_counts.items())},
        "interpretation": "Inspired robustness benchmark; not an exact replication of Zhu et al.",
        "probability_method": "Exact enumeration over two discrete-uniform opponent values with uniform tie-breaking.",
    }
    (output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    manifest, digest = generate(args.output_dir)
    print(json.dumps({"manifest": str(manifest), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
