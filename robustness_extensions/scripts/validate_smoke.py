#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common import read_jsonl


def deduplicate(rows: list[dict]) -> list[dict]:
    by_key = {}
    for row in rows:
        key = (str(row["scenario_id"]), int(row["budget"]), int(row["repeat_seed"]))
        by_key[key] = row
    return list(by_key.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    raw = Path(args.output_dir) / "raw"
    paths = {
        "reasoning": raw / f"fixed_{args.model}_reasoning_greedy.jsonl",
        "no_think": raw / f"fixed_{args.model}_no_think_greedy.jsonl",
        "stochastic": raw / f"fixed_{args.model}_reasoning_stochastic.jsonl",
    }
    expected = {"reasoning": 8, "no_think": 2, "stochastic": 4}
    errors: list[str] = []
    report: dict[str, object] = {"model": args.model, "files": {}}
    groups: dict[str, list[dict]] = {}
    for name, path in paths.items():
        rows = deduplicate(read_jsonl(path))
        groups[name] = rows
        report["files"][name] = {"path": str(path), "unique_rows": len(rows)}
        if len(rows) != expected[name]:
            errors.append(f"{name}: found {len(rows)} unique rows, expected {expected[name]}")
        for row in rows:
            if not row.get("valid_output"):
                errors.append(
                    f"{name}: invalid bid at scenario={row.get('scenario_id')} "
                    f"budget={row.get('budget')} seed={row.get('repeat_seed')}"
                )
            if row.get("readout_method") != "constrained_option_letter_next_token":
                errors.append(f"{name}: constrained readout marker is missing")
            chosen = row.get("chosen_bid")
            if chosen is None or not any(
                abs(float(chosen) - float(bid)) < 1e-6 for bid in row.get("option_order", [])
            ):
                errors.append(f"{name}: chosen bid is absent from option order")
            ratio = float(row.get("test_utility_ratio", math.nan))
            regret = float(row.get("test_regret", math.nan))
            confidence = float(row.get("answer_geomean_probability", math.nan))
            if not math.isfinite(ratio) or not 0 <= ratio <= 1 + 1e-9:
                errors.append(f"{name}: invalid utility ratio {ratio}")
            if not math.isfinite(regret) or regret < -1e-9:
                errors.append(f"{name}: invalid regret {regret}")
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                errors.append(f"{name}: invalid constrained-answer confidence {confidence}")

    for row in groups.get("reasoning", []):
        budget = int(row["budget"])
        tokens = int(row["reasoning_tokens"])
        if budget >= 0 and tokens != budget:
            errors.append(f"reasoning: budget {budget} produced {tokens} reasoning tokens")
        if budget < 0 and not 0 < tokens <= 1024:
            errors.append(f"reasoning: unconstrained run produced {tokens} tokens")
    for row in groups.get("no_think", []):
        if int(row["budget"]) != 0 or int(row["reasoning_tokens"]) != 0:
            errors.append("no_think: smoke baseline is not a zero-reasoning decision")
    for row in groups.get("stochastic", []):
        verbal = row.get("verbal_confidence")
        if verbal is None or not math.isfinite(float(verbal)) or not 0 <= float(verbal) <= 1:
            errors.append(
                f"stochastic: unparseable verbal confidence at scenario={row.get('scenario_id')} "
                f"seed={row.get('repeat_seed')}: {row.get('verbal_confidence_text')!r}"
            )

    report["status"] = "PASS" if not errors else "FAIL"
    report["errors"] = errors
    report_path = Path(args.output_dir) / "smoke_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
