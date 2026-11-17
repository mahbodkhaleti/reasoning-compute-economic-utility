#!/usr/bin/env python3
"""Compare new traces with the paper's saved traces (CPU only, no model needed).

drift768: new forced-768 traces must equal the paper's forced-768 traces token
for token, and the readout must pick the same option.

run1 (smoke): the paper's unforced run used a 1,024-token cap. Greedy decoding
is prefix-consistent, so a new 4,096-cap trace must equal the paper's trace when
the paper's finished, and must start with it when the paper's hit 1,024.

Usage:
  python followup/compare_drift.py results/followup/smoke/drift768
  python followup/compare_drift.py results/followup/smoke/run1_nat4096
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference" / "paper_first10"


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def first_diff(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def main() -> None:
    run_dir = Path(sys.argv[1])
    unforced = "run1" in run_dir.name
    ok_all = True
    for new_file in sorted(run_dir.glob("*.jsonl")):
        model = new_file.stem
        ref = {(r["scenario_id"], int(r["repeat_seed"])): r for r in load(REF / f"{model}.jsonl")
               if int(r["budget"]) == (-1 if unforced else 768)}
        rows = load(new_file)
        n = same_text = same_choice = 0
        notes = []
        for r in rows:
            p = ref.get((r["scenario_id"], int(r["repeat_seed"])))
            if p is None:
                continue
            n += 1
            new, old = r["reasoning_text"], p["reasoning_text"]
            if unforced:
                if int(p["reasoning_tokens"]) >= 1024:
                    match = new.startswith(old[:-8]) if len(old) > 8 else new.startswith(old)
                else:
                    match = new == old
            else:
                match = new == old
            same_text += match
            if not unforced:
                same_choice += abs(float(r["chosen_bid"]) - float(p["chosen_bid"])) < 1e-6
            if not match:
                i = first_diff(new, old)
                notes.append(f"  {r['scenario_id']}: first difference at character {i} of {len(old)}")
        line = f"{model}: {same_text}/{n} traces identical to the paper"
        if not unforced:
            line += f", {same_choice}/{n} same chosen option"
        print(line)
        for x in notes[:5]:
            print(x)
        ok_all &= n > 0 and same_text == n
    print("RESULT:", "no drift" if ok_all else "DRIFT FOUND (send me this output)")


if __name__ == "__main__":
    main()
