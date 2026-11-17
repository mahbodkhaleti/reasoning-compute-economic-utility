#!/usr/bin/env python3
"""Quick per-model table for a results folder (CPU only). The full analysis
(gap recovered with family-bootstrap CIs, Table 2 split, break-even) is done
afterwards with the paper's analysis code."""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path


def main() -> None:
    run_dir = Path(sys.argv[1])
    print(f"{'model':24s} {'setting':13s} {'n':>4s} {'q':>6s} {'at cap':>7s} {'median tok':>10s} {'mean tok':>8s} "
          f"{'s/decision':>10s} {'J/decision':>10s} {'est. full h':>11s}")
    for f in sorted(run_dir.glob("*.jsonl")):
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        for setting in sorted({r["setting"] for r in rows}):
            rs = [r for r in rows if r["setting"] == setting]
            q = st.mean(r["test_utility_ratio"] for r in rs)
            cap = sum(r["hit_cap"] for r in rs) / len(rs)
            tok = [r["reasoning_tokens"] for r in rs]
            sec = st.mean(r["wall_seconds"] for r in rs)
            en = [r["energy_joules"] for r in rs if r["energy_joules"] == r["energy_joules"]]
            print(f"{f.stem:24s} {setting:13s} {len(rs):4d} {q:6.3f} {cap:7.1%} {st.median(tok):10.0f} {st.mean(tok):8.0f} "
                  f"{sec:10.1f} {(st.mean(en) if en else float('nan')):10.0f} {sec * 648 / 3600:11.1f}"
                  + ("   <- over 10% at cap" if cap > 0.10 else ""))


if __name__ == "__main__":
    main()
