#!/usr/bin/env python3
"""Build expected experiment-cell manifests from the checked benchmark."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data-dir",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--models",required=True); a=p.parse_args()
    scenarios=[json.loads(x) for x in (a.data_dir/"synthetic_ipv_manifest.jsonl").read_text().splitlines() if x]
    rows=[{"model":m,"scenario_id":s["scenario_id"],"reasoning_budget":b,"option_order_seed":o} for m in a.models.split(",") for s in scenarios for b in [0,64,128,256,384,512,768] for o in [11,23,37]]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w") as f:
        for row in rows: f.write(json.dumps(row,separators=(",",":"))+"\n")
    print(json.dumps({"rows":len(rows),"per_model":len(scenarios)*7*3,"models":a.models.split(",")},indent=2))
if __name__=="__main__": main()
