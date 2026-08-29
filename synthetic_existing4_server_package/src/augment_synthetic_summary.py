#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--paper-ready",type=Path,required=True);a=p.parse_args();ph=pd.read_csv(a.output_dir/"tables/synthetic_normalized_phase_grid.csv.gz");lines=["","## Strategic robustness","","This uses the deterministic synthetic frontier and is conditional on the specified normalized contest.",""]
 for model in sorted(ph.model.unique()):
  regions=sorted(ph[(ph.model==model)&np.isclose(ph.gamma,1.0)].phase.unique());raw=next((a.output_dir/"raw").glob(f"synthetic_{model}_deterministic.jsonl"));first=json.loads(next(x for x in raw.read_text().splitlines() if x));qual=" The reasoning protocol differs from the matched Qwen protocol, so this is a protocol sensitivity." if not first.get("reasoning_protocol_matched",True) else ""
  lines.append(f"- {model}: gamma=1 regions on the declared grid are {', '.join(regions)}.{qual}")
 summary=a.paper_ready/"RESULTS_SUMMARY.md";summary.write_text(summary.read_text()+"\n"+"\n".join(lines)+"\n")
if __name__=="__main__":main()
