#!/usr/bin/env python3
"""Derive expected historical inference cells from the checked scenario file."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
def main():
    p=argparse.ArgumentParser();p.add_argument("--scenarios",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--models",required=True);a=p.parse_args()
    ids=[str(x).zfill(12) for x in pd.read_csv(a.scenarios,dtype={"scenario_id":str}).scenario_id];rows=[]
    for model in a.models.split(","):
        for sid in ids:
            for b in [0,64,128,256,512,768]:
                for seed in [11,23,37]:rows.append({"model":model,"phase":"deterministic","scenario_id":sid,"budget":b,"repeat_seed":seed,"option_order_seed":seed})
            for seed in [11,23,37]:rows.append({"model":model,"phase":"unconstrained","scenario_id":sid,"budget":-1,"repeat_seed":seed,"option_order_seed":seed})
            for seed in [11,23,37]:rows.append({"model":model,"phase":"no_think","scenario_id":sid,"budget":0,"repeat_seed":seed,"option_order_seed":seed})
            for b in [128,384,768]:
                for seed in [1,7,21,42,99]:rows.append({"model":model,"phase":"stochastic","scenario_id":sid,"budget":b,"repeat_seed":seed,"option_order_seed":314159})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w") as f:
        for row in rows:f.write(json.dumps(row,separators=(",",":"))+"\n")
    print(json.dumps({"scenarios":len(ids),"models":a.models.split(","),"rows":len(rows),"per_model":len(rows)//len(a.models.split(","))},indent=2))
if __name__=="__main__":main()
