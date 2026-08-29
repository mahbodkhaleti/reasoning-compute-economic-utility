#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
def read(path): return [json.loads(x) for x in path.read_text().splitlines() if x]
def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--model",required=True);a=p.parse_args()
    h=read(a.root/f"historical/raw/fixed_{a.model}_reasoning_greedy.jsonl");s=read(a.root/f"synthetic/raw/synthetic_{a.model}_deterministic.jsonl")
    assert len(h)==1 and len(s)==1
    for row in h+s:
        assert row.get("readout_method")=="constrained_option_letter_next_token"
        assert row.get("resolved_model_revision")==row.get("requested_model_revision")
    assert h[0].get("reasoning_tokens")==64 and s[0].get("realized_reasoning_tokens")==64
    assert h[0].get("freeform_final_text") is not None and s[0].get("freeform_response") is not None
    assert h[0].get("test_utility_ratio") is not None and s[0].get("utility_ratio") is not None
    print(json.dumps({"model":a.model,"historical_rows":1,"synthetic_rows":1,"valid":True},indent=2))
if __name__=="__main__":main()
