#!/usr/bin/env python3
"""Combine saved existing-four and newly generated deterministic frontiers."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
LABELS={"phi4_mini_reasoning":"Phi-4-mini-reasoning","qwen3_4b":"Qwen3-4B","deepseek_r1_llama_8b":"DeepSeek-R1-Distill-Llama-8B","qwen3_8b":"Qwen3-8B","qwen3_14b":"Qwen3-14B","granite_3_2_8b_instruct":"Granite-3.2-8B-Instruct"}
def main():
 p=argparse.ArgumentParser();p.add_argument("--package-root",type=Path,required=True);a=p.parse_args();root=a.package_root;saved=root/"data/existing4_results"
 for line in (saved/"SHA256SUMS").read_text().splitlines():
  expected,name=line.split(maxsplit=1)
  if hashlib.sha256((saved/name.strip()).read_bytes()).hexdigest()!=expected:raise RuntimeError(f"Saved-result checksum mismatch: {name}")
 paths=list(saved.glob("fixed_*_reasoning_greedy.jsonl"))+list((root/"results/historical/raw").glob("fixed_*_reasoning_greedy.jsonl"));rows=[]
 for path in paths: rows += [json.loads(x) for x in path.read_text().splitlines() if x]
 d=pd.DataFrame(rows).drop_duplicates(["model","scenario_id","budget","repeat_seed"],keep="last");d=d[d.budget.isin([0,64,128,256,512,768])]
 if set(d.model)!=set(LABELS):raise RuntimeError(f"Expected six models, found {sorted(d.model.unique())}")
 f=d.groupby(["model","budget"],as_index=False).agg(q=("test_utility_ratio","mean"),regret=("test_regret","mean"),exact_optimality=("test_correct","mean"));f["model_label"]=f.model.map(LABELS);out=root/"paper_ready";out.mkdir(parents=True,exist_ok=True);f.to_csv(out/"updated_six_model_deterministic_frontier.csv",index=False)
 fig,ax=plt.subplots(figsize=(6.4,4.0))
 for model in LABELS:
  x=f[f.model==model].sort_values("budget");ax.plot(x.budget,x.q,marker="o",linewidth=1.4,label=LABELS[model])
 ax.set(xlabel="Assigned reasoning tokens",ylabel="Mean held-out utility ratio");ax.grid(alpha=.25);ax.legend(frameon=False,fontsize=7,ncol=2);fig.tight_layout()
 for ext in ["pdf","png"]:fig.savefig(out/f"updated_six_model_deterministic_frontier.{ext}",dpi=300)
 plt.close(fig);print(f.to_string(index=False))
if __name__=="__main__":main()
