#!/usr/bin/env python3
"""Descriptive Qwen-family scale sensitivity; never reruns a checkpoint."""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS={"qwen3_4b":"Qwen3-4B","qwen3_8b":"Qwen3-8B","qwen3_14b":"Qwen3-14B"}
def verify_saved(root):
    for line in (root/"SHA256SUMS").read_text().splitlines():
        expected,name=line.split(maxsplit=1);actual=hashlib.sha256((root/name.strip()).read_bytes()).hexdigest()
        if actual!=expected:raise RuntimeError(f"Saved-result checksum mismatch: {name}")
def load(paths):
    auction=pd.read_csv(root/"data/historical/auction_splits.csv");prize=float(auction.loc[auction.split=="test","final_price"].mean());rows=[]
    for p in paths:
        rows += [json.loads(x) for x in p.read_text().splitlines() if x]
    return pd.DataFrame(rows)
def main():
    p=argparse.ArgumentParser();p.add_argument("--package-root",type=Path,required=True);a=p.parse_args();root=a.package_root;old=root/"data/existing4_results";new=root/"results/historical/raw";out=root/"paper_ready";out.mkdir(parents=True,exist_ok=True)
    verify_saved(old);greedy=load([old/"fixed_qwen3_4b_reasoning_greedy.jsonl",old/"fixed_qwen3_8b_reasoning_greedy.jsonl",new/"fixed_qwen3_14b_reasoning_greedy.jsonl"]);sto=load([old/"fixed_qwen3_4b_reasoning_stochastic.jsonl",old/"fixed_qwen3_8b_reasoning_stochastic.jsonl",new/"fixed_qwen3_14b_reasoning_stochastic.jsonl"])
    greedy=greedy.drop_duplicates(["model","scenario_id","budget","repeat_seed"],keep="last");sto=sto.drop_duplicates(["model","scenario_id","budget","repeat_seed"],keep="last")
    f=greedy[greedy.budget.isin([0,64,128,256,512,768])].groupby(["model","budget"],as_index=False).agg(q=("test_utility_ratio","mean"),accuracy=("test_correct","mean"),seconds=("elapsed_seconds","mean"),cost=("total_compute_cost_usd","mean"))
    sf=sto.groupby(["model","budget"],as_index=False).agg(q=("test_utility_ratio","mean"))
    sys.path.insert(0,str(Path(__file__).parent));import analyze_normalized_game as game_analysis
    game_analysis.MODEL_LABELS.update(LABELS)
    rows=[]
    for model in LABELS:
        x=f[f.model==model].set_index("budget");s=sf[sf.model==model].set_index("budget");gain=x.loc[768,"q"]-x.loc[0,"q"];positive=x.loc[x.index>0]
        time_slope=float(np.polyfit(positive.index.to_numpy(float),positive.seconds.to_numpy(float),1)[0]);cost_slope=float(np.polyfit(positive.index.to_numpy(float),positive.cost.to_numpy(float),1)[0]) if np.isfinite(positive.cost).sum()>1 else math.nan
        qgame=s.loc[[128,384,768],"q"].to_numpy(float);trans=game_analysis.exact_transition_rows(model,qgame,np.array([128,384,768]),1.0)
        nash="; ".join(f"{r['compute_before']}→{r['compute_after']}@{r['threshold']:.3g}" for r in trans if r["boundary_type"]=="private_NE_kappa")
        planner="; ".join(f"{r['compute_before']}→{r['compute_after']}@{r['threshold']:.3g}" for r in trans if r["boundary_type"]=="planner_kappa_over_rho")
        rows.append({"model":model,"model_label":LABELS[model],"q0":x.loc[0,"q"],"q768":x.loc[768,"q"],"absolute_endpoint_gain":gain,"relative_endpoint_gain":gain/x.loc[0,"q"] if x.loc[0,"q"] else math.nan,"accuracy0_percent":100*x.loc[0,"accuracy"],"accuracy768_percent":100*x.loc[768,"accuracy"],"stochastic_q128":s.loc[128,"q"],"stochastic_q768":s.loc[768,"q"],"stochastic_endpoint_gain":s.loc[768,"q"]-s.loc[128,"q"],"incremental_runtime_seconds_per_token":time_slope,"lambda_usd_per_token":cost_slope,"heldout_prize_benchmark_usd":prize,"kappa_benchmark":cost_slope/prize if math.isfinite(cost_slope) and prize>0 else math.nan,"gamma1_nash_thresholds":nash,"gamma1_planner_thresholds":planner})
    table=pd.DataFrame(rows);table.to_csv(out/"qwen_scale_table.csv",index=False)
    def fmt(v): return f"{v:.3f}" if isinstance(v,(float,np.floating)) and math.isfinite(v) else "--" if pd.isna(v) else str(v).replace("_",r"\_")
    (out/"qwen_scale_table.tex").write_text("\n".join(" & ".join(fmt(v) for v in row)+r" \\" for row in table.itertuples(index=False,name=None))+"\n")
    fig,ax=plt.subplots(figsize=(5.4,3.4))
    for model in LABELS:
        part=f[f.model==model].sort_values("budget");ax.plot(part.budget,part.q,marker="o",label=LABELS[model])
    ax.set(xlabel="Assigned reasoning tokens",ylabel="Mean held-out utility ratio",title="Qwen-family scale sensitivity");ax.grid(alpha=.25);ax.legend(frameon=False);fig.tight_layout()
    for ext in ["pdf","png"]:fig.savefig(out/f"qwen_scale_frontier.{ext}",dpi=300)
    plt.close(fig)
    note=["","## Qwen family scale sensitivity","","The three checkpoint rows are reported descriptively; they do not establish a scaling law.",""]
    for row in table.itertuples(): note.append(f"{row.model_label}: q(0)={row.q0:.3f}, q(768)={row.q768:.3f}, endpoint gain={row.absolute_endpoint_gain:.3f}.")
    summary=out/"HISTORICAL_NEW_MODELS_SUMMARY.md"
    if summary.exists(): summary.write_text(summary.read_text()+"\n"+"\n".join(note)+"\n")
    print(table.to_string(index=False))
if __name__=="__main__":main()
