#!/usr/bin/env python3
"""Create manuscript-facing CSV/LaTeX rows only from completed analysis outputs."""
from __future__ import annotations
import argparse, json, math, shutil
from pathlib import Path
import numpy as np
import pandas as pd

MODELS={"qwen3_14b":"Qwen3-14B","granite_3_2_8b_instruct":"Granite-3.2-8B-Instruct"}

def load_jsonl(paths):
    rows=[]
    for path in paths:
        if path.exists(): rows += [json.loads(x) for x in path.read_text().splitlines() if x]
    return pd.DataFrame(rows)

def tex(frame: pd.DataFrame, path: Path):
    def value(x):
        if pd.isna(x): return "--"
        if isinstance(x,(float,np.floating)): return f"{x:.3f}"
        return str(x).replace("_",r"\_").replace("%",r"\%")
    path.write_text("\n".join(" & ".join(value(x) for x in row)+r" \\" for row in frame.itertuples(index=False,name=None))+"\n",encoding="utf-8")

def export(frame, name, out):
    frame.to_csv(out/f"{name}.csv",index=False); tex(frame,out/f"{name}.tex")

def main():
    p=argparse.ArgumentParser();p.add_argument("--package-root",type=Path,required=True);a=p.parse_args();root=a.package_root;h=root/"results/historical";out=root/"paper_ready/historical";out.mkdir(parents=True,exist_ok=True)
    greedy=load_jsonl([h/f"raw/fixed_{m}_reasoning_greedy.jsonl" for m in MODELS]); stochastic=load_jsonl([h/f"raw/fixed_{m}_reasoning_stochastic.jsonl" for m in MODELS])
    if greedy.empty or stochastic.empty: raise RuntimeError("Historical inference is incomplete; paper outputs require both greedy and stochastic files")
    greedy=greedy.drop_duplicates(["model","scenario_id","budget","repeat_seed"],keep="last");stochastic=stochastic.drop_duplicates(["model","scenario_id","budget","repeat_seed"],keep="last")
    constrained=greedy[greedy.budget.isin([0,64,128,256,512,768])]
    frontier=constrained.groupby(["model","budget"],as_index=False).agg(q=("test_utility_ratio","mean"),regret=("test_regret","mean"),exact_optimality=("test_correct","mean"));frontier["model_label"]=frontier.model.map(MODELS);frontier["exact_optimality_percent"]=100*frontier.exact_optimality
    export(frontier[["model","model_label","budget","q","regret","exact_optimality_percent"]],"complete_frontier_additional_rows",out)
    paired=pd.read_csv(h/"tables/paired_budget_comparisons.csv")
    export(paired[paired.model.isin(MODELS)],"table1_additional_rows",out)
    sto=stochastic.groupby(["model","budget"],as_index=False).agg(q=("test_utility_ratio","mean"),regret=("test_regret","mean"),exact_optimality=("test_correct","mean"),bid_dispersion=("chosen_bid","std"));sto["model_label"]=sto.model.map(MODELS);sto["exact_optimality_percent"]=100*sto.exact_optimality
    export(sto,"stochastic_frontier_additional_rows",out)
    copies={"freeform_sensitivity_summary.csv":"freeform_additional_rows","freeform_sensitivity_by_budget.csv":"freeform_available_case_diagnostics","auction_cdf_bootstrap_summary.csv":"cdf_bootstrap_additional_rows","normalized_transition_boundaries.csv":"gamma1_threshold_additional_rows","hybrid_zero_compute_bootstrap_summary.csv":"zero_compute_additional_rows","stochastic_game_uncertainty_summary.csv":"stochastic_hierarchical_intervals","confidence_calibration_summary.csv":"confidence_calibration_additional_rows","adaptive_sampling_summary.csv":"adaptive_sampling_additional_rows"}
    for source,name in copies.items():
        frame=pd.read_csv(h/"tables"/source); frame=frame[frame["model"].isin(MODELS)] if "model" in frame else frame
        if name.startswith("gamma1") and "gamma" in frame: frame=frame[np.isclose(frame.gamma,1.0)]
        export(frame,name,out)
    runtime=[]
    for model,part in stochastic.groupby("model"):
        x=part.budget.to_numpy(float); y=part.elapsed_seconds.to_numpy(float); slope=float(np.polyfit(x,y,1)[0]); energy=part.get("energy_joules",pd.Series(dtype=float)).to_numpy(float)
        energy_slope=float(np.polyfit(x[np.isfinite(energy)],energy[np.isfinite(energy)],1)[0]) if np.isfinite(energy).sum()>1 else math.nan
        cfg=json.loads((root/"config/experiment.json").read_text()); price=cfg["cost"].get("gpu_hour_usd"); lam=slope*price/3600 if price is not None else math.nan
        auction=pd.read_csv(root/"data/historical/auction_splits.csv"); prize=float(auction.loc[auction.split=="test","final_price"].mean()); kappa=lam/prize if math.isfinite(lam) and prize>0 else math.nan
        runtime.append({"model":model,"model_label":MODELS[model],"incremental_seconds_per_token":slope,"incremental_energy_joules_per_token":energy_slope,"gpu_hour_usd":price,"lambda_usd_per_token":lam,"heldout_prize_benchmark_usd":prize,"kappa":kappa})
    runtime=pd.DataFrame(runtime);export(runtime,"runtime_cost_additional_rows",out)
    metadata=[]
    for model,label in MODELS.items():
        spec=next(x for x in json.loads((root/"config/experiment.json").read_text())["models"] if x["slug"]==model);sample=greedy[greedy.model==model].iloc[0]
        metadata.append({"model":model,"model_label":label,"requested_model_id":spec["hf_id"],"resolved_model_revision":sample.get("resolved_model_revision"),"resolved_tokenizer_revision":sample.get("resolved_tokenizer_revision"),"dtype":sample.get("dtype"),"device_map":sample.get("device_map"),"resolved_device_map":sample.get("resolved_device_map"),"gpu_devices":sample.get("gpu_devices"),"hardware_name":sample.get("hardware_name"),"gpu_count_billed":sample.get("gpu_count_billed"),"reasoning_protocol":sample.get("reasoning_protocol"),"reasoning_protocol_matched":sample.get("reasoning_protocol_matched"),"gpu_driver":sample.get("gpu_driver")})
    export(pd.DataFrame(metadata),"model_metadata_additional_rows",out)
    freeform=pd.read_csv(h/"tables/freeform_sensitivity_summary.csv"); transitions=pd.read_csv(h/"tables/normalized_transition_boundaries.csv"); phases=pd.read_csv(h/"tables/normalized_phase_grid.csv.gz")
    summary=["# Additional-model results","","Generated mechanically from saved rows; no values are hand-entered. Strategic statements below are conditional on the specified normalized contest.",""]
    endpoints=paired[paired.model.isin(MODELS)]
    for model,label in MODELS.items():
        f=frontier[frontier.model==model].set_index("budget");s=sto[sto.model==model].set_index("budget");e=endpoints[endpoints.model==model].iloc[0];r=runtime[runtime.model==model].iloc[0];ff=freeform[freeform.model==model].iloc[0]
        tr=transitions[(transitions.model==model)&np.isclose(transitions.gamma,1.0)]; nash=tr[tr.boundary_type=="private_NE_kappa"].to_dict("records");planner=tr[tr.boundary_type=="planner_kappa_over_rho"].to_dict("records");regions=sorted(phases[(phases.model==model)&np.isclose(phases.gamma,1.0)].phase.unique())
        summary += [f"## {label}","",f"Deterministic q(0)={f.loc[0,'q']:.3f}; q(768)={f.loc[768,'q']:.3f}; paired difference={e.mean_paired_difference:.3f}, 95% CI [{e.ci_low:.3f}, {e.ci_high:.3f}].",f"Exact optimality changes from {f.loc[0,'exact_optimality_percent']:.1f}% to {f.loc[768,'exact_optimality_percent']:.1f}%.",f"Stochastic q(128/384/768)={s.loc[128,'q']:.3f}/{s.loc[384,'q']:.3f}/{s.loc[768,'q']:.3f}.",f"Free-form validity={100*ff.freeform_validity_rate:.1f}%; constrained/free-form agreement conditional on validity={100*ff.agreement_conditional_valid:.1f}%.",f"Incremental runtime/token={r.incremental_seconds_per_token:.6f}s; kappa={r.kappa if math.isfinite(r.kappa) else 'unpriced'}.",f"Gamma=1 Nash thresholds: `{nash}`.",f"Gamma=1 planner thresholds: `{planner}`.",f"Allocation regions present on the declared grid: {', '.join(regions)}.","",]
    (out/"RESULTS_SUMMARY.md").write_text("\n".join(summary),encoding="utf-8")
    (root/"paper_ready/HISTORICAL_NEW_MODELS_SUMMARY.md").write_text("\n".join(summary),encoding="utf-8")
    g="granite_3_2_8b_instruct";gf=frontier[frontier.model==g].set_index("budget");gs=sto[sto.model==g].set_index("budget");ge=endpoints[endpoints.model==g].iloc[0];gregions=sorted(phases[(phases.model==g)&np.isclose(phases.gamma,1.0)].phase.unique())
    granite=["# Granite cross-family robustness","",f"The deterministic 0→768 difference is {ge.mean_paired_difference:.3f}, with 95% CI [{ge.ci_low:.3f}, {ge.ci_high:.3f}]. Its sign is {'positive' if ge.mean_paired_difference>0 else 'negative' if ge.mean_paired_difference<0 else 'zero'}, and the interval {'excludes' if ge.ci_low>0 or ge.ci_high<0 else 'includes'} zero.",f"The stochastic 128→768 direction is {'positive' if gs.loc[768,'q']>gs.loc[128,'q'] else 'negative' if gs.loc[768,'q']<gs.loc[128,'q'] else 'zero'}. Gamma=1 grid regions: {', '.join(gregions)}.","", "Granite uses its checkpoint's `thinking=True` template plus external token budgeting. Because that mechanism is not protocol-identical to Qwen3 native thinking, this comparison is a protocol sensitivity."]
    (out/"GRANITE_CROSS_FAMILY_ROBUSTNESS.md").write_text("\n".join(granite)+"\n")
    print(f"Wrote paper-ready outputs to {out}")
if __name__=="__main__":main()
