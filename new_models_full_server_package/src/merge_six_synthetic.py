#!/usr/bin/env python3
"""Merge synthetic rows only after benchmark/scoring provenance agrees exactly."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument("--package-root",type=Path,required=True);p.add_argument("--existing4-package",type=Path,required=True);a=p.parse_args();root=a.package_root.resolve();old=a.existing4_package.resolve()
    manifests=[old/"data/synthetic_ipv_manifest.jsonl",root/"data/synthetic/synthetic_ipv_manifest.jsonl"]
    hashes=[digest(x) for x in manifests]
    if len(set(hashes))!=1:raise RuntimeError(f"Synthetic benchmark mismatch: {hashes}")
    target=root/"results/synthetic_six_model";raw=target/"raw";raw.mkdir(parents=True,exist_ok=True)
    scenario_ids={json.loads(x)["scenario_id"] for x in manifests[0].read_text().splitlines() if x}
    sources=list((old/"results/raw").glob("synthetic_*_deterministic.jsonl"))+list((root/"results/synthetic/raw").glob("synthetic_*_deterministic.jsonl"))
    if len(sources)!=6:raise RuntimeError(f"Expected six model files, found {len(sources)}")
    models=set()
    for src in sources:
        rows=[json.loads(x) for x in src.read_text().splitlines() if x]; row_hashes={x.get("benchmark_sha256") for x in rows}; protocols={x.get("scoring_version") for x in rows}
        keys={(x["scenario_id"],int(x["reasoning_budget"]),int(x["option_order_seed"])) for x in rows};expected={(sid,b,o) for sid in scenario_ids for b in [0,64,128,256,384,512,768] for o in [11,23,37]}
        if row_hashes!={hashes[0]} or protocols!={"synthetic-ipv-exact-discrete-v1"} or keys!=expected:raise RuntimeError(f"Provenance or coverage mismatch in {src}")
        model=rows[0]["model_slug"];models.add(model);shutil.copy2(src,raw/src.name)
    if len(models)!=6:raise RuntimeError(f"Expected six unique models, found {sorted(models)}")
    paper=root/"paper_ready/synthetic_six_model";paper.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(root/"src/analyze_synthetic.py"),"--output-dir",str(target),"--paper-ready",str(paper)],check=True)
    subprocess.run([sys.executable,str(root/"src/analyze_synthetic_game.py"),"--output-dir",str(target)],check=True)
    shutil.copy2(target/"tables/synthetic_gamma1_exact_thresholds.csv",paper/"six_model_gamma1_thresholds.csv")
    shutil.copy2(target/"tables/synthetic_gamma1_exact_thresholds.tex",paper/"six_model_gamma1_thresholds.tex")
    phases=__import__('pandas').read_csv(target/"tables/synthetic_normalized_phase_grid.csv.gz")
    lines=["# Six-model synthetic strategic summary","","This deterministic-frontier strategic robustness summary is conditional on the specified normalized contest.",""]
    for model in sorted(models):
        regions=sorted(phases[(phases.model==model)&(abs(phases.gamma-1.0)<1e-12)].phase.unique());lines.append(f"- {model}: gamma=1 regions on the declared grid: {', '.join(regions)}.")
    (paper/"SIX_MODEL_STRATEGIC_SUMMARY.md").write_text("\n".join(lines)+"\n")
    pd=__import__('pandas');qwen=pd.read_csv(paper/"deterministic_frontier.csv");qwen=qwen[qwen.model.isin(["qwen3_4b","qwen3_8b","qwen3_14b"])]
    qwen.to_csv(paper/"synthetic_qwen_family_frontier.csv",index=False)
    qe=pd.read_csv(paper/"endpoint_table.csv");qe=qe[qe.model.isin(["qwen3_4b","qwen3_8b","qwen3_14b"])].copy();qe["relative_endpoint_gain"]=qe["difference"]/qe["q0"]
    qe.to_csv(paper/"synthetic_qwen_family_endpoints.csv",index=False);(paper/"synthetic_qwen_family_endpoints.tex").write_text(qe.to_latex(index=False,float_format="%.4f"),encoding="utf-8")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(5.4,3.4))
    for model,part in qwen.groupby("model"):ax.plot(part.budget,part.q,marker="o",label=part.model_label.iloc[0])
    ax.set(xlabel="Assigned reasoning tokens",ylabel="Mean synthetic utility ratio",title="Synthetic Qwen-family sensitivity");ax.grid(alpha=.25);ax.legend(frameon=False);fig.tight_layout()
    for ext in ["pdf","png"]:fig.savefig(paper/f"synthetic_qwen_family_frontier.{ext}",dpi=300)
    plt.close(fig)
    print(json.dumps({"benchmark_sha256":hashes[0],"models":sorted(models),"output":str(target)},indent=2))
if __name__=="__main__":main()
