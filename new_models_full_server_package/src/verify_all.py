#!/usr/bin/env python3
"""CPU-only coverage and integrity audit for both additional models."""
from __future__ import annotations
import argparse, hashlib, json, math
from collections import Counter
from pathlib import Path

MODELS = {"qwen3_14b": "40c069824f4251a91eefaf281ebe4c544efd3e18", "granite_3_2_8b_instruct": "610d8c6ee9c84ce51f6dfd7bc5c0215d95d49695"}

def rows(path: Path):
    good, corrupt = [], []
    if not path.exists(): return good, corrupt
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        try: good.append(json.loads(line))
        except Exception as exc: corrupt.append({"line": n, "error": str(exc)})
    return good, corrupt

def audit(path: Path, expected: set, revision: str, budget_field="budget", seed_field="repeat_seed", hashes: dict | None=None):
    data, corrupt = rows(path)
    keys = [(str(r.get("scenario_id")).zfill(12), int(r.get(budget_field, -999)), int(r.get(seed_field, -999))) for r in data]
    counts = Counter(keys); observed = set(keys)
    invalid = finite = token = revision_bad = hash_bad = 0
    for r in data:
        label=r.get("constrained_option_label",r.get("selected_option_label"));allowed=[chr(65+i) for i in range(int(r.get("option_count",0)))]
        invalid += r.get("readout_method") != "constrained_option_letter_next_token" or r.get("chosen_bid", r.get("selected_bid")) is None or label not in allowed
        numeric = [r.get(k) for k in ("test_utility_ratio", "test_regret")] if budget_field == "budget" else [r.get(k) for k in ("utility_ratio", "regret")]
        finite += any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in numeric)
        b = int(r.get(budget_field, -999)); realized = r.get("reasoning_tokens", r.get("realized_reasoning_tokens"))
        token += (b == 0 and realized != 0) or (b > 0 and realized != b)
        revision_bad += r.get("resolved_model_revision") != revision or r.get("resolved_tokenizer_revision") != revision
        if hashes: hash_bad += any(r.get(field)!=value for field,value in hashes.items())
    return {"file": str(path), "expected": len(expected), "rows": len(data), "unique_expected": len(observed & expected), "missing": len(expected-observed), "unexpected": len(observed-expected), "duplicates": sum(v-1 for v in counts.values() if v>1), "corrupt": len(corrupt), "invalid_constrained": invalid, "nan_or_inf": finite, "token_violations": token, "revision_mismatches": revision_bad,"benchmark_hash_mismatches":hash_bad}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--package-root",type=Path,required=True); p.add_argument("--allow-incomplete",action="store_true"); a=p.parse_args(); root=a.package_root
    scenarios=[str(x).zfill(12) for x in __import__('pandas').read_csv(root/"data/historical/scenarios.csv",dtype={"scenario_id":str})["scenario_id"]]
    synthetic=[json.loads(x)["scenario_id"] for x in (root/"data/synthetic/synthetic_ipv_manifest.jsonl").read_text().splitlines() if x]
    digest=hashlib.sha256((root/"data/synthetic/synthetic_ipv_manifest.jsonl").read_bytes()).hexdigest(); expected_digest=(root/"data/synthetic/manifest.sha256").read_text().split()[0];historical_hashes={"scenario_manifest_sha256":hashlib.sha256((root/"data/historical/scenarios.csv").read_bytes()).hexdigest(),"auction_split_sha256":hashlib.sha256((root/"data/historical/auction_splits.csv").read_bytes()).hexdigest()}
    report={"historical_scenarios":len(scenarios),"synthetic_scenarios":len(synthetic),"synthetic_hash_matches":digest==expected_digest,"models":{}}
    failed=len(scenarios)!=216 or len(synthetic)!=216 or digest!=expected_digest
    for model, rev in MODELS.items():
        det={(s,b,o) for s in scenarios for b in [0,64,128,256,512,768,-1] for o in [11,23,37]}
        nt={(s,0,o) for s in scenarios for o in [11,23,37]}
        sto={(s,b,g) for s in scenarios for b in [128,384,768] for g in [1,7,21,42,99]}
        syn={(s,b,o) for s in synthetic for b in [0,64,128,256,384,512,768] for o in [11,23,37]}
        audits=[audit(root/f"results/historical/raw/fixed_{model}_reasoning_greedy.jsonl",det,rev,hashes=historical_hashes),audit(root/f"results/historical/raw/fixed_{model}_no_think_greedy.jsonl",nt,rev,hashes=historical_hashes),audit(root/f"results/historical/raw/fixed_{model}_reasoning_stochastic.jsonl",sto,rev,hashes=historical_hashes),audit(root/f"results/synthetic/raw/synthetic_{model}_deterministic.jsonl",syn,rev,"reasoning_budget","option_order_seed",{"benchmark_sha256":expected_digest})]
        greedy_rows,_=rows(root/f"results/historical/raw/fixed_{model}_reasoning_greedy.jsonl");greedy_keys={(str(x.get('scenario_id')).zfill(12),int(x.get('budget',-999)),int(x.get('repeat_seed',-999))) for x in greedy_rows}
        deterministic_expected={(s,b,o) for s in scenarios for b in [0,64,128,256,512,768] for o in [11,23,37]};unconstrained_expected={(s,-1,o) for s in scenarios for o in [11,23,37]}
        category_coverage={"historical_deterministic":{"expected":len(deterministic_expected),"completed":len(greedy_keys&deterministic_expected),"missing":len(deterministic_expected-greedy_keys)},"historical_unconstrained":{"expected":len(unconstrained_expected),"completed":len(greedy_keys&unconstrained_expected),"missing":len(unconstrained_expected-greedy_keys)},"historical_no_think":{"expected":648,"completed":audits[1]["unique_expected"],"missing":audits[1]["missing"]},"historical_stochastic":{"expected":3240,"completed":audits[2]["unique_expected"],"missing":audits[2]["missing"]},"synthetic":{"expected":4536,"completed":audits[3]["unique_expected"],"missing":audits[3]["missing"]}}
        report["models"][model]={"expected_breakdown":{"historical_deterministic":3888,"historical_unconstrained":648,"historical_no_think":648,"historical_stochastic":3240,"synthetic":4536,"total":12960},"category_coverage":category_coverage,"audits":audits}
        for item in audits:
            failed |= bool(item["unexpected"] or item["duplicates"] or item["corrupt"] or item["invalid_constrained"] or item["nan_or_inf"] or item["token_violations"] or item["revision_mismatches"] or item["benchmark_hash_mismatches"])
            if not a.allow_incomplete: failed |= bool(item["missing"])
    report["valid"]=not failed
    out=root/"results/verification_report.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if failed: raise SystemExit(1)
if __name__=="__main__": main()
