"""eBay numbers for the arXiv version that were not stored in one place: q and gap recovered per forced budget with
family-bootstrap intervals, the unforced 1,024 run, finished shares and token use. Uses r_common (paper code path).
Writes results/arxiv/ebay_budget.json."""
import sys, json, os, numpy as np
sys.path.insert(0, "/tmp/www2027/analysis/rev25")
from r_common import *
os.makedirs("/tmp/www2027/results/arxiv", exist_ok=True)
R = {}
for m in MODELS:
    z = tf[tf.model == m]; r = {}
    q0 = z[z.budget == 0].sort_values(["scenario_id", "repeat_seed"]); fams = q0.fam.values
    qc = q0.qcalc.values; a0 = q0.q_read.values
    for b in sorted(z.budget.unique()):
        zb = z[z.budget == b].sort_values(["scenario_id", "repeat_seed"]); qb = zb.q_read.values
        st = lambda ids: (qb[ids].mean() - a0[ids].mean()) / (qc[ids].mean() - a0[ids].mean())
        r[int(b)] = dict(q=float(qb.mean()), gap=fboot(st, fams) if b != 0 else None,
                         finished=float(np.mean([len(hv) >= K for hv, K in zip(zb.hv, zb.K)])), finished_matcher=float(np.mean([len(fp) >= K for fp, K in zip(zb.fp, zb.K)])))
    R[m] = dict(by_budget=r, qcalc=float(qc.mean()))
json.dump(R, open("/tmp/www2027/results/arxiv/ebay_budget.json", "w"), indent=1, default=float)
for m in MODELS:
    print(LAB[m], {b: (round(v["q"], 3), None if v["gap"] is None else round(100 * v["gap"][0]["pt"]), round(100*v["finished_matcher"])) for b, v in R[m]["by_budget"].items()})
