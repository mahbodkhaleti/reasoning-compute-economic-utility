"""R1: readout-template ablation on the saved eBay 768-token traces. Original template vs closing the reasoning block
with the model's own marker + "Final answer: Option". Per model: reproduction of the saved letter, letter-stated
agreement (paper's stated-answer parser), q(768) and gap recovered (paper q(0), calculator), family bootstrap."""
import sys, json, glob, gzip, re
import numpy as np
sys.path.insert(0, "/tmp/www2027/analysis/rev25")
from r_common import *
src = open("/tmp/www2027/analysis/rev25/r3_stated_answer.py").read(); exec(src[src.index("SUP ="):src.index("rows = []")])
RAW = "/tmp/www2027/gpu_followup/followup_gpu/data/ebay_raw768/"
OUT = {}
for f in sorted(glob.glob("/tmp/www2027/results/R1/results/extra/R1_readout_ablation/*.jsonl")):
    m = f.split("/")[-1][:-6]
    A = {(int(r["scenario_id"]), int(r["repeat_seed"])): r for r in map(json.loads, open(f))}
    P = {(int(r["scenario_id"]), int(r["repeat_seed"])): r for r in map(json.loads, gzip.open(RAW + m + ".jsonl.gz", "rt"))}
    z = tf[(tf.model == m) & (tf.budget == 768)].reset_index(drop=True)
    z = z[[ (i, s) in A for i, s in zip(z.scenario_id, z.repeat_seed)]].reset_index(drop=True)
    st = np.array([stated(P[(i, s)]["reasoning_text"], P[(i, s)]["option_order"]) for i, s in zip(z.scenario_id, z.repeat_seed)], dtype=object)
    has = np.array([x is not None for x in st]); stv = np.array([x if x is not None else -1 for x in st], float)
    orig = np.array([A[(i, s)]["orig_bid"] for i, s in zip(z.scenario_id, z.repeat_seed)]); alt = np.array([A[(i, s)]["alt_bid"] for i, s in zip(z.scenario_id, z.repeat_seed)])
    saved = z.chosen_bid.values
    qo = np.array([q_of(i, b) for i, b in zip(z.scenario_id, orig)]); qa = np.array([q_of(i, b) for i, b in zip(z.scenario_id, alt)])
    qc = z.qcalc.values; fams = z.fam.values; q0m = q0[m]
    gr = lambda q: (lambda ids: (q[ids].mean() - q0m) / (qc[ids].mean() - q0m))
    OUT[m] = dict(n=int(len(z)), reproduces_saved=float(np.isclose(orig, saved).mean()), alt_equals_orig=float(np.isclose(alt, orig).mean()),
                  agree_stated_orig=float(np.isclose(orig[has], stv[has]).mean()), agree_stated_alt=float(np.isclose(alt[has], stv[has]).mean()),
                  right_orig=float(np.isclose(orig, z.calc.values).mean()), right_alt=float(np.isclose(alt, z.calc.values).mean()),
                  q_orig=float(qo.mean()), q_alt=float(qa.mean()), gap_orig=fboot(gr(qo), fams)[0], gap_alt=fboot(gr(qa), fams)[0],
                  gap_diff=fboot(lambda ids: gr(qa)(ids) - gr(qo)(ids), fams)[0], alt_suffix=A[list(A)[0]]["alt_suffix"])
json.dump(OUT, open("/tmp/www2027/results/R1/r1_summary.json", "w"), indent=1)
f_ = lambda r: f"{100*r['pt']:.0f} [{100*r['ci'][0]:.0f}, {100*r['ci'][1]:.0f}]"
for m, r in OUT.items():
    print(m, "n", r["n"], "reproduces saved %.3f | letter=stated orig %.0f%% alt %.0f%% | right orig %.0f%% alt %.0f%% | gap orig %s alt %s diff %s | alt==orig %.0f%%" % (
        r["reproduces_saved"], 100*r["agree_stated_orig"], 100*r["agree_stated_alt"], 100*r["right_orig"], 100*r["right_alt"], f_(r["gap_orig"]), f_(r["gap_alt"]), f_(r["gap_diff"]), 100*r["alt_equals_orig"]))
# Table 2 split with the closed-block answer
SPL = {}
for f in sorted(glob.glob("/tmp/www2027/results/R1/results/extra/R1_readout_ablation/*.jsonl")):
    m = f.split("/")[-1][:-6]
    A = {(int(r["scenario_id"]), int(r["repeat_seed"])): r for r in map(json.loads, open(f))}
    z = tf[(tf.model == m) & (tf.budget == 768)].reset_index(drop=True)
    alt = np.array([A[(i, s)]["alt_bid"] for i, s in zip(z.scenario_id, z.repeat_seed)])
    qa = np.array([q_of(i, b) for i, b in zip(z.scenario_id, alt)]); gap = z.qcalc.values - qa
    cl = np.array([classify(hv, i, K, b) for hv, i, K, b in zip(z.hv, z.scenario_id, z.K, alt)])
    SPL[m] = dict(zip(CATS, fboot(lambda ids: np.array([gap[ids][cl[ids] == k].sum() / gap[ids].sum() for k in CATS]), z.fam.values)))
    print(m, "closed-block split", {k: f_(v) for k, v in SPL[m].items()})
OUT["_split_closed"] = SPL
json.dump(OUT, open("/tmp/www2027/results/R1/r1_summary.json", "w"), indent=1)
