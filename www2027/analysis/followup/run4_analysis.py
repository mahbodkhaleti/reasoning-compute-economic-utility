"""Run 4 (prompt baseline): one added instruction ("End by listing the expected profit of each option, then answer
with the option that has the largest."), forced 768 and unforced cap 1,024, for Phi-mini, Granite-8B, DeepSeek-8B and
Qwen3-8B. Same pipeline as the paper (letter readout, hybrid written values, categories, family bootstrap); q(0) and
the calculator from the paper's records. Compared with the paper's own forced-768 and 1,024-run decisions (paired)."""
import os, sys, json, re
import numpy as np, pandas as pd
sys.path.insert(0, "/tmp/www2027/analysis/cpu_revisions")
from cr_common import *
src = open("/tmp/www2027/analysis/rev25/r3_stated_answer.py").read()
exec(src[src.index("SUP ="):src.index("rows = []")])          # the paper's stated-answer parser
A = "/tmp/www2027/analysis"; RUN = "/tmp/www2027/results/followup/run4/results/followup/run4_prompt"
M4 = ["phi4_mini_reasoning", "granite_3_2_8b_instruct", "deepseek_r1_llama_8b", "qwen3_8b"]
CATS = ["not_reached", "arith", "incomplete", "passed"]
nf = pd.read_csv(f"{A}/followup_run4_first.csv")
nf["K"] = nf.scenario_id.map(lambda i: S[i]["K"]); nf["fp"] = nf.first_pos.map(json.loads)
nf["q_read"] = [q_of(i, b) for i, b in zip(nf.scenario_id, nf.chosen_bid)]
nf["calc"] = nf.scenario_id.map(lambda i: S[i]["calc"]); nf["qcalc"] = nf.scenario_id.map(lambda i: q_of(i, S[i]["calc"]))
nf["fam"] = nf.scenario_id.map(lambda i: S[i]["fam"])
seg = pd.read_csv(f"{A}/followup_run4_segments.csv.gz").rename(columns={"seed": "repeat_seed", "n_chars": "n_chars_seg"})
nf = nf.merge(seg, on=["model", "scenario_id", "repeat_seed", "budget"], how="left")
nf["segs"] = nf.segments.fillna("[]").map(json.loads)
info_h = {i: dict(zip(S[i]["bids"], S[i]["dp"])) for i in S}
nf["vals"] = [read_vals(sg, S[i]["v"], info_h[i], 600) for sg, i in zip(nf.segs, nf.scenario_id)]
nf["hv"] = [hybrid(vv, fp, i) for vv, fp, i in zip(nf.vals, nf.fp, nf.scenario_id)]
nf["cls"] = [classify(hv, i, K, b) for hv, i, K, b in zip(nf.hv, nf.scenario_id, nf.K, nf.chosen_bid)]
nf["gap"] = nf.qcalc - nf.q_read
raw = pd.concat([pd.read_json(f"{RUN}/{m}.jsonl", lines=True) for m in M4]); raw["scenario_id"] = raw.scenario_id.astype(np.int64)
raw = raw.drop(columns=["budget"]).rename(columns={"setting": "budget"})
nf = nf.merge(raw[["model", "scenario_id", "repeat_seed", "budget", "reasoning_text", "option_order", "hit_cap", "test_utility_ratio"]], on=["model", "scenario_id", "repeat_seed", "budget"])
assert np.allclose(nf.q_read, nf.test_utility_ratio)
g = d[d.decoding == "greedy"]
def boot(fn, fams, n=2000, seed=20261112):
    r = np.random.default_rng(seed); pt = np.atleast_1d(fn(np.arange(len(fams))))
    bs = np.array([np.atleast_1d(fn(family_boot_ids(fams, r))) for _ in range(n)])
    return [dict(pt=float(pt[j]), ci=[float(np.nanpercentile(bs[:, j], 2.5)), float(np.nanpercentile(bs[:, j], 97.5))]) for j in range(len(pt))]
OUT = {}
for m in M4:
    for setting, pb in (("forced768", 768), ("unforced1024", -1)):
        z = nf[(nf.model == m) & (nf.budget == setting)].sort_values(["scenario_id", "repeat_seed"]).reset_index(drop=True)
        key = list(zip(z.scenario_id, z.repeat_seed))
        pz = tf[(tf.model == m) & (tf.budget == pb)].set_index(["scenario_id", "repeat_seed"]).loc[key].reset_index()
        q0v = g[(g.model == m) & (g.budget == 0)].set_index(["scenario_id", "repeat_seed"]).test_utility_ratio.reindex(key).values
        qn, qp, qc, fams = z.q_read.values, pz.q_read.values, z.qcalc.values, z.fam.values
        gr = lambda q: (lambda ids: (q[ids].mean() - q0v[ids].mean()) / (qc[ids].mean() - q0v[ids].mean()))
        st = np.array([stated(t, o) if isinstance(t, str) else None for t, o in zip(z.reasoning_text, z.option_order)], dtype=object)
        has = np.array([x is not None for x in st]); ans = np.where(has, [x if x is not None else 0 for x in st], z.chosen_bid.values).astype(float)
        qs = np.array([q_of(i, b) for i, b in zip(z.scenario_id, ans)])
        gap = z.gap.values; cl = z.cls.values
        fin = np.array([len(h) >= K for h, K in zip(z.hv, z.K)]); pfin = np.array([len(h) >= K for h, K in zip(pz.hv, pz.K)])
        OUT[f"{m}|{setting}"] = dict(
            q_new=float(qn.mean()), q_paper=float(qp.mean()),
            gap_new=boot(gr(qn), fams)[0], gap_paper=boot(gr(qp), fams)[0],
            gap_diff=boot(lambda ids: gr(qn)(ids) - gr(qp)(ids), fams)[0],
            gap_new_stated=boot(gr(qs), fams)[0],
            stated_coverage=float(has.mean()), stated_agrees_letter=float(np.mean(np.isclose(ans[has], z.chosen_bid.values[has]))),
            stated_right=float(np.mean(np.isclose(ans[has], z.calc.values[has]))), letter_right=float(np.mean(np.isclose(z.chosen_bid, z.calc))),
            finished_new=float(fin.mean()), finished_paper=float(pfin.mean()), at_cap=float(z.hit_cap.mean()),
            split_new=dict(zip(CATS, boot(lambda ids: np.array([gap[ids][cl[ids] == k].sum() / gap[ids].sum() for k in CATS]), fams))),
            wrong_new=int((cl != "right").sum()), wrong_paper=int((pz.cls.values != "right").sum()))
os.makedirs("/tmp/www2027/results/followup", exist_ok=True)
json.dump(OUT, open("/tmp/www2027/results/followup/run4_summary.json", "w"), indent=1)
f = lambda r: f"{100*r['pt']:.0f} [{100*r['ci'][0]:.0f}, {100*r['ci'][1]:.0f}]"
for k, r in OUT.items():
    print(k, "q %.3f (paper %.3f) | gap %s vs paper %s | diff %s | stated-scored %s | stated cov %.0f%% agree %.0f%% stated right %.0f%% letter right %.0f%% | finished %.0f%% (paper %.0f%%) cap %.0f%% | wrong %d (paper %d)" % (
        r["q_new"], r["q_paper"], f(r["gap_new"]), f(r["gap_paper"]), f(r["gap_diff"]), f(r["gap_new_stated"]), 100 * r["stated_coverage"], 100 * r["stated_agrees_letter"],
        100 * r["stated_right"], 100 * r["letter_right"], 100 * r["finished_new"], 100 * r["finished_paper"], 100 * r["at_cap"], r["wrong_new"], r["wrong_paper"]))
    print("    split", {c: f(v) for c, v in r["split_new"].items()})
