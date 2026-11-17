"""Run 1 (natural stopping, cap 4,096): metrics computed through the paper's own code paths.
- q, gap recovered (family bootstrap, 2,000 draws), with q(0) and the calculator from the paper's saved records;
- cap share, tokens, finished-list share (hybrid written values cover all K options);
- Table 2 split (hybrid written values, categories in the paper's order) with intervals;
- break-even with the paper's cost basis (seconds per token on the original GPUs, RunPod rates).
Outputs: results/followup/run1_summary.json and printed tables."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "/tmp/www2027/analysis/cpu_revisions")
from cr_common import *          # paper tf (forced 768 etc.) with hv/cls, S, q_of, family_boot_ids, d, read_vals, hybrid, classify, decompose
import cr_common as C
OUTF = "/tmp/www2027/results/followup"
os.makedirs(OUTF, exist_ok=True)
A = "/tmp/www2027/analysis"
RUN = "/tmp/www2027/results/followup/results/followup/run1_nat4096"
M1 = ["qwen3_4b", "qwen3_8b", "qwen3_14b", "deepseek_r1_llama_8b"]

# ---- new traces through the same steps as load_parse + t10_judge7 + t10b
nf = pd.read_csv(f"{A}/followup_run1_first.csv")
nf["K"] = nf.scenario_id.map(lambda i: S[i]["K"]); nf["fp"] = nf.first_pos.map(json.loads)
nf["q_read"] = [q_of(i, b) for i, b in zip(nf.scenario_id, nf.chosen_bid)]
nf["calc"] = nf.scenario_id.map(lambda i: S[i]["calc"]); nf["qcalc"] = nf.scenario_id.map(lambda i: q_of(i, S[i]["calc"]))
nf["fam"] = nf.scenario_id.map(lambda i: S[i]["fam"])
seg = pd.read_csv(f"{A}/followup_run1_segments.csv.gz").rename(columns={"seed": "repeat_seed", "n_chars": "n_chars_seg"})
nf = nf.merge(seg, on=["model", "scenario_id", "repeat_seed", "budget"], how="left")
nf["segs"] = nf.segments.fillna("[]").map(json.loads)
info_h = {i: dict(zip(S[i]["bids"], S[i]["dp"])) for i in S}
nf["vals"] = [read_vals(sg, S[i]["v"], info_h[i], 600) for sg, i in zip(nf.segs, nf.scenario_id)]
nf["hv"] = [hybrid(vv, fp, i) for vv, fp, i in zip(nf.vals, nf.fp, nf.scenario_id)]
nf["cls"] = [classify(hv, i, K, b) for hv, i, K, b in zip(nf.hv, nf.scenario_id, nf.K, nf.chosen_bid)]
nf["gap"] = nf.qcalc - nf.q_read
raw = pd.concat([pd.read_json(f"{RUN}/{m}.jsonl", lines=True) for m in M1])
raw["scenario_id"] = raw.scenario_id.astype(np.int64)
nf = nf.merge(raw[["model", "scenario_id", "repeat_seed", "hit_cap", "wall_seconds", "energy_joules", "test_utility_ratio", "freeform_valid", "freeform_chosen_bid"]],
              on=["model", "scenario_id", "repeat_seed"], how="left")
assert np.allclose(nf.q_read, nf.test_utility_ratio)

g = d[d.decoding == "greedy"]
cost = pd.read_csv(f"{A}/runtime_cost_recomputed.csv").set_index("model")
cats = ["not_reached", "arith", "incomplete", "passed"]
def boot(fn, fams, n=2000, seed=20261111):
    r = np.random.default_rng(seed); pt = fn(np.arange(len(fams)))
    bs = np.array([fn(family_boot_ids(fams, r)) for _ in range(n)])
    return dict(pt=float(pt), ci=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))])
out = {}
for m in M1:
    z = nf[nf.model == m].sort_values(["scenario_id", "repeat_seed"]).reset_index(drop=True)
    key = list(zip(z.scenario_id, z.repeat_seed))
    def col(b):
        y = g[(g.model == m) & (g.budget == b)].set_index(["scenario_id", "repeat_seed"]).test_utility_ratio
        return y.reindex(key).values
    q0, q768, q1024 = col(0), col(768), col(-1)
    qn, qc, fams = z.q_read.values, z.qcalc.values, z.fam.values
    gr = lambda qq: (lambda ids: (qq[ids].mean() - q0[ids].mean()) / (qc[ids].mean() - q0[ids].mean()))
    sec_tok = float(cost.loc[m, "seconds_per_reasoning_token"]); price = 0.79 if m == "qwen3_14b" else 0.74
    lam = sec_tok * price / 3600; tok = z.reasoning_tokens.values
    c768, cnat = 768 * lam, tok.mean() * lam
    dec = {}
    T = z.gap.values; cl = z.cls.values
    def sh(k): return lambda ids: T[ids][cl[ids] == k].sum() / T[ids].sum()
    for k in cats: dec[k] = dict(n=int((cl == k).sum()), **boot(sh(k), fams))
    hv_all = np.array([len(h) >= K for h, K in zip(z.hv, z.K)])
    out[m] = dict(
        n=int(len(z)), q_natural=boot(lambda ids: qn[ids].mean(), fams), q0=float(q0.mean()), q768=float(q768.mean()),
        q1024_run=float(q1024.mean()), q_calc=float(qc.mean()),
        gap_recovered_natural=boot(gr(qn), fams), gap_recovered_768=boot(gr(q768), fams), gap_recovered_1024run=boot(gr(q1024), fams),
        at_cap=float(z.hit_cap.mean()), tokens_median=float(np.median(tok)), tokens_mean=float(tok.mean()),
        tokens_p90=float(np.percentile(tok, 90)),
        finished_list_hybrid=float(hv_all.mean()), finished_list_parser=float((z.fp.map(len) == z.K).mean()),
        picks_calculator=float(np.isclose(z.chosen_bid, z.calc).mean()), wrong_n=int((cl != "right").sum()),
        freeform_valid=float(z.freeform_valid.mean()),
        split=dec,
        cost_basis=dict(seconds_per_token_paper=sec_tok, gpu_hour_usd=price, c768_cents=100 * c768, cnat_cents=100 * cnat),
        breakeven_768_usd=c768 / (q768.mean() - q0.mean()), breakeven_natural_usd=cnat / (qn.mean() - q0.mean()),
        measured_this_gpu=dict(seconds_per_decision=float(z.wall_seconds.mean()), wh_per_decision=float(z.energy_joules.mean() / 3600)))
json.dump(out, open(f"{OUTF}/run1_summary.json", "w"), indent=1)
f = lambda x: f"{100*x:.0f}"
ci = lambda r: f"{f(r['pt'])} [{f(r['ci'][0])}, {f(r['ci'][1])}]"
for m, r in out.items():
    print(LAB[m], "n", r["n"], "q0 %.3f q768 %.3f q1024 %.3f qnat %.3f qcalc %.3f" % (r["q0"], r["q768"], r["q1024_run"], r["q_natural"]["pt"], r["q_calc"]))
    print("   gap: 768", ci(r["gap_recovered_768"]), "| 1024 run", ci(r["gap_recovered_1024run"]), "| natural", ci(r["gap_recovered_natural"]))
    print("   cap %.1f%% tokens med %.0f mean %.0f p90 %.0f | finished list %.0f%% (parser %.0f%%) | picks calc %.1f%% wrong %d | freeform valid %.0f%%" % (
        100 * r["at_cap"], r["tokens_median"], r["tokens_mean"], r["tokens_p90"], 100 * r["finished_list_hybrid"], 100 * r["finished_list_parser"], 100 * r["picks_calculator"], r["wrong_n"], 100 * r["freeform_valid"]))
    print("   split", {k: (v["n"], ci(v)) for k, v in r["split"].items()})
    print("   cost c768 %.2fc cnat %.2fc | B/e 768 $%.3f natural $%.3f | this GPU %.1fs %.2fWh" % (
        r["cost_basis"]["c768_cents"], r["cost_basis"]["cnat_cents"], r["breakeven_768_usd"], r["breakeven_natural_usd"],
        r["measured_this_gpu"]["seconds_per_decision"], r["measured_this_gpu"]["wh_per_decision"]))
