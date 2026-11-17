"""Part A analyses (CPU only, saved outputs). Each section writes results/cpu_revisions/<A#>.json.
A1/A7 audit incentives and dollar intervals; A2 can an auditor see v; A3 letter vs free-text answer;
A4 cut mid-option vs at a boundary; A5 natural stopping; A6 leave-one-value-family-out audit."""
import sys, json
import numpy as np, pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit
sys.path.insert(0, "/tmp/www2027/analysis/cpu_revisions")
from cr_common import *
which = sys.argv[1:] or ["A1", "A2", "A3", "A4", "A5", "A6"]
z768 = tf[tf.budget == 768].copy()
def reader_pick(vals, b, need=2):
    return max(vals.items(), key=lambda kv: kv[1][0])[0] if len(vals) >= need else b
z768["audit_bid"] = [S[i]["calc"] if not np.isclose(pick(vv, 2, b), b) else b for vv, b, i in zip(z768.vals, z768.chosen_bid, z768.scenario_id)]

# ---------------- A1 + A7
if "A1" in which:
    def econ(i, b):
        h = HELD[i][float(b)]; v = S[i]["v"]; p = h["test_p"]; x = TEST_PRICES[S[i]["product"]]; win = x <= b
        return dict(pyb_rev=p * b, pyb_sur=p * (v - b), sp_rev=float(np.mean(np.where(win, x, 0))), sp_sur=float(np.mean(np.where(win, v - x, 0))))
    R = {}
    for m in MODELS:
        z = z768[z768.model == m].reset_index(drop=True)
        for who, col in (("model", "chosen_bid"), ("audit", "audit_bid"), ("calc", "calc")):
            e = pd.DataFrame([econ(i, b) for i, b in zip(z.scenario_id, z[col])])
            for k in e: z[f"{who}_{k}"] = e[k].values
        for rule in ("pyb", "sp"):
            for who in ("model", "audit", "calc"): z[f"{who}_{rule}_tot"] = z[f"{who}_{rule}_rev"] + z[f"{who}_{rule}_sur"]
        res = {}
        for rule in ("pyb", "sp"):
            for meas in ("rev", "sur", "tot"):
                cols = [f"{w}_{rule}_{meas}" for w in ("model", "audit", "calc")]
                f = lambda w, cols=cols: np.array([1000 * w[c].mean() for c in cols] + [1000 * (w[cols[1]] - w[cols[0]]).mean(), 1000 * (w[cols[2]] - w[cols[0]]).mean()])
                b = boot_ci(f, z)
                res[f"{rule}_{meas}"] = dict(zip(["model", "audit", "calc", "audit_minus_model", "calc_minus_model"], b))
        res["calls"] = float(np.mean(~np.isclose(z.audit_bid, z.chosen_bid)))
        R[m] = res
    dump("A1_audit_incentives", R)
    for m in MODELS:
        r = R[m]; f_ = lambda x: f"{x['pt']/1000:.1f}k [{x['ci'][0]/1000:.1f}, {x['ci'][1]/1000:.1f}]"
        print(LAB[m], "PYB rev audit-model", f_(r["pyb_rev"]["audit_minus_model"]), "| sur", f_(r["pyb_sur"]["audit_minus_model"]), "| tot", f_(r["pyb_tot"]["audit_minus_model"]),
              "|| SP rev", f_(r["sp_rev"]["audit_minus_model"]), "| sur", f_(r["sp_sur"]["audit_minus_model"]), "| tot", f_(r["sp_tot"]["audit_minus_model"]))
        print("   A7 model-calc PYB rev", f_({"pt": -r["pyb_rev"]["calc_minus_model"]["pt"], "ci": [-r["pyb_rev"]["calc_minus_model"]["ci"][1], -r["pyb_rev"]["calc_minus_model"]["ci"][0]]}),
              "sur", f_({"pt": -r["pyb_sur"]["calc_minus_model"]["pt"], "ci": [-r["pyb_sur"]["calc_minus_model"]["ci"][1], -r["pyb_sur"]["calc_minus_model"]["ci"][0]]}))

# ---------------- A2
if "A2" in which:
    vm = pd.read_csv(f"{OUT}/v_mentions.csv")
    R = {"v_stated": {}, "v_minus_b_written": {}, "reader_without_v": {}}
    for m in MODELS:
        for bud in (768, -1):
            w = vm[(vm.model == m) & (vm.budget == bud)]
            R["v_stated"][f"{m}|{bud}"] = float(w.v_stated.mean()); R["v_minus_b_written"][f"{m}|{bud}"] = float(w.v_minus_b_written.mean())
    # text reader without v: skip only the bid and the displayed probability (not v or v-b)
    def read_vals_nov(segs, info, W=600):
        out = {}
        for b, st, nxt, nums in segs:
            b = float(b)
            if b not in info: continue
            last = None
            for x, pos in nums:
                if pos >= nxt or pos >= st + W: break
                if round(x, 4) == round(b, 4) or abs(x - info[b]) < 5e-5: continue
                last = (x, pos)
            if last is not None: out[b] = last
        return out
    for m in MODELS:
        z = z768[z768.model == m]
        vn = [read_vals_nov(sg, info_h[i]) for sg, i in zip(z.segs, z.scenario_id)]
        f_with = np.array([not np.isclose(pick(vv, 2, b), b) for vv, b in zip(z.vals, z.chosen_bid)])
        f_wo = np.array([not np.isclose(pick(vv, 2, b), b) for vv, b in zip(vn, z.chosen_bid)])
        R["reader_without_v"][m] = dict(same_flag=float((f_with == f_wo).mean()), flag_rate_with_v=float(f_with.mean()), flag_rate_without_v=float(f_wo.mean()))
    dump("A2_v_visibility", R)
    print(json.dumps(R, indent=0, default=lambda o: round(o, 3))[:3000])

# ---------------- A3
if "A3" in which:
    R = {"by_model_budget": {}, "freeform_gap": {}, "position_freeform": {}}
    q0 = tf[tf.budget == 0].groupby("model").q_read.mean()
    for m in MODELS:
        for bud in (0, 64, 128, 256, 512, 768, -1):
            w = tf[(tf.model == m) & (tf.budget == bud)]
            inmenu = np.array([(not np.isnan(fb)) and any(np.isclose(fb, x) for x in S[i]["bids"]) for fb, i in zip(w.freeform_chosen_bid, w.scenario_id)])
            dis = ~np.isclose(w.freeform_chosen_bid.fillna(-1).values, w.chosen_bid.values)
            R["by_model_budget"][f"{m}|{bud}"] = dict(parse_rate=float(inmenu.mean()), disagree_among_parsed=float(dis[inmenu].mean()) if inmenu.any() else None)
        # (a) free-text utility on parsed decisions at 768 (and the letter answer on the same decisions)
        w = tf[(tf.model == m) & (tf.budget == 768)].copy()
        w0 = tf[(tf.model == m) & (tf.budget == 0)].set_index(["scenario_id", "repeat_seed"]).q_read
        w["parsed"] = [(not np.isnan(fb)) and any(np.isclose(fb, x) for x in S[i]["bids"]) for fb, i in zip(w.freeform_chosen_bid, w.scenario_id)]
        w = w[w.parsed].reset_index(drop=True)
        w["q_free"] = [q_of(i, next(x for x in S[i]["bids"] if np.isclose(x, fb))) for i, fb in zip(w.scenario_id, w.freeform_chosen_bid)]
        w["q0"] = [w0[(i, s_)] for i, s_ in zip(w.scenario_id, w.repeat_seed)]
        f = lambda x: np.array([(x.q_free.mean() - x.q0.mean()) / (x.qcalc.mean() - x.q0.mean()), (x.q_read.mean() - x.q0.mean()) / (x.qcalc.mean() - x.q0.mean())])
        b = boot_ci(f, w)
        R["freeform_gap"][m] = dict(n_parsed=int(len(w)), parse_rate=float(len(w) / 648), q_free=float(w.q_free.mean()), q_letter_same=float(w.q_read.mean()),
                                    gap_free=b[0], gap_letter_same=b[1])
        # (b) position test on the free-text answer (as Table 3: complete traces at 768 and unforced, scenario x budget groups)
        y = tf[(tf.model == m) & tf.budget.isin([768, -1]) & tf.complete].copy()
        y = y[[(not np.isnan(fb)) and any(np.isclose(fb, x) for x in S[i]["bids"]) for fb, i in zip(y.freeform_chosen_bid, y.scenario_id)]]
        y["ignored"] = (~np.isclose(y.freeform_chosen_bid, y.best_eval_bid)).astype(int)
        def runner(i): s = S[i]; o = np.argsort(-s["eu"]); return float(s["bids"][o[1]])
        y["best_pos_rel"] = [(l.index(b) / (k - 1)) for l, b, k in zip(y.order_list, y.best_eval_bid, y.K)]
        y["runner_pos_rel"] = [(l.index(runner(i)) / (k - 1)) for l, i, k in zip(y.order_list, y.scenario_id, y.K)]
        y["grp"] = y.scenario_id.astype(str) + "|" + y.budget.astype(str)
        var = y.groupby("grp").ignored.transform(lambda s: s.nunique() > 1); yy = y[var]
        try:
            fit = ConditionalLogit(yy.ignored.values, yy[["best_pos_rel", "runner_pos_rel"]].values, groups=pd.factorize(yy.grp)[0]).fit(disp=0)
            ci = np.exp(fit.conf_int())
            R["position_freeform"][m] = dict(or_best=float(np.exp(fit.params[0])), ci=[float(ci[0, 0]), float(ci[0, 1])], p=float(fit.pvalues[0]), events=int(yy.ignored.sum()), n=int(len(y)))
        except Exception as e:
            R["position_freeform"][m] = dict(error=str(e), events=int(yy.ignored.sum()))
    dump("A3_letter_vs_freetext", R)
    for m in MODELS:
        print(LAB[m], {b_: (round(R["by_model_budget"][f"{m}|{b_}"]["parse_rate"], 2), round(R["by_model_budget"][f"{m}|{b_}"]["disagree_among_parsed"] or 0, 2)) for b_ in (0, 256, 512, 768, -1)},
              "gap free/letter", round(R["freeform_gap"][m]["gap_free"]["pt"], 2), round(R["freeform_gap"][m]["gap_letter_same"]["pt"], 2), "pos", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in R["position_freeform"][m].items() if k != "ci"}, [round(x, 2) for x in R["position_freeform"][m].get("ci", [])])

# ---------------- A4
if "A4" in which:
    R = {}
    for m in MODELS:
        for bud in (128, 256, 512, 768):
            z = tf[(tf.model == m) & (tf.budget == bud)].reset_index(drop=True)
            def cut_type(segs, hv):
                if not segs: return "no_option_mentioned"
                last_b = float(max(segs, key=lambda s_: s_[1])[0])
                return "boundary" if last_b in hv else "mid_option"
            z["cut"] = [cut_type(sg, hv) for sg, hv in zip(z.segs, z.hv)]
            out = {}
            for g_ in ("mid_option", "boundary", "no_option_mentioned"):
                w = z[z.cut == g_]
                if not len(w): continue
                T = w.gap.sum()
                out[g_] = dict(n=int(len(w)), share=float(len(w) / len(z)), accuracy=float(np.isclose(w.chosen_bid, w.calc).mean()),
                               split={k: (float(w.gap[w.cls == k].sum() / T) if T > 0 else None) for k in cats},
                               share_of_incomplete_gap=float(w.gap[w.cls == "incomplete"].sum() / max(z.gap[z.cls == "incomplete"].sum(), 1e-12)))
            R[f"{m}|{bud}"] = out
    dump("A4_cut_position", R)
    for m in MODELS:
        o = R[f"{m}|768"]
        print(LAB[m], {g_: (v["n"], round(v["accuracy"], 2), round(v["share_of_incomplete_gap"], 2)) for g_, v in o.items()})

# ---------------- A5
if "A5" in which:
    R = {}
    q0s = tf[tf.budget == 0].set_index(["model", "scenario_id", "repeat_seed"]).q_read
    for m in MODELS:
        u = tf[(tf.model == m) & (tf.budget == -1)]
        z = u[u.reasoning_tokens < 1024].reset_index(drop=True)
        n = len(z)
        res = dict(n=int(n), share=float(n / len(u)), too_small=bool(n < 50))
        if n:
            z["q0"] = [q0s[(m, i, s_)] for i, s_ in zip(z.scenario_id, z.repeat_seed)]
            res.update(q=float(z.q_read.mean()), qcalc=float(z.qcalc.mean()),
                       gap_recovered=float((z.q_read.mean() - z.q0.mean()) / (z.qcalc.mean() - z.q0.mean())) if z.qcalc.mean() > z.q0.mean() else None)
            T = z.gap.sum(); res["split"] = {k: (float(z.gap[z.cls == k].sum() / T) if T > 0 else None) for k in cats}
            fin = z[z.hv.map(len) >= z.K]
            res["n_finished"] = int(len(fin)); res["passed_rate_finished"] = float((fin.cls == "passed").mean()) if len(fin) else None
            res["wrong_rate"] = float((z.cls != "right").mean())
        R[m] = res
    dump("A5_natural_stopping", R)
    for m in MODELS: print(LAB[m], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R[m].items() if k != "split"}, {k: (round(v, 2) if v is not None else None) for k, v in R[m].get("split", {}).items()})

# ---------------- A6
if "A6" in which:
    grid = [(W, need) for W in (300, 600, 1000) for need in ("2", "K-1")]
    R = {}
    z = z768.copy()
    z["product"] = z.scenario_id.map(lambda i: S[i]["product"])
    for W, need in grid:
        z[f"qa_{W}_{need}"] = [q_of(i, S[i]["calc"]) if not np.isclose(pick(vv, (K - 1) if need == "K-1" else 2, b), b) else q_of(i, b)
                               for i, vv, K, b in zip(z.scenario_id, z[f"vals{W}"], z.K, z.chosen_bid)]
    q0 = tf[tf.budget == 0].groupby("model").q_read.mean()
    for m in MODELS:
        w = z[z.model == m].reset_index(drop=True); pred = np.zeros(len(w)); chosen = []
        for fam in w.fam.unique():
            tr = w[w.fam != fam]; te = w.fam == fam
            # fit on the other 17 families of the same product
            prod = w.loc[te, "product"].iloc[0]; tr = tr[tr["product"] == prod]
            best = max(grid, key=lambda g_: tr[f"qa_{g_[0]}_{g_[1]}"].mean())
            chosen.append(f"{best[0]}|{best[1]}"); pred[te.values] = w.loc[te, f"qa_{best[0]}_{best[1]}"].values
        w["q_cv"] = pred
        f = lambda x: np.array([(x.q_cv.mean() - q0[m]) / (x.qcalc.mean() - q0[m])])
        b = boot_ci(f, w)
        R[m] = dict(gap_recovered=b[0], chosen_settings=dict(pd.Series(chosen).value_counts()))
    dump("A6_audit_lofo_family", R)
    for m in MODELS: print(LAB[m], round(R[m]["gap_recovered"]["pt"], 3), [round(x, 3) for x in R[m]["gap_recovered"]["ci"]], R[m]["chosen_settings"])
