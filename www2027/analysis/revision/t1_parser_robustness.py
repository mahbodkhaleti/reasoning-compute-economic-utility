"""Task 1: parser robustness at 1%, 2%, 5% tolerance, and an annotator-label version of the best-evaluated readout.
Outputs: out/t1_parser_robustness.json, out/t1_parser_robustness.csv
"""
import json, numpy as np, pandas as pd
from common_rev import *

rng = np.random.default_rng(20261112)
d = records(); g = d[d.decoding == "greedy"]
q0 = g[g.budget == 0].groupby("model").test_utility_ratio.mean()
B = 2000
rows, out = [], {}
for tol in [0.01, 0.02, 0.05]:
    tf = load_parse(tol); out[tol] = {}
    for m in MODELS:
        res = {}
        for bud in [512, 768]:
            y = tf[(tf.model == m) & (tf.budget == bud)].reset_index(drop=True)
            fams = y.fam.values
            qc = y.qcalc.values; qr = y.q_read.values; qb = y.q_be.values
            gap = qc - qr; unf = (~y.complete).values; wrong = ~np.isclose(y.chosen_bid.values, y.calc.values)
            m0 = q0[m]
            def stats(ids):
                den = qc[ids].mean() - m0
                return ((qr[ids].mean() - m0) / den, (qb[ids].mean() - m0) / den,
                        gap[ids][unf[ids]].sum() / gap[ids].sum(), y.complete.values[ids].mean())
            pt = stats(np.arange(len(y)))
            bs = np.array([stats(family_boot_ids(fams, rng)) for _ in range(B)])
            ci = lambda j: [float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]
            res[bud] = dict(gap_recovered_readout=float(pt[0]), gap_recovered_readout_ci=ci(0),
                            gap_recovered_best_eval=float(pt[1]), gap_recovered_best_eval_ci=ci(1),
                            unfinished_share_of_gap=float(pt[2]), unfinished_share_ci=ci(2),
                            ignored_share_of_gap=float(1 - pt[2]), complete=float(pt[3]), complete_ci=ci(3))
            rows.append(dict(tol=tol, model=LAB[m], budget=bud, **{k: v for k, v in res[bud].items() if not isinstance(v, list)}))
        out[tol][m] = res

# annotator-label version on the 150 annotated traces
vs = json.load(open("../validation_sample.json")); L = {}
for p in (0, 1):
    for x in json.load(open(f"/tmp/www2027/judge/label/labels_{p}.json")): L[x["item"]] = x
tf2 = load_parse(0.02).set_index(["model", "scenario_id", "repeat_seed", "budget"])
ann = []
for k, r in enumerate(vs):
    sid = int(r["scenario_id"]); s = S[sid]
    lab = {o["label"]: o for o in L[k]["options"]}
    stated = [(float(b), lab["ABCDEFG"[j]]["value"]) for j, b in enumerate(r["option_order"]) if lab["ABCDEFG"[j]]["stated"]]
    row = tf2.loc[(r["model"], sid, r["repeat_seed"], r["budget"])]
    # annotator version 1: option with the highest annotator-recorded stated value
    b_val = max(stated, key=lambda z: z[1] if z[1] is not None else -1e18)[0] if stated else np.nan
    # annotator version 2: among annotator-stated options, the one with the highest true expected profit
    b_true = max(stated, key=lambda z: s["eu"][np.isclose(s["bids"], z[0])][0])[0] if stated else np.nan
    qr = q_of(sid, row.chosen_bid)
    ann.append(dict(model=r["model"], budget=r["budget"], complete_ann=len(stated) == s["K"], complete_parser=bool(row.complete),
                    q_read=qr, q_be_parser=row.q_be,
                    q_be_ann_value=q_of(sid, b_val) if b_val == b_val else qr, q_be_ann_true=q_of(sid, b_true) if b_true == b_true else qr,
                    same_pick=bool(np.isclose(b_val, row.best_eval_bid)) if (b_val == b_val and row.best_eval_bid == row.best_eval_bid) else (b_val != b_val and row.best_eval_bid != row.best_eval_bid)))
A = pd.DataFrame(ann)
out["annotated_150"] = dict(n=len(A), q_read=float(A.q_read.mean()), q_be_parser=float(A.q_be_parser.mean()),
                            q_be_annotator_values=float(A.q_be_ann_value.mean()), q_be_annotator_true=float(A.q_be_ann_true.mean()),
                            same_pick_share=float(A.same_pick.mean()),
                            by_model={LAB[m]: dict(q_read=float(z.q_read.mean()), q_be_parser=float(z.q_be_parser.mean()),
                                                   q_be_annotator_values=float(z.q_be_ann_value.mean())) for m, z in A.groupby("model")})
json.dump(out, open(f"{OUTDIR}/t1_parser_robustness.json", "w"), indent=1, default=float)
pd.DataFrame(rows).to_csv(f"{OUTDIR}/t1_parser_robustness.csv", index=False)
T = pd.DataFrame(rows)
print(T[T.budget == 768].pivot_table(index="model", columns="tol", values=["complete", "unfinished_share_of_gap", "gap_recovered_best_eval"]).round(3).to_string())
print(T[T.budget == 512].pivot_table(index="model", columns="tol", values=["gap_recovered_readout", "gap_recovered_best_eval"]).round(3).to_string())
print(json.dumps(out["annotated_150"], indent=1, default=float))
