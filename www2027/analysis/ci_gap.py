import json, numpy as np, pandas as pd
exec(open("batch7_judge_r3.py").read().split("# ---------- R1")[0])
rng = np.random.default_rng(5)
out = {}
for m in MODELS:
    y7 = tf[(tf.model == m) & (tf.budget == 768)].copy()
    y0 = g[(g.model == m) & (g.budget == 0)][["scenario_id", "repeat_seed", "test_utility_ratio"]].rename(columns={"test_utility_ratio": "q0"})
    y7 = y7.merge(y0, on=["scenario_id", "repeat_seed"])
    y7["q_be"] = [q_of(i, b) if b == b else q for i, b, q in zip(y7.scenario_id, y7.best_eval_bid, y7.q_read)]
    fams = y7.scenario_id.map(lambda i: S[i]["fam"]).values; idx = {f: np.where(fams == f)[0] for f in FAM}
    def gr(ids, col): return (y7[col].values[ids].mean() - y7.q0.values[ids].mean()) / (y7.qcalc.values[ids].mean() - y7.q0.values[ids].mean())
    allids = np.arange(len(y7)); b1, b2 = [], []
    for _ in range(2000):
        ids = np.concatenate([idx[f] for p in PRODUCTS for f in rng.choice([f for f in FAM if f.startswith(p)], 18, replace=True)])
        b1.append(gr(ids, "q_read")); b2.append(gr(ids, "q_be"))
    out[m] = dict(gap_ratio=gr(allids, "q_read"), gap_ratio_ci=[np.percentile(b1, 2.5), np.percentile(b1, 97.5)],
                  gap_best_eval=gr(allids, "q_be"), gap_best_eval_ci=[np.percentile(b2, 2.5), np.percentile(b2, 97.5)],
                  q768=y7.q_read.mean(), q_be=y7.q_be.mean())
json.dump(out, open("out/gap_ci.json", "w"), indent=1, default=float)
for m, v in out.items(): print(LAB[m], {k: (np.round(v_, 3) if not isinstance(v_, list) else [round(a, 2) for a in v_]) for k, v_ in v.items()})
