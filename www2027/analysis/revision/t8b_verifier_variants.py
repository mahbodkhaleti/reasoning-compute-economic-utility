"""Verifier variants on the tokens-versus-value frontier.
always:   best evaluated option whenever any option is evaluated (as in t8 F3)
complete: best evaluated option only when every option is evaluated; otherwise the model's answer
k_minus_1: best evaluated option when at least K-1 options are evaluated; otherwise the model's answer
"""
import json, numpy as np, pandas as pd
from common_rev import *
tf = load_parse(0.02)
d = records(); g = d[d.decoding == "greedy"]; q0 = g[g.budget == 0].groupby("model").test_utility_ratio.mean()
tf["q_complete_v"] = np.where(tf.complete, tf.q_be, tf.q_read)
tf["q_km1_v"] = np.where(tf.n_eval >= tf.K - 1, tf.q_be, tf.q_read)
out = {}
for m in MODELS:
    pts = []
    for bud in [0, 64, 128, 256, 512, 768, -1]:
        z = tf[(tf.model == m) & (tf.budget == bud)]; den = z.qcalc.mean() - q0[m]
        pts.append(dict(budget=bud, tokens=float(z.reasoning_tokens.fillna(0).mean()) if bud == -1 else float(bud),
                        q_readout=float(z.q_read.mean()), q_always=float(z.q_be.mean()), q_complete=float(z.q_complete_v.mean()), q_km1=float(z.q_km1_v.mean()),
                        gap_readout=float((z.q_read.mean() - q0[m]) / den), gap_complete=float((z.q_complete_v.mean() - q0[m]) / den),
                        gap_km1=float((z.q_km1_v.mean() - q0[m]) / den), gap_always=float((z.q_be.mean() - q0[m]) / den)))
    out[m] = pts
json.dump(out, open(f"{OUTDIR}/t8b_verifier_variants.json", "w"), indent=1)
for m in MODELS: print(LAB[m], [(p["budget"], round(p["q_readout"], 3), round(p["q_always"], 3), round(p["q_complete"], 3), round(p["q_km1"], 3)) for p in out[m]])
