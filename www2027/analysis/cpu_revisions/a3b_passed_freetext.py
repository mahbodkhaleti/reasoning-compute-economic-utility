"""A3 detail: in complete forced-768 traces whose letter answer passes over the trace's best option, how often is the
free-text answer parsed, and when parsed, does it name the trace's best option (i.e. is the pass-over a letter artefact)?"""
import sys, numpy as np
sys.path.insert(0, "/tmp/www2027/analysis/cpu_revisions")
from cr_common import *
R = {}
for m in MODELS:
    for bud in (768, -1):
        z = tf[(tf.model == m) & (tf.budget == bud) & tf.complete]
        pas = z[~np.isclose(z.chosen_bid, z.best_eval_bid)]
        parsed = pas[[(not np.isnan(fb)) and any(np.isclose(fb, x) for x in S[i]["bids"]) for fb, i in zip(pas.freeform_chosen_bid, pas.scenario_id)]]
        R[f"{m}|{bud}"] = dict(n_passed=int(len(pas)), parsed_share=float(len(parsed) / max(len(pas), 1)),
                              free_names_trace_best=float(np.isclose(parsed.freeform_chosen_bid, parsed.best_eval_bid).mean()) if len(parsed) else None,
                              free_equals_letter=float(np.isclose(parsed.freeform_chosen_bid, parsed.chosen_bid).mean()) if len(parsed) else None)
dump("A3b_passed_freetext", R)
for k, v in R.items(): print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})
