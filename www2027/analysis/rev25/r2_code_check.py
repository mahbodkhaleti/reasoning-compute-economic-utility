"""Item 2: arithmetic checked by code on every saved trace (no LLM annotators in the loop).
For each option the trace writes a value for (text reader, no arithmetic), compare it with the true expected profit
p_hat(b)(v - b) computed from the displayed numbers. A value is correct if within 2% (or $0.05) of the true value.
Two independent code methods are compared: the matcher (true value appears anywhere in the trace) and the reader
(last result number in the option's own segment). Reports, per model and budget, traces checked, options written,
reader-vs-truth agreement, matcher-vs-reader agreement, and how the Arith. category changes if values come from the
reader alone instead of the hybrid rule."""
import sys; sys.path.insert(0, "/tmp/www2027/analysis/rev25")
from r_common import *
ok = lambda x, t: abs(x - t) <= max(0.02 * abs(t), 0.05)
R = {}
for m in MODELS:
    for bud in (256, 512, 768, -1):
        z = tf[(tf.model == m) & (tf.budget == bud)]
        n_opt = n_correct = both = both_agree = matcher_only = reader_only = 0
        traces_with_error = 0
        for vals, fp, i in zip(z.vals, z.fp, z.scenario_id):
            s = S[i]; tr = dict(zip(s["bids"], s["eu"])); err = False
            fpb = {float(b) for b in fp}
            for b, (x, _) in vals.items():
                n_opt += 1; c = ok(x, tr[b]); n_correct += c
                if b in fpb: both += 1; both_agree += c
                else: reader_only += 1; err |= not c
            matcher_only += len(fpb - set(vals))
            traces_with_error += err
        # Arith. category with reader-only values vs hybrid
        def cls_reader(vals, i, K, chosen):
            s = S[i]; calc = s["calc"]; eu = lambda o: s["eu"][np.isclose(s["bids"], o)][0]
            if np.isclose(chosen, calc): return "right"
            if calc not in vals: return "not_reached"
            if any(vals[o][0] >= vals[calc][0] for o in vals if o != calc and eu(o) < eu(calc)): return "arith"
            return "incomplete" if len(vals) < K else "passed"
        cr = np.array([cls_reader(v, i, K, b) for v, i, K, b in zip(z.vals, z.scenario_id, z.K, z.chosen_bid)])
        R[f"{m}|{bud}"] = dict(traces=int(len(z)), options_written=n_opt, reader_correct_share=n_correct / max(n_opt, 1),
                               options_found_by_both=both, reader_matches_truth_when_matcher_found=both_agree / max(both, 1),
                               options_matcher_only=matcher_only, options_reader_only=reader_only,
                               traces_with_a_miscomputed_value=traces_with_error / max(len(z), 1),
                               arith_share_of_wrong_hybrid=float((z.cls == "arith").sum() / max((z.cls != "right").sum(), 1)),
                               arith_share_of_wrong_reader_only=float((cr == "arith").sum() / max((cr != "right").sum(), 1)),
                               same_category_hybrid_vs_reader_only=float((cr == z.cls.values).mean()))
dump25("r2_code_check", R)
tot = sum(v["traces"] for v in R.values()); print("traces checked:", tot)
for k, v in R.items():
    if k.endswith("|768") or k.endswith("|-1"):
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
