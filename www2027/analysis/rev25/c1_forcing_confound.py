"""C1: forcing confound. (a) Table 2 split on the unforced 1,024-token traces. (b) Decisions whose unforced trace
stopped on its own before 768 tokens: answer at the natural stop (unforced run) vs the forced-768 answer.
(c) Share of forced-768 "passed over" errors whose unforced trace had already stopped before 768 tokens."""
import sys, json, glob; sys.path.insert(0, "/tmp/www2027/analysis/rev25")
from r_common import *
tok = {}
for fn in glob.glob("/tmp/www2027/raw_traces/fixed_*_reasoning_greedy.jsonl"):
    for l in open(fn):
        r = json.loads(l)
        if r["budget"] == -1: tok[(r["model"], int(r["scenario_id"]), int(r["repeat_seed"]))] = r["reasoning_tokens"]
R = {}
for m in MODELS:
    u = tf[(tf.model == m) & (tf.budget == -1)].reset_index(drop=True)
    f = tf[(tf.model == m) & (tf.budget == 768)].set_index(["scenario_id", "repeat_seed"])
    gap = u.gap.values; cl = u.cls.values; fams = u.fam.values
    split = dict(zip(CATS, fboot(lambda ids: np.array([gap[ids][cl[ids] == k].sum() / gap[ids].sum() for k in CATS]), fams)))
    nt = np.array([tok[(m, i, s)] for i, s in zip(u.scenario_id, u.repeat_seed)])
    early = nt < 768
    ur = np.isclose(u.chosen_bid.values, u.calc.values)
    fr = np.array([np.isclose(f.loc[(i, s)].chosen_bid, f.loc[(i, s)].calc) for i, s in zip(u.scenario_id, u.repeat_seed)])
    fcls = np.array([f.loc[(i, s)].cls for i, s in zip(u.scenario_id, u.repeat_seed)])
    R[m] = dict(unforced_split=split, n_unforced_wrong=int((cl != "right").sum()), n_stopped_before_768=int(early.sum()),
                share_stopped_before_768=float(early.mean()),
                right_unforced_wrong_forced=int((early & ur & ~fr).sum()), wrong_unforced_right_forced=int((early & ~ur & fr).sum()),
                right_to_wrong_rate=float((early & ur & ~fr).sum() / max((early & ur).sum(), 1)),
                passed_forced_n=int((fcls == "passed").sum()), passed_forced_after_natural_stop=int(((fcls == "passed") & early).sum()),
                passed_forced_after_natural_stop_share=float(((fcls == "passed") & early).sum() / max((fcls == "passed").sum(), 1)))
json.dump(R, open("/tmp/www2027/results/extra/C1_forcing_confound.json", "w"), indent=1, default=float)
for m, r in R.items():
    print(LAB[m], {k: f"{100*v['pt']:.0f} [{100*v['ci'][0]:.0f},{100*v['ci'][1]:.0f}]" for k, v in r["unforced_split"].items()}, "| wrong", r["n_unforced_wrong"],
          "| stopped<768:", r["n_stopped_before_768"], f"({100*r['share_stopped_before_768']:.0f}%)", "right->wrong", r["right_unforced_wrong_forced"], f"({100*r['right_to_wrong_rate']:.1f}%)",
          "wrong->right", r["wrong_unforced_right_forced"], "| passed forced", r["passed_forced_n"], "after natural stop", r["passed_forced_after_natural_stop"], f"({100*r['passed_forced_after_natural_stop_share']:.0f}%)")
