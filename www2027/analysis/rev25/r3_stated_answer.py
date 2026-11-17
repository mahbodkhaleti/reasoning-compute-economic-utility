"""Item 3: is Phi-mini's "passed over" an artefact of the letter readout? A stricter rule-based parser reads the
answer the trace itself states ("Option C has the highest expected profit", "the best bid to choose is Option C",
"I choose $420", "answer is C"), taking the last such statement in the trace (reasoning and any text after </think>).
No LLM involved. Then the Table 2 split is recomputed with that stated answer in place of the letter.
Check of the parser: for models whose letter readout is not in question, the stated answer should agree with the
letter in finished traces."""
import sys, re, json, glob; sys.path.insert(0, "/tmp/www2027/analysis/rev25")
from r_common import *
RAW = "/tmp/www2027/raw_traces"
SUP = r"(?:highest|best|largest|maximum|greatest|optimal|max)"
PATS = [
    re.compile(SUP + r"[^.\n]{0,80}?\b[Oo]ption\s+([A-G])\b"),
    re.compile(r"\b[Oo]ption\s+([A-G])\b[^.\n]{0,60}?" + SUP),
    re.compile(r"(?:choose|select|pick|go with|answer is|answer:)\s*:?\s*\**(?:[Oo]ption\s+)?\(?([A-G])\)?\b(?!\w)"),
]
AMT = re.compile(r"(?:choose|select|pick|go with|answer is|bid of|best bid is|optimal bid is|final bid is)[^.\n]{0,15}?\$\s?(\d+(?:\.\d+)?)", re.I)
def stated(text, order):
    t = text.replace(",", ""); best = None
    for p in PATS:
        for m in p.finditer(t):
            j = ord(m.group(1)) - 65
            if j < len(order) and (best is None or m.start() > best[0]): best = (m.start(), float(order[j]))
    for m in AMT.finditer(t):
        x = float(m.group(1))
        if any(abs(x - b) < 1e-6 for b in order) and (best is None or m.start() > best[0]): best = (m.start(), x)
    return None if best is None else best[1]
rows = []
for fn in glob.glob(f"{RAW}/fixed_*_reasoning_greedy.jsonl"):
    for line in open(fn):
        r = json.loads(line)
        if r["budget"] not in (512, 768, -1): continue
        txt = (r.get("reasoning_text") or "")
        rows.append(dict(model=r["model"], scenario_id=int(r["scenario_id"]), repeat_seed=int(r["repeat_seed"]), budget=int(r["budget"]),
                         stated=stated(txt, r["option_order"])))
st = pd.DataFrame(rows)
R = {}
for m in MODELS:
    for bud in (768, -1):
        z = tf[(tf.model == m) & (tf.budget == bud)].merge(st, on=["model", "scenario_id", "repeat_seed", "budget"], how="left").reset_index(drop=True)
        has = z.stated.notna().values
        fin = np.array([len(h) >= K for h, K in zip(z.hv, z.K)])
        agree = np.isclose(z.stated.fillna(-1).values, z.chosen_bid.values)
        stated_right = np.isclose(z.stated.fillna(-1).values, z.calc.values)
        letter_right = np.isclose(z.chosen_bid.values, z.calc.values)
        res = dict(coverage=float(has.mean()), coverage_finished=float(has[fin].mean()),
                   agree_with_letter=float(agree[has].mean()), agree_with_letter_finished=float(agree[has & fin].mean()),
                   letter_right_where_parsed=float(letter_right[has].mean()), stated_right_where_parsed=float(stated_right[has].mean()))
        if bud == 768:
            # split with the stated answer replacing the letter wherever the trace states one
            ans = np.where(has, z.stated.fillna(0).values, z.chosen_bid.values)
            qa = np.array([q_of(i, b) for i, b in zip(z.scenario_id, ans)])
            cl = np.array([classify(hv, i, K, b) for hv, i, K, b in zip(z.hv, z.scenario_id, z.K, ans)])
            gap = z.qcalc.values - qa; fams = z.fam.values; q0m = q0[m]
            res["gap_recovered_stated"] = fboot(lambda ids: (qa[ids].mean() - q0m) / (z.qcalc.values[ids].mean() - q0m), fams)[0]
            res["gap_recovered_letter"] = float((z.q_read.mean() - q0m) / (z.qcalc.mean() - q0m))
            res["split_stated"] = dict(zip(CATS, fboot(lambda ids: np.array([gap[ids][cl[ids] == k].sum() / gap[ids].sum() for k in CATS]), fams)))
            pl = (z.cls.values == "passed"); res["letter_passed_n"] = int(pl.sum())
            res["letter_passed_with_stated"] = int((pl & has).sum())
            res["letter_passed_where_stated_is_best"] = int((pl & stated_right).sum())
            res["stated_passed_n"] = int((cl == "passed").sum())
        R[f"{m}|{bud}"] = res
dump25("r3_stated_answer", R)
for k, v in R.items():
    print(k, {a: (round(b, 3) if isinstance(b, float) else (f"{100*b['pt']:.0f} [{100*b['ci'][0]:.0f},{100*b['ci'][1]:.0f}]" if isinstance(b, dict) and 'pt' in b else b)) for a, b in v.items() if a != "split_stated"})
    if "split_stated" in v: print("    split stated:", {c: f"{100*x['pt']:.0f} [{100*x['ci'][0]:.0f},{100*x['ci'][1]:.0f}]" for c, x in v["split_stated"].items()})
