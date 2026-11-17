"""Item 3: build a 100-trace hand-labelling package (stratified by model, 768-token runs) and the code outputs to
score labels against: per-option values found by the reader and the matcher, finished list, stated choice and
gap category. Labels themselves must be made by a person; nothing here fills them in."""
import sys, json, csv, glob, re; sys.path.insert(0, "/tmp/www2027/analysis/rev25")
from r_common import *
exec(open("r3_stated_answer.py").read().split("rows = []")[0].split("from r_common import *")[1])
OUTD = "/tmp/www2027/human_annotation/hand_label_100"
raw = {}
for fn in glob.glob("/tmp/www2027/raw_traces/fixed_*_reasoning_greedy.jsonl"):
    for line in open(fn):
        r = json.loads(line)
        if r["budget"] == 768: raw[(r["model"], int(r["scenario_id"]), int(r["repeat_seed"]))] = r
rng = np.random.default_rng(20261115)
z = tf[tf.budget == 768].reset_index(drop=True)
per = {m: 17 for m in MODELS}; per[MODELS[0]] = 15; per[MODELS[1]] = 15     # 17*4 + 15*2 = 98 -> top up below
picks = []
for m in MODELS:
    idx = np.where(z.model == m)[0]; picks += list(rng.choice(idx, per[m], replace=False))
extra = rng.choice(np.setdiff1d(np.arange(len(z)), picks), 100 - len(picks), replace=False); picks += list(extra)
order = rng.permutation(len(picks))
sheet, key = [], []
for k, j in enumerate(order):
    r = z.iloc[picks[j]]; tid = f"T{k+1:03d}"; R_ = raw[(r.model, int(r.scenario_id), int(r.repeat_seed))]
    L = "ABCDEFG"; opts = R_["option_order"]; s = S[int(r.scenario_id)]
    pmap = dict(zip(s["bids"], s["dp"]))
    header = f"Private value v = ${s['v']:.2f}\n" + "\n".join(f"Option {L[i]}: bid ${b:.2f}; estimated win probability {pmap[float(b)]:.4f}" for i, b in enumerate(opts))
    open(f"{OUTD}/traces/{tid}.txt", "w").write(header + "\n\n===== TRACE (768 tokens, cut off at the end) =====\n" + R_["reasoning_text"])
    row = {"trace_id": tid, "n_options": len(opts)}
    for i in range(7):
        row[f"value_written_{L[i]}"] = "" if i < len(opts) else "n/a"; row[f"value_correct_{L[i]}"] = "" if i < len(opts) else "n/a"
    row.update(list_finished="", stated_choice="", notes="")
    sheet.append(row)
    vals, fp = r.vals, {float(b) for b in r.fp}
    key.append(dict(trace_id=tid, model=r.model, scenario_id=int(r.scenario_id), repeat_seed=int(r.repeat_seed), option_order=json.dumps(opts),
                    true_values=json.dumps({L[i]: round(float(s["eu"][np.isclose(s["bids"], b)][0]), 4) for i, b in enumerate(opts)}),
                    reader_values=json.dumps({L[i]: vals[float(b)][0] for i, b in enumerate(opts) if float(b) in vals}),
                    matcher_found=json.dumps([L[i] for i, b in enumerate(opts) if float(b) in fp]),
                    code_list_finished=int(len(r.hv) >= r.K), code_stated_choice=(L[opts.index(st_)] if (st_ := stated(R_["reasoning_text"], opts)) is not None else ""),
                    letter_answer=R_["constrained_option_label"], calculator_option=L[opts.index(s["calc"])], code_category=r.cls))
with open(f"{OUTD}/label_sheet.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sheet[0])); w.writeheader(); w.writerows(sheet)
with open(f"{OUTD}/for_scoring/code_outputs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(key[0])); w.writeheader(); w.writerows(key)
print(pd.Series([k["model"] for k in key]).value_counts().to_dict())
