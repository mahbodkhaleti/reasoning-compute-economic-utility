"""Scores your hand labels in annotation_sheet_40.csv against the parser (v2, 2%).
Fill evaluated_X with 1 (the trace states an expected-profit value for option X) or 0, and value_X with the last
stated value (optional). Blank evaluated cells for options that exist count as 0; options beyond K are ignored.
usage: python3 t6_score_annotations.py annotation_sheet_40_filled.csv [trace_first_v2.csv] [scenarios.csv]
Prints option-level precision, recall and Cohen's kappa, trace-level completeness agreement and kappa, and the
share of your stated values within 2% of the true expected profit.
"""
import csv, json, sys
SHEET = sys.argv[1]
PARSE = sys.argv[2] if len(sys.argv) > 2 else "../trace_first_v2.csv"
SCEN = sys.argv[3] if len(sys.argv) > 3 else "../scenarios.csv"
parse = {}
for r in csv.DictReader(open(PARSE)):
    parse[(r["model"], str(int(r["scenario_id"])), int(r["repeat_seed"]), int(r["budget"]))] = {float(b) for b in json.loads(r["first_pos"])}
eu = {}
for r in csv.DictReader(open(SCEN)):
    for c in json.loads(r["candidates_json"]):
        eu[(str(int(r["scenario_id"])), float(c["bid"]))] = (float(r["value"]) - c["bid"]) * round(c["train_p"], 4)
def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n; pa = sum(a) / n; pb = sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb); return (po - pe) / (1 - pe) if pe < 1 else 1.0
H, Pp, TH, TP, vals = [], [], [], [], []
for r in csv.DictReader(open(SHEET, encoding="utf-8")):
    k = (r["model"], str(int(r["scenario_id"])), int(r["order_seed"]), int(r["budget"]))
    opts = [o.split(":")[0].strip() for o in r["options"].split(";")]
    bids = [float(o.split("$")[1].split(",")[0]) for o in r["options"].split(";")]
    hs, ps = [], []
    for L, b in zip(opts, bids):
        h = 1 if str(r.get(f"evaluated_{L}", "")).strip().lower() in ("1", "y", "yes", "true", "x") else 0
        p = 1 if b in parse[k] else 0
        hs.append(h); ps.append(p); H.append(h); Pp.append(p)
        v = str(r.get(f"value_{L}", "")).strip()
        if h and v:
            try:
                t = eu[(k[1], b)]; vals.append(abs(float(v.replace("$", "")) - t) <= max(0.02 * abs(t), 0.05))
            except ValueError: pass
    TH.append(int(all(hs))); TP.append(int(all(ps)))
tp = sum(h and p for h, p in zip(H, Pp)); fp = sum((not h) and p for h, p in zip(H, Pp)); fn = sum(h and (not p) for h, p in zip(H, Pp))
print(f"traces {len(TH)}, options {len(H)}")
print(f"option precision {tp / max(tp + fp, 1):.3f}  recall {tp / max(tp + fn, 1):.3f}  kappa {kappa(H, Pp):.3f}")
print(f"trace completeness: you {sum(TH) / len(TH):.3f}, parser {sum(TP) / len(TP):.3f}, agreement {sum(a == b for a, b in zip(TH, TP)) / len(TH):.3f}, kappa {kappa(TH, TP):.3f}")
if vals: print(f"your stated values within 2% of the truth: {sum(vals) / len(vals):.3f} (n={len(vals)})")
