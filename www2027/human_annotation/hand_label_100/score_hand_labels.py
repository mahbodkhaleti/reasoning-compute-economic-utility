"""Score hand labels against the code outputs (for_scoring/code_outputs.csv).
Option level: the reader's value vs the human's written value (match within 2% or $0.05); the matcher's 'evaluated'
flag vs the human's 'value written and correct'. Trace level: finished list, stated choice, gap category (derived
from the human labels with the paper's rules: Not reached / Arithmetic / Cut off / Passed over / right)."""
import csv, json, sys
lab = {r["trace_id"]: r for r in csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else "label_sheet.csv"))}
code = {r["trace_id"]: r for r in csv.DictReader(open("for_scoring/code_outputs.csv"))}
close = lambda a, b: abs(a - b) <= max(0.02 * abs(b), 0.05)
L = "ABCDEFG"; tp = fp = fn = 0; mtp = mfp = mfn = 0; fin = st = cat = n = 0; st_n = 0
def human_cat(h, c):
    true = json.loads(c["true_values"]); calc = c["calculator_option"]; ans = c["letter_answer"]; K = len(true)
    if ans == calc: return "right"
    vals = {x: float(h[f"value_written_{x}"]) for x in true if h.get(f"value_written_{x}", "").strip() not in ("", "n/a")}
    if calc not in vals: return "not_reached"
    if any(vals[o] >= vals[calc] for o in vals if o != calc and true[o] < true[calc]): return "arith"
    return "incomplete" if len(vals) < K else "passed"
for tid, h in lab.items():
    if not any(h.get(f"value_written_{x}", "").strip() for x in L) and not h.get("list_finished"): continue
    c = code[tid]; n += 1; true = json.loads(c["true_values"]); rd = json.loads(c["reader_values"]); mf = set(json.loads(c["matcher_found"]))
    for x in true:
        hv = h.get(f"value_written_{x}", "").strip()
        if hv:
            if x in rd and close(rd[x], float(hv)): tp += 1
            elif x in rd: fp += 1; fn += 1
            else: fn += 1
        elif x in rd: fp += 1
        hc = h.get(f"value_correct_{x}", "").strip().upper() == "Y"
        mtp += hc and x in mf; mfp += (not hc) and x in mf; mfn += hc and x not in mf
    fin += (h["list_finished"].strip().upper() == "Y") == (c["code_list_finished"] == "1")
    if h["stated_choice"].strip() or c["code_stated_choice"]:
        st_n += 1; st += h["stated_choice"].strip().upper() == c["code_stated_choice"]
    cat += human_cat(h, c) == c["code_category"]
print(f"traces labelled: {n}")
print(f"reader values: precision {tp/max(tp+fp,1):.3f}, recall {tp/max(tp+fn,1):.3f}")
print(f"2% matcher (correct values): precision {mtp/max(mtp+mfp,1):.3f}, recall {mtp/max(mtp+mfn,1):.3f}")
print(f"finished list agreement {fin/max(n,1):.3f}; stated choice agreement {st/max(st_n,1):.3f} (n={st_n}); gap category agreement {cat/max(n,1):.3f}")
