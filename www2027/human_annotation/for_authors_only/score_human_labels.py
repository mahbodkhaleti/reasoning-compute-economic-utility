"""Scores filled human annotation workbooks against the machine labels and the LLM annotators.
usage: python3 score_human_labels.py answer_key.json filled1.xlsx [filled2.xlsx ...]
Also accepts a pair of CSVs per annotator: option_labels_<X>.csv,trace_labels_<X>.csv (comma-joined).
Writes human_agreement.json next to the first workbook and prints a summary.
Needs: pandas, openpyxl."""
import json, sys, os, itertools
import numpy as np, pandas as pd

def kappa(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool); po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0
def yn(x): return str(x).strip().upper().startswith("Y")
def num(x):
    try: return float(str(x).replace("$", "").replace(",", "").strip())
    except Exception: return None
def letters(x):
    s = str(x).strip().upper()
    return set() if s in ("", "NAN", "NONE") else {c for c in s.replace(" ", "").split(",") if c}

def load(path):
    if path.endswith(".xlsx"):
        o = pd.read_excel(path, sheet_name="Option labels"); t = pd.read_excel(path, sheet_name="Trace labels")
    else:
        po, pt = path.split(","); o = pd.read_csv(po); t = pd.read_csv(pt)
    o = o[o.trace_id.astype(str).str.match(r"T\d\d") & (o.trace_id != "T00")]
    t = t[t.trace_id.astype(str).str.match(r"T\d\d") & (t.trace_id != "T00")]
    return o, t

key = json.load(open(sys.argv[1])); files = sys.argv[2:]
K = {k["trace_id"]: k for k in key}
H = {os.path.basename(f).split(".")[0]: load(f) for f in files}
res = {"annotators": list(H), "per_annotator": {}, "between_annotators": {}}
for name, (o, t) in H.items():
    rows = []
    for r in o.itertuples():
        k = K[r.trace_id]; opt = next(x for x in k["options"] if x["letter"] == r.option)
        rows.append(dict(tid=r.trace_id, opt=r.option, h=yn(r._5), hv=num(r.written_value), parser=opt["parser_evaluated"],
                         reader=opt["reader_value"] is not None, rv=opt["reader_value"], llm=opt["llm_written"], lv=opt["llm_value"], eu=opt["true_eu"]))
    d = pd.DataFrame(rows)
    filled = o["written (Y/N)"].notna().sum()
    out = dict(option_rows=int(len(d)), option_rows_filled=int(filled))
    for m in ("parser", "reader", "llm"):
        out[f"written_vs_{m}"] = dict(agreement=float((d.h == d[m]).mean()), kappa=kappa(d.h, d[m]),
                                      precision_of_machine=float(d.h[d[m]].mean()) if d[m].any() else None,
                                      recall_of_machine=float(d[m][d.h].mean()) if d.h.any() else None)
    both = d[d.h & d.reader & d.hv.notna()]
    out["reader_value_matches_human"] = float(np.mean([abs(a - b) <= 0.02 * abs(a) + 0.05 for a, b in zip(both.hv, both.rv)])) if len(both) else None
    w = d[d.h & d.hv.notna()]
    out["human_values_off_true_by_2pct"] = float(np.mean([abs(a - b) > 0.02 * abs(b) + 0.05 for a, b in zip(w.hv, w.eu)])) if len(w) else None
    lw = d[d.h & d.llm & d.hv.notna() & d.lv.notna()]
    out["llm_value_matches_human"] = float(np.mean([abs(a - b) <= 0.02 * abs(a) + 0.05 for a, b in zip(lw.hv, lw.lv)])) if len(lw) else None
    tr = []
    for r in t.itertuples():
        k = K[r.trace_id]
        tr.append(dict(fin=yn(r._3), parser_complete=k["parser_complete"], best=letters(r.trace_best), hyb=k["hybrid_best_letter"]))
    tt = pd.DataFrame(tr)
    out["finished_vs_parser"] = dict(agreement=float((tt.fin == tt.parser_complete).mean()), kappa=kappa(tt.fin, tt.parser_complete))
    out["trace_best_matches_machine"] = float(np.mean([(h == set() and m == "NONE") or (m in h) for h, m in zip(tt.best, tt.hyb)]))
    res["per_annotator"][name] = out
    H[name] = (o, t, d, tt)
for a, b in itertools.combinations(list(H), 2):
    da, db = H[a][2], H[b][2]; m = da.merge(db, on=["tid", "opt"], suffixes=("_a", "_b"))
    ta, tb = H[a][3], H[b][3]
    res["between_annotators"][f"{a}~{b}"] = dict(written_agreement=float((m.h_a == m.h_b).mean()), written_kappa=kappa(m.h_a, m.h_b),
                                                trace_best_agreement=float(np.mean([x == y for x, y in zip(ta.best, tb.best)])),
                                                finished_kappa=kappa(ta.fin, tb.fin))
outp = os.path.join(os.path.dirname(os.path.abspath(files[0].split(",")[0])), "human_agreement.json")
json.dump(res, open(outp, "w"), indent=1); print(json.dumps(res, indent=1)); print("written to", outp)
