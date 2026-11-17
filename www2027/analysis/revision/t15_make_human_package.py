"""Builds the human-annotation package: 50 traces drawn (stratified by model and budget, seed 20261107) from the 150
traces the LLM annotators labeled, so human labels can be compared with the parser, the text reader and the LLM labels.
Annotator files hide the model, budget and all machine labels. The answer key and scorer go in a separate folder."""
import json, random, html, os, shutil, csv, numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
_src = open("t10b_hybrid_split.py").read()
exec(_src[:_src.index('R = {"forced768"')])
OUT = "/tmp/www2027/human_annotation"
if os.path.exists(OUT): shutil.rmtree(OUT)
PK = f"{OUT}/annotator_package"; AU = f"{OUT}/for_authors_only"
os.makedirs(PK); os.makedirs(AU)
vs = json.load(open("../validation_sample.json")); L = {}
for p in (0, 1):
    for x in json.load(open(f"/tmp/www2027/judge/label/labels_{p}.json")): L[x["item"]] = x
rnd = random.Random(20261107)
by = {}
for k, r in enumerate(vs): by.setdefault(r["model"], []).append(k)
quota = dict(zip(sorted(by), [9, 9, 8, 8, 8, 8]))
pick = []
for m, ks in sorted(by.items()):
    ks = sorted(ks, key=lambda k: (vs[k]["budget"], rnd.random()))
    step = len(ks) / quota[m]; pick += [ks[int(i * step)] for i in range(quota[m])]
rnd.shuffle(pick)
key = tf.set_index(["model", "scenario_id", "repeat_seed", "budget"])
traces, answer = [], []
for t, k in enumerate(pick, 1):
    r = vs[k]; sid = int(r["scenario_id"]); s = S[sid]; tid = f"T{t:02d}"
    opts = []
    for j, b in enumerate(r["option_order"]):
        b = float(b); p_ = float(s["dp"][np.isclose(s["bids"], b)][0]); opts.append(("ABCDEFG"[j], b, p_))
    traces.append(dict(tid=tid, value=float(r["value"]), opts=opts, text=r["reasoning_text"]))
    row = key.loc[(r["model"], sid, int(r["repeat_seed"]), int(r["budget"]))]
    row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
    lab = {o["label"]: o for o in L[k]["options"]}
    hv = row.hv
    hb = max(hv.items(), key=lambda kv: kv[1][0])[0] if hv else None
    answer.append(dict(trace_id=tid, model=r["model"], scenario_id=sid, repeat_seed=int(r["repeat_seed"]), budget=int(r["budget"]),
                       validation_item=k, private_value=float(r["value"]), calculator_bid=float(s["calc"]),
                       model_answer_bid=float(row.chosen_bid),
                       hybrid_best_letter=next((L_ for L_, b, _ in opts if hb is not None and np.isclose(b, hb)), "NONE"),
                       parser_complete=bool(row.complete),
                       options=[dict(letter=L_, bid=b, p=p_, true_eu=float(s["eu"][np.isclose(s["bids"], b)][0]),
                                     parser_evaluated=any(np.isclose(float(x), b) for x in row.fp),
                                     reader_value=(row.vals[b][0] if b in row.vals else None),
                                     llm_written=bool(lab[L_]["stated"]), llm_value=lab[L_]["value"]) for L_, b, p_ in opts]))
json.dump(answer, open(f"{AU}/answer_key.json", "w"), indent=1)

# ---------------- workbook
F = "Arial"; yellow = PatternFill("solid", fgColor="FFF2CC"); grey = PatternFill("solid", fgColor="E7E6E6")
hdr_font = Font(name=F, bold=True); body = Font(name=F); thin = Side(style="thin", color="BFBFBF")
wb = Workbook(); ws = wb.active; ws.title = "Read me"
readme = [
 "Trace annotation: 50 reasoning traces from an auction-bidding task",
 "",
 "What you are doing: a language model was given a private value and 5 or 7 candidate bids, each with a win probability, and asked to pick the bid with the highest expected profit = (private value - bid) x win probability. You will record what the model WROTE in its reasoning, not whether it was right.",
 "",
 "Files: open traces.html in a browser to read each trace (same text is in the 'Traces' sheet). Record your labels in the two yellow-column sheets: 'Option labels' (one row per option) and 'Trace labels' (one row per trace).",
 "",
 "Only edit the yellow cells. Do not add, delete or sort rows. Row 2 of each labels sheet is an EXAMPLE (trace T00) showing the format; leave it as it is.",
 "",
 "Option labels, for each option of each trace:",
 "  written (Y/N): Y if the trace writes a number that it presents as THIS option's expected profit (the result of the multiplication), even if the number is wrong or rounded. N if it never reaches a result for this option (only the formula, only the difference value - bid, or the text is cut off before the result).",
 "  written_value: the number as written, without $ or commas (e.g. 1053.04). If the trace writes several values for the same option, use the LAST one. Leave blank when written = N.",
 "",
 "Trace labels, for each trace:",
 "  list_finished (Y/N): Y if every option has written = Y.",
 "  trace_best: the letter of the option with the LARGEST written value, judged only from the numbers the trace wrote (do not recompute). NONE if no option has a written value. For a tie write both letters, e.g. A,C.",
 "  stated_choice: if the trace itself says which option or bid it will pick (e.g. 'so I choose option C' or 'the best bid is $250'), that option's letter. NONE if it never says (many traces are cut off).",
 "  minutes (optional): roughly how long the trace took you.",
 "  notes (optional): anything unclear.",
 "",
 "Rules: do not use a calculator to check the model's arithmetic, and do not use ChatGPT, Claude or any other AI tool. Work alone and do not compare answers with another annotator until both of you are done. Do not look anything up about the models or the paper.",
 "",
 "Time: about 3-5 minutes per trace, so 3-4 hours in total. You can stop and continue later; save often.",
 "",
 "When finished: save the file as annotation_workbook_<your initials>.xlsx and send it back.",
]
for i, line in enumerate(readme, 1):
    c = ws.cell(row=i, column=1, value=line); c.font = Font(name=F, bold=(i == 1), size=13 if i == 1 else 11); c.alignment = Alignment(wrap_text=True, vertical="top")
ws.column_dimensions["A"].width = 120
ws.cell(row=len(readme) + 2, column=1, value="Progress").font = hdr_font
ws.cell(row=len(readme) + 3, column=1, value="=\"Option rows done: \"&COUNTA('Option labels'!E3:E1000)&\" of \"&(COUNTA('Option labels'!A3:A1000))").font = body
ws.cell(row=len(readme) + 4, column=1, value="=\"Trace rows done: \"&COUNTA('Trace labels'!C3:C1000)&\" of \"&(COUNTA('Trace labels'!A3:A1000))").font = body

wt = wb.create_sheet("Traces")
heads = ["trace_id", "private_value", "options (letter: bid, win probability)", "trace text"]
for j, h in enumerate(heads, 1):
    c = wt.cell(row=1, column=j, value=h); c.font = hdr_font; c.fill = grey
for i, t in enumerate(traces, 2):
    vals = [t["tid"], t["value"], "\n".join(f"{L_}: ${b:g}, p={p_:.4f}" for L_, b, p_ in t["opts"]), t["text"]]
    for j, v in enumerate(vals, 1):
        c = wt.cell(row=i, column=j, value=v); c.font = body; c.alignment = Alignment(wrap_text=True, vertical="top")
for col, w in zip("ABCD", [9, 13, 28, 150]): wt.column_dimensions[col].width = w
wt.freeze_panes = "A2"

yn = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
wo = wb.create_sheet("Option labels")
oh = ["trace_id", "option", "bid ($)", "win probability", "written (Y/N)", "written_value", "notes"]
for j, h in enumerate(oh, 1):
    c = wo.cell(row=1, column=j, value=h); c.font = hdr_font; c.fill = yellow if j in (5, 6, 7) else grey
ex = [("T00", "A", 870, 0.6421, "Y", 1053.04, "EXAMPLE row"), ]
r_ = 2
for e in ex:
    for j, v in enumerate(e, 1):
        c = wo.cell(row=r_, column=j, value=v); c.font = Font(name=F, italic=True, color="808080")
r_ = 3
for t in traces:
    for L_, b, p_ in t["opts"]:
        for j, v in enumerate([t["tid"], L_, b, p_], 1):
            c = wo.cell(row=r_, column=j, value=v); c.font = body
        for j in (5, 6, 7):
            c = wo.cell(row=r_, column=j); c.fill = yellow; c.font = body; c.border = Border(left=thin, right=thin, top=thin, bottom=thin)
        r_ += 1
wo.add_data_validation(yn); yn.add(f"E3:E{r_ - 1}")
wo.cell(row=2, column=4).number_format = "0.0000"
for rr in range(3, r_): wo.cell(row=rr, column=4).number_format = "0.0000"
for col, w in zip("ABCDEFG", [9, 8, 10, 14, 13, 14, 40]): wo.column_dimensions[col].width = w
wo.freeze_panes = "A3"

wl = wb.create_sheet("Trace labels")
th = ["trace_id", "number of options", "list_finished (Y/N)", "trace_best", "stated_choice", "minutes", "notes"]
for j, h in enumerate(th, 1):
    c = wl.cell(row=1, column=j, value=h); c.font = hdr_font; c.fill = yellow if j >= 3 else grey
for j, v in enumerate(["T00", 5, "N", "A", "NONE", 4, "EXAMPLE row: cut off before option E"], 1):
    c = wl.cell(row=2, column=j, value=v); c.font = Font(name=F, italic=True, color="808080")
yn2 = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
for i, t in enumerate(traces, 3):
    wl.cell(row=i, column=1, value=t["tid"]).font = body; wl.cell(row=i, column=2, value=len(t["opts"])).font = body
    for j in range(3, 8):
        c = wl.cell(row=i, column=j); c.fill = yellow; c.font = body; c.border = Border(left=thin, right=thin, top=thin, bottom=thin)
wl.add_data_validation(yn2); yn2.add(f"C3:C{len(traces) + 2}")
for col, w in zip("ABCDEFG", [9, 18, 18, 12, 14, 9, 50]): wl.column_dimensions[col].width = w
wl.freeze_panes = "A3"
wb.save(f"{PK}/annotation_workbook.xlsx")

# ---------------- CSV fallback
with open(f"{PK}/option_labels.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(oh)
    for t in traces:
        for L_, b, p_ in t["opts"]: w.writerow([t["tid"], L_, b, f"{p_:.4f}", "", "", ""])
with open(f"{PK}/trace_labels.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(th)
    for t in traces: w.writerow([t["tid"], len(t["opts"]), "", "", "", "", ""])

# ---------------- HTML reader
parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>Traces for annotation</title><style>",
         "body{font-family:Arial,sans-serif;max-width:900px;margin:2em auto;line-height:1.45;color:#222}",
         "h2{border-top:2px solid #999;padding-top:1em}table{border-collapse:collapse;margin:.5em 0}td,th{border:1px solid #bbb;padding:3px 10px}",
         "pre{white-space:pre-wrap;background:#f7f7f7;padding:1em;border:1px solid #ddd;font-family:Arial,sans-serif;font-size:14px}",
         "nav a{margin-right:.6em}@media print{h2{page-break-before:always}}</style></head><body>",
         "<h1>Traces for annotation (50)</h1><p>Read each trace and fill the matching rows of <b>annotation_workbook.xlsx</b>. Instructions are in INSTRUCTIONS.md and in the workbook's 'Read me' sheet.</p><nav>"]
parts += [f"<a href='#{t['tid']}'>{t['tid']}</a>" for t in traces] + ["</nav>"]
for t in traces:
    parts.append(f"<h2 id='{t['tid']}'>{t['tid']}</h2><p>Private value: <b>${t['value']:g}</b></p><table><tr><th>Option</th><th>Bid</th><th>Win probability</th></tr>")
    parts += [f"<tr><td>{L_}</td><td>${b:g}</td><td>{p_:.4f}</td></tr>" for L_, b, p_ in t["opts"]]
    parts.append(f"</table><pre>{html.escape(t['text'])}</pre>")
parts.append("</body></html>")
open(f"{PK}/traces.html", "w", encoding="utf-8").write("".join(parts))
print(len(traces), "traces;", sum(len(t["opts"]) for t in traces), "option rows")
