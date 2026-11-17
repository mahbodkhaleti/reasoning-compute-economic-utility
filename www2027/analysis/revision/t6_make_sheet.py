"""Builds the human annotation sheet from the raw traces (stdlib only; run where the raw traces are).
usage: python3 t6_make_sheet.py ROOT disp.json t6_annotation_keys.csv OUT.csv"""
import json, csv, sys, glob
ROOT, DISP, KEYS, OUT = sys.argv[1:5]
disp = json.load(open(DISP))
keys = {(r["model"], r["scenario_id"].zfill(12) if False else r["scenario_id"], int(r["repeat_seed"]), int(r["budget"])): r["sheet_id"] for r in csv.DictReader(open(KEYS))}
files = sorted(glob.glob(ROOT + "/robustness_extensions/server_outputs/raw/fixed_*_reasoning_greedy.jsonl") +
               glob.glob(ROOT + "/preserved_outputs/*/historical/raw/fixed_*_reasoning_greedy.jsonl"))
rows = {}
for fn in files:
    for line in open(fn):
        r = json.loads(line); k = (r["model"], str(int(r["scenario_id"])), int(r["repeat_seed"]), int(r["budget"]))
        if k in keys:
            dp = disp[str(int(r["scenario_id"]))]
            opts = "; ".join(f"{'ABCDEFG'[j]}: bid ${float(b):g}, p={dp[str(float(b))]:.4f}" for j, b in enumerate(r["option_order"]))
            rows[int(keys[k])] = [keys[k], r["model"], r["scenario_id"], r["repeat_seed"], r["budget"], r["value"], len(r["option_order"]), opts, r["reasoning_text"]]
hdr = ["sheet_id", "model", "scenario_id", "order_seed", "budget", "private_value", "K", "options", "trace"]
for L in "ABCDEFG": hdr += [f"evaluated_{L}", f"value_{L}"]
hdr += ["notes"]
w = csv.writer(open(OUT, "w", newline="", encoding="utf-8")); w.writerow(hdr)
for i in sorted(rows):
    r = rows[i]; w.writerow(r + [""] * (2 * 7) + [""])
print(len(rows), "rows written")
