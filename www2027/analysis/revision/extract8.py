# Stdlib-only. Raw material for the no-arithmetic text reader: for every mention of an option in a trace, the result
# numbers that follow a result cue (=, about, is, :) up to 1,000 characters on, with their positions, plus where the
# next option mention starts. Filtering (window length, skipping inputs) is done downstream.
# Usage: python3 extract8.py ROOT OUT.csv historical|synthetic
import json, re, csv, sys, glob
ROOT, OUT, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
if MODE == "historical":
    files = sorted(glob.glob(ROOT + "/robustness_extensions/server_outputs/raw/fixed_*_reasoning_greedy.jsonl") +
                   glob.glob(ROOT + "/preserved_outputs/*/historical/raw/fixed_*_reasoning_greedy.jsonl"))
else:
    files = sorted(glob.glob(ROOT + "/preserved_outputs/*/raw/synthetic_*_deterministic.jsonl") +
                   glob.glob(ROOT + "/preserved_outputs/*/synthetic/raw/synthetic_*_deterministic.jsonl"))
LAB = "ABCDEFG"
RESULT = re.compile(r"(?:=|≈|~|\babout\b|\bapproximately\b|\baround\b|\broughly\b|\bgives\b|\bis\b|:)\s*\$?\s*(-?\d*\.?\d+)")
def mentions(t, L):
    return [m.start() for m in re.finditer(r"(?:Option|option)\s+" + L + r"\b|(?:^|\n)\s*(?:[-*]\s*)?\**" + L + r"\**\s*[:)]", t)]
w = csv.writer(open(OUT, "w", newline=""))
w.writerow(["model", "scenario_id", "seed", "budget", "n_chars", "segments"])
for fn in files:
    for line in open(fn):
        r = json.loads(line)
        if MODE == "historical":
            t = (r.get("reasoning_text") or ""); bids = [float(b) for b in (r.get("option_order") or [])]
            key = [r["model"], r["scenario_id"], r["repeat_seed"], r["budget"]]
        else:
            t = (r.get("raw_reasoning") or ""); bids = [float(c["bid"]) for c in r["displayed_candidates"]]
            key = [r["model_slug"], r["scenario_id"], r["option_order_seed"], r["reasoning_budget"]]
        t = t.replace(",", "")
        starts = sorted((p, j) for j in range(len(bids)) for p in mentions(t, LAB[j]))
        segs = []
        for k, (p, j) in enumerate(starts):
            nxt = starts[k + 1][0] if k + 1 < len(starts) else len(t)
            nums = [[float(m.group(1)), m.start()] for m in RESULT.finditer(t, p, min(len(t), p + 1000))]
            segs.append([bids[j], p, nxt, nums])
        w.writerow(key + [len(t), json.dumps(segs)])
    print(fn.split("/")[-1], flush=True)
