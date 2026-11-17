# Stdlib-only parser v2 (validated against hand labels): an option counts as evaluated when a number within
# 2% (or 0.05 absolute) of its true expected profit (v-b)*p_displayed appears in the trace, excluding numbers
# equal to a displayed quantity. Records the character offset of the first such number per option.
import json, re, csv, sys, glob
ROOT, DISP, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
disp = json.load(open(DISP))
files = sorted(glob.glob(ROOT + "/robustness_extensions/server_outputs/raw/fixed_*_reasoning_greedy.jsonl") +
               glob.glob(ROOT + "/preserved_outputs/*/historical/raw/fixed_*_reasoning_greedy.jsonl"))
NUM = re.compile(r"(?<![\d.])\d+(?:\.\d+)?")
w = csv.writer(open(OUT, "w", newline=""))
w.writerow(["model","scenario_id","repeat_seed","budget","chosen_bid","reasoning_tokens","n_chars","close","first_pos"])
for fn in files:
    for line in open(fn):
        r = json.loads(line)
        raw = r.get("reasoning_text") or ""; t = raw.replace(",", "")
        v = float(r["value"]); dp = disp[str(int(r["scenario_id"]))]
        bids = [float(b) for b in r["option_order"]]
        eu = {b: (v - b) * dp[str(b)] for b in bids}
        shown = set([v] + bids + [v - b for b in bids] + [dp[str(b)] for b in bids])
        first = {}
        for m in NUM.finditer(t):
            x = float(m.group(0))
            if any(abs(x - s) < 1e-9 for s in shown): continue
            for b in bids:
                if b in first: continue
                if abs(x - eu[b]) <= max(0.02 * abs(eu[b]), 0.05): first[b] = m.start()
        w.writerow([r["model"], r["scenario_id"], r["repeat_seed"], r["budget"], r["chosen_bid"], r.get("reasoning_tokens"), len(raw),
                    int("</think>" in raw), json.dumps({str(b): p for b, p in first.items()})])
    print(fn.split("/")[-1], flush=True)
