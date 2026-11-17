"""Parser v2 with a tolerance argument (stdlib only; runs where the raw traces are).
An option counts as evaluated when the trace states a number within TOL (relative; 0.05 absolute floor)
of its expected profit (v-b)*p_displayed, excluding numbers equal to a displayed quantity.
Records the character offset of the first such number per option.
usage: python3 extract6.py ROOT DISP_JSON TOL OUT.csv [historical|synthetic] [FLOOR]"""
import json, re, csv, sys, glob
ROOT, DISP, TOL, OUT = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
MODE = sys.argv[5] if len(sys.argv) > 5 else "historical"
FLOOR = float(sys.argv[6]) if len(sys.argv) > 6 else 0.05  # absolute floor; 0.05 suits eBay dollar values
NUM = re.compile(r"(?<![\d.])\d+(?:\.\d+)?")
def first_positions(t, v, cands):
    eu = {b: (v - b) * p for b, p in cands}
    shown = set([v] + [b for b, _ in cands] + [v - b for b, _ in cands] + [p for _, p in cands])
    first = {}
    for m in NUM.finditer(t):
        x = float(m.group(0))
        if any(abs(x - s) < 1e-9 for s in shown): continue
        for b, _ in cands:
            if b in first: continue
            if abs(x - eu[b]) <= max(TOL * abs(eu[b]), FLOOR): first[b] = m.start()
    return first
w = csv.writer(open(OUT, "w", newline=""))
if MODE == "historical":
    disp = json.load(open(DISP))
    files = sorted(glob.glob(ROOT + "/robustness_extensions/server_outputs/raw/fixed_*_reasoning_greedy.jsonl") +
                   glob.glob(ROOT + "/preserved_outputs/*/historical/raw/fixed_*_reasoning_greedy.jsonl"))
    w.writerow(["model","scenario_id","repeat_seed","budget","chosen_bid","reasoning_tokens","n_chars","close","first_pos"])
    for fn in files:
        for line in open(fn):
            r = json.loads(line); raw = r.get("reasoning_text") or ""
            dp = disp[str(int(r["scenario_id"]))]
            cands = [(float(b), dp[str(float(b))]) for b in r["option_order"]]
            fp = first_positions(raw.replace(",", ""), float(r["value"]), cands)
            w.writerow([r["model"], r["scenario_id"], r["repeat_seed"], r["budget"], r["chosen_bid"], r.get("reasoning_tokens"), len(raw),
                        int("</think>" in raw), json.dumps({str(b): p for b, p in fp.items()})])
else:
    files = sorted(glob.glob(ROOT + "/preserved_outputs/*/raw/synthetic_*_deterministic.jsonl") +
                   glob.glob(ROOT + "/preserved_outputs/*/synthetic/raw/synthetic_*_deterministic.jsonl"))
    w.writerow(["model","scenario_id","order_seed","budget","K","v","displayed","selected_bid","utility_ratio","tokens","n_chars","first_pos"])
    for fn in files:
        for line in open(fn):
            r = json.loads(line); raw = r.get("raw_reasoning") or ""
            cands = [(float(c["bid"]), round(float(c["win_probability"]), 4)) for c in r["displayed_candidates"]]  # prompt shows 4 decimals
            fp = first_positions(raw.replace(",", ""), float(r["private_value"]), cands)
            w.writerow([r["model_slug"], r["scenario_id"], r["option_order_seed"], r["reasoning_budget"], r["option_count"], r["private_value"],
                        json.dumps(cands), r["selected_bid"], r["utility_ratio"], r["realized_reasoning_tokens"], len(raw),
                        json.dumps({str(b): p for b, p in fp.items()})])
