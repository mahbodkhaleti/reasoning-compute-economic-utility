# Stdlib only. For every greedy reasoning trace: does the trace state the private value v, and does it write (v - b)
# for some listed bid b? Also records reasoning_tokens and trace length.
# Usage: python3 extract_v_mentions.py ROOT OUT.csv
import json, re, csv, sys, glob
ROOT, OUT = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(ROOT + "/robustness_extensions/server_outputs/raw/fixed_*_reasoning_greedy.jsonl") +
               glob.glob(ROOT + "/preserved_outputs/*/historical/raw/fixed_*_reasoning_greedy.jsonl"))
def num_forms(x):
    x = float(x); i = int(round(x))
    forms = {f"{i}", f"{i:,}", f"{x:.2f}", f"{i:,}.00"}
    return sorted(forms, key=len, reverse=True)
w = csv.writer(open(OUT, "w", newline=""))
w.writerow(["model", "scenario_id", "repeat_seed", "budget", "reasoning_tokens", "n_chars", "v_stated", "v_minus_b_written"])
for fn in files:
    for line in open(fn):
        r = json.loads(line)
        t = r.get("reasoning_text") or ""
        v = float(r["value"]); bids = [float(b) for b in (r.get("option_order") or [])]
        vf = "|".join(re.escape(f) for f in num_forms(v))
        v_stated = re.search(r"(?<![\d.,])\$?\s?(?:" + vf + r")(?![\d,]|\.\d)", t) is not None
        vmb = False
        for b in bids:
            bf = "|".join(re.escape(f) for f in num_forms(b))
            if re.search(r"\$?\s?(?:" + vf + r")\s*[-−–]\s*\$?\s?(?:" + bf + r")(?![\d,]|\.\d)", t):
                vmb = True; break
        w.writerow([r["model"], r["scenario_id"], r["repeat_seed"], r["budget"], r.get("reasoning_tokens"), len(t), int(v_stated), int(vmb)])
    print(fn.split("/")[-1], flush=True)
