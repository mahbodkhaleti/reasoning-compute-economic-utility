# Stdlib-only: parse synthetic traces (displayed candidates carried in each record).
import json, re, csv, sys, glob
ROOT = sys.argv[1]
OPS = r"\s*(?:\*|×|x|·|\\times|\\cdot)\s*"
RES = r"\s*(?:=|≈|is|equals|gives)\s*(?:about\s*|approximately\s*|roughly\s*|around\s*|~\s*)?\$?\s*(-?\d+(?:\.\d+)?)"
def nf(x):
    x = float(x); return [str(int(x))] if x == int(x) else ["%g" % x]
w = csv.writer(open("synthetic_matches.csv", "w", newline=""))
w.writerow(["model","scenario_id","order_seed","budget","K","v","displayed","selected_bid","utility_ratio","tokens","n_chars","matches"])
for fn in sorted(glob.glob(ROOT + "/preserved_outputs/*/raw/synthetic_*_deterministic.jsonl") + glob.glob(ROOT + "/preserved_outputs/*/synthetic/raw/synthetic_*_deterministic.jsonl")):
    for line in open(fn):
        r = json.loads(line)
        t = (r.get("raw_reasoning") or "").replace(",", ""); v = float(r["private_value"])
        disp = [(float(c["bid"]), float(c["win_probability"])) for c in r["displayed_candidates"]]
        ms = []
        for b, p in disp:
            for ds in nf(v - b):
                for pat in (r"(?<![\d.])" + re.escape(ds) + r"(?:\.0+)?" + OPS + r"(0?\.\d+)" + RES,
                            r"(?<![\d])(0?\.\d+)" + OPS + r"\(?\s*" + re.escape(ds) + r"(?:\.0+)?(?![\d.])\s*\)?" + RES):
                    for m in re.finditer(pat, t):
                        ms.append([b, float(m.group(1)), float(m.group(2)), m.start()])
        w.writerow([r["model_slug"], r["scenario_id"], r["option_order_seed"], r["reasoning_budget"], r["option_count"], v,
                    json.dumps(disp), r["selected_bid"], r["utility_ratio"], r["realized_reasoning_tokens"], len(r.get("raw_reasoning") or ""), json.dumps(ms)])
    print(fn.split("/")[-1], flush=True)
