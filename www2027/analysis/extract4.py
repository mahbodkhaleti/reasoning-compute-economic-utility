# Stdlib-only. Per-option values a trace computed, coverage, costs; plus synthetic summary.
import json, re, csv, sys, glob
ROOT = sys.argv[1]
files = sorted(glob.glob(ROOT + "/robustness_extensions/server_outputs/raw/fixed_*_reasoning_greedy.jsonl") +
               glob.glob(ROOT + "/preserved_outputs/*/historical/raw/fixed_*_reasoning_greedy.jsonl"))
def nf(x):
    x = float(x); return [str(int(x))] if x == int(x) else ["%g" % x]
OPS = r"\s*(?:\*|×|x|·|\\times|\\cdot)\s*"
RES = r"\s*(?:=|≈|is|equals|gives)\s*(?:about\s*|approximately\s*|roughly\s*|around\s*|~\s*)?\$?\s*(-?\d+(?:\.\d+)?)"
LAB = "ABCDEFG"
w = csv.writer(open("trace_matches.csv", "w", newline=""))
w.writerow(["model","scenario_id","repeat_seed","budget","option_order","chosen_bid","reasoning_tokens","elapsed","readout_elapsed",
            "energy","readout_energy","close","values","evaluated","n_chars"])
for fn in files:
    for line in open(fn):
        r = json.loads(line)
        t = (r.get("reasoning_text") or "").replace(",", "")
        bids = [float(b) for b in (r.get("option_order") or [])]; v = float(r["value"])
        vals, ev, matches = {}, set(), []
        for j, b in enumerate(bids):
            D = v - b; last = None; lastpos = -1
            for vs in nf(v):
                for bs in nf(b):
                    if re.search(r"(?<![\d.])" + re.escape(vs) + r"\s*(?:-|−|–)\s*\$?" + re.escape(bs) + r"(?![\d])", t): ev.add(b)
            for ds in nf(D):
                for pat in (r"(?<![\d.])" + re.escape(ds) + r"(?:\.0+)?" + OPS + r"0?\.\d+" + RES,
                            r"(?<![\d])0?\.\d+" + OPS + r"\(?\s*" + re.escape(ds) + r"(?:\.0+)?(?![\d.])\s*\)?" + RES):
                    for m in re.finditer(pat, t):
                        ev.add(b)
                        pm = re.search(r"0?\.\d+", m.group(0)[len(ds):] if m.group(0).startswith(ds) else m.group(0))
                        matches.append([b, float(pm.group(0)) if pm else None, float(m.group(1)), m.start(), "prod"])
            L = LAB[j]
            for m in re.finditer(r"(?:Option\s+|option\s+|\b)" + L + r"\b\s*(?:is|=|≈|gives|has|yields)\s*(?:an?\s+expected\s+profit\s+of\s*)?(?:about\s*|approximately\s*|roughly\s*|around\s*|~\s*)?\$\s*(\d+(?:\.\d+)?)", t):
                ev.add(b)
                matches.append([b, None, float(m.group(1)), m.start(), "label"])
        raw = r.get("reasoning_text") or ""
        w.writerow([r["model"], r["scenario_id"], r["repeat_seed"], r["budget"], json.dumps(bids), r["chosen_bid"], r.get("reasoning_tokens"),
                    r.get("elapsed_seconds"), r.get("constrained_readout_elapsed_seconds"), r.get("energy_joules"), r.get("constrained_readout_energy_joules"),
                    int("</think>" in raw), json.dumps(matches), json.dumps(sorted(ev)), len(raw)])
    print(fn.split("/")[-1], flush=True)
