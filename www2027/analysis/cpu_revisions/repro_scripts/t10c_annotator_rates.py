"""Per-model share of annotator-recorded stated values that are off the true expected profit by more than 2% (+0.05)."""
import json, numpy as np
from collections import defaultdict
from common_rev import S, OUTDIR
vs = json.load(open("../validation_sample.json")); L = {}
for p in (0, 1):
    for x in json.load(open(f"/tmp/www2027/judge/label/labels_{p}.json")): L[x["item"]] = x
n = defaultdict(int); e = defaultdict(int)
for k, r in enumerate(vs):
    s = S[int(r["scenario_id"])]; lab = {o["label"]: o for o in L[k]["options"]}
    for j, b in enumerate(r["option_order"]):
        o = lab["ABCDEFG"[j]]
        if o["stated"] and o["value"] is not None:
            tv = s["eu"][np.isclose(s["bids"], float(b))][0]; n[r["model"]] += 1; e[r["model"]] += int(abs(o["value"] - tv) > 0.02 * abs(tv) + 0.05)
R = {m: dict(n=n[m], error_rate=e[m] / n[m]) for m in n}
json.dump(R, open(f"{OUTDIR}/t10c_annotator_rates.json", "w"), indent=1); print(R)
