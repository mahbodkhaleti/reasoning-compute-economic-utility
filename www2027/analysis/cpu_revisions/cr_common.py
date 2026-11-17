"""Shared setup for the CPU revision analyses. Reuses the paper's code paths unchanged: it executes the head of
analysis/revision/t10b_hybrid_split.py (which runs common_rev, the 2% matcher via load_parse(0.02), the text reader
from extract8 segments, the hybrid written values and the four-way gap categories). Outputs go to results/cpu_revisions/."""
import os, sys, json
import numpy as np, pandas as pd
REV = "/tmp/www2027/analysis/revision"
OUT = "/tmp/www2027/results/cpu_revisions"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, "/tmp/www2027/analysis/cpu_revisions/repro_scripts")   # common_rev with OUTDIR redirected
_cwd = os.getcwd(); os.chdir(REV)
_src = open("t10b_hybrid_split.py").read()
exec(_src[:_src.index('R = {"forced768"')])
_tm = pd.read_csv("../trace_matches.csv", usecols=["model", "scenario_id", "repeat_seed", "budget", "option_order"])
tf = tf.merge(_tm, on=["model", "scenario_id", "repeat_seed", "budget"], how="left")
tf["order_list"] = tf.option_order.map(lambda x: [float(b) for b in json.loads(x)])
_rec = d[d.decoding == "greedy"][["model", "scenario_id", "repeat_seed", "budget", "freeform_chosen_bid", "freeform_valid", "test_utility"]]
tf = tf.merge(_rec, on=["model", "scenario_id", "repeat_seed", "budget"], how="left")
SC = pd.read_csv("../scenarios.csv").set_index("scenario_id")
HELD = {i: {float(x["bid"]): x for x in json.loads(r.candidates_json)} for i, r in SC.iterrows()}
AU = pd.read_csv("../auction_splits.csv")
def _prod(item):
    it = item.lower(); return next(p for p in PRODUCTS if p.split()[0].lower() in it)
AU["product"] = AU["item"].map(_prod)
TEST_PRICES = {p: AU[(AU["product"] == p) & (AU.split != "train")].final_price.values for p in PRODUCTS}
os.chdir(_cwd)
def boot_ci(fn, frame, n=2000, seed=20261108):
    """Family bootstrap used throughout the paper: resample the 18 value families within each product."""
    rng_ = np.random.default_rng(seed); frame = frame.reset_index(drop=True)
    pt = fn(frame); bs = np.array([fn(frame.iloc[family_boot_ids(frame.fam.values, rng_)]) for _ in range(n)])
    pt = np.atleast_1d(pt); bs = bs.reshape(n, -1)
    return [dict(pt=float(pt[j]), ci=[float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]) for j in range(len(pt))]
def dump(name, obj):
    json.dump(obj, open(f"{OUT}/{name}.json", "w"), indent=1, default=float)
