"""Shared setup for the v25 revision analyses (CPU only, saved outputs). Reuses the paper's code path through
analysis/cpu_revisions/cr_common.py (2% matcher, text reader, hybrid written values, gap categories, family
bootstrap). Outputs go to results/rev25/."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "/tmp/www2027/analysis/cpu_revisions")
from cr_common import *          # noqa: F401,F403

OUT25 = "/tmp/www2027/results/rev25"
os.makedirs(OUT25, exist_ok=True)
CATS = ["not_reached", "arith", "incomplete", "passed"]
g = d[d.decoding == "greedy"]
q0 = g[g.budget == 0].groupby("model").test_utility_ratio.mean()
def dump25(name, obj):
    json.dump(obj, open(f"{OUT25}/{name}.json", "w"), indent=1, default=float)
def fboot(fn, fams, n=2000, seed=20261112):
    """Paper's family bootstrap on index arrays: fn(ids) -> scalar or vector."""
    r = np.random.default_rng(seed); pt = np.atleast_1d(fn(np.arange(len(fams))))
    bs = np.array([np.atleast_1d(fn(family_boot_ids(fams, r))) for _ in range(n)])
    return [dict(pt=float(pt[j]), ci=[float(np.nanpercentile(bs[:, j], 2.5)), float(np.nanpercentile(bs[:, j], 97.5))]) for j in range(len(pt))]
def hboot_ids(fams, r):
    """Hierarchical bootstrap: resample the 3 products with replacement, then the 18 value families within each drawn product."""
    idx = {f: np.where(fams == f)[0] for f in FAM}
    prods = r.choice(PRODUCTS, len(PRODUCTS), replace=True)
    return np.concatenate([idx[f] for p in prods for f in r.choice(FAM_BY_PROD[p], len(FAM_BY_PROD[p]), replace=True)])
def hboot(fn, fams, n=2000, seed=20261113):
    r = np.random.default_rng(seed); pt = np.atleast_1d(fn(np.arange(len(fams))))
    bs = np.array([np.atleast_1d(fn(hboot_ids(fams, r))) for _ in range(n)])
    return [dict(pt=float(pt[j]), ci=[float(np.nanpercentile(bs[:, j], 2.5)), float(np.nanpercentile(bs[:, j], 97.5))]) for j in range(len(pt))]
