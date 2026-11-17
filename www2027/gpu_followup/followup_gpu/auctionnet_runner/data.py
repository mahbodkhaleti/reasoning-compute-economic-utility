"""Download AuctionNet periods, verify the schema, and build per-impression competing prices and win curves.

Competing price per impression (pv) for a new bidder in the 3-slot GSP auction:
  m_any = 3rd-highest of the 48 recorded bids (beat it to win some slot) -- primary
  m_top = highest recorded bid (beat it to win the top slot)               -- robustness
Exposure and slot effects are ignored. leastWinningCost in the data is the 4th-highest bid (the price the 3rd slot
pays); we check our reconstruction against it.
Products: the competing price of an impression is the same for every advertiser, so advertiser categories cannot
give different price scales. We use terciles of the impression's mean pValue across the 48 advertisers (low/mid/
high-intent impressions), with tercile cut points from the construction periods. A diagnostic of category-level
prices is written anyway."""
from __future__ import annotations
import glob, json, os, subprocess, zipfile
import numpy as np, pandas as pd

REQUIRED = ["deliveryPeriodIndex", "advertiserNumber", "advertiserCategoryIndex", "pvIndex", "pValue", "bid", "leastWinningCost", "CPAConstraint", "timeStepIndex"]
PRODUCTS = ["low-intent impressions", "mid-intent impressions", "high-intent impressions"]

def download(cfg, log):
    raw = cfg["paths"]["raw_dir"]; os.makedirs(raw, exist_ok=True)
    for z in cfg["data"]["zips"]:
        dst = os.path.join(raw, z)
        if not os.path.exists(dst + ".done"):
            log(f"downloading {z}")
            subprocess.run(["curl", "-L", "-C", "-", "--retry", "5", "-o", dst, cfg["data"]["base_url"] + z], check=True)
            open(dst + ".done", "w").close()
        if not os.path.exists(dst + ".unzipped"):
            with zipfile.ZipFile(dst) as zf: zf.extractall(raw)
            open(dst + ".unzipped", "w").close()
            os.remove(dst)       # keep disk use low; the CSVs stay
    return sorted(glob.glob(os.path.join(raw, "**", "*.csv"), recursive=True))

def period_table(csv_path, log):
    head = pd.read_csv(csv_path, nrows=5)
    missing = [c for c in REQUIRED if c not in head.columns]
    if missing: raise ValueError(f"{csv_path}: missing columns {missing}; columns are {list(head.columns)}")
    df = pd.read_csv(csv_path, usecols=REQUIRED, dtype={"bid": "float32", "pValue": "float32", "leastWinningCost": "float32"})
    out = []
    for per, d in df.groupby("deliveryPeriodIndex"):
        d = d.sort_values(["timeStepIndex", "pvIndex", "advertiserNumber"])
        key = d["timeStepIndex"].astype(np.int64) * 10**8 + d["pvIndex"].astype(np.int64)
        counts = key.value_counts()
        n_adv = int(counts.mode().iloc[0])
        if (counts != n_adv).any(): raise ValueError(f"period {per}: impressions without exactly {n_adv} bids")
        bids = d["bid"].to_numpy().reshape(-1, n_adv); pv = d["pValue"].to_numpy().reshape(-1, n_adv)
        cats = d["advertiserCategoryIndex"].to_numpy().reshape(-1, n_adv)
        srt = -np.sort(-bids, axis=1)
        lwc = d["leastWinningCost"].to_numpy().reshape(-1, n_adv)[:, 0]
        chk = float(np.max(np.abs(np.maximum(srt[:, 3], 0.01) - np.maximum(lwc, 0.01))))
        log(f"period {per}: {len(bids)} impressions, {n_adv} bids each; max |4th bid - leastWinningCost| = {chk:.5f}")
        t = pd.DataFrame({"period": per, "m_any": srt[:, 2], "m_top": srt[:, 0], "pvalue_mean": pv.mean(axis=1)})
        for c in np.unique(cats):  # diagnostic: 3rd-highest bid among rivals outside category c
            b2 = np.where(cats == c, -np.inf, bids); t[f"m_any_excl_cat{int(c)}"] = -np.sort(-b2, axis=1)[:, 2]
        out.append(t)
    return out, chk

def build_landscapes(cfg, log):
    csvs = download(cfg, log); tabs, checks = [], []
    for c in csvs:
        t, chk = period_table(c, log); tabs += t; checks.append(chk)
    T = pd.concat(tabs, ignore_index=True)
    con = T.period.isin(cfg["data"]["construction_periods"]); held = T.period.isin(cfg["data"]["heldout_periods"])
    if con.sum() == 0 or held.sum() == 0: raise ValueError(f"periods found: {sorted(T.period.unique())}")
    cuts = np.quantile(T.loc[con, "pvalue_mean"], [1 / 3, 2 / 3])
    T["product"] = np.select([T.pvalue_mean <= cuts[0], T.pvalue_mean <= cuts[1]], PRODUCTS[:2], PRODUCTS[2])
    T["split"] = np.where(con, "train", np.where(held, "test", "unused"))
    scale = cfg["data"]["cpm_scale"]
    for col in ("m_any", "m_top"): T[col + "_cpm"] = T[col] * scale
    diag = {"pvalue_tercile_cuts": cuts.tolist(), "leastWinningCost_check_max_abs": max(checks),
            "share_construction": float(con.sum() / (con.sum() + held.sum())),
            "median_m_any_cpm_by_product": T[con].groupby("product").m_any_cpm.median().to_dict(),
            "median_m_any_cpm_excluding_each_category": {c: float(T.loc[con, c].median() * scale) for c in T.columns if c.startswith("m_any_excl")}}
    log(json.dumps(diag, indent=1))
    return T, diag
