"""Shared loaders for the revision analyses (analysis/revision/).
Inputs (in analysis/): scenarios.csv, saved_records.csv.gz, parser outputs first_tol*.csv.
"""
import json, os
import numpy as np, pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUTDIR, exist_ok=True)
MODELS = ["deepseek_r1_llama_8b", "phi4_mini_reasoning", "qwen3_4b", "qwen3_8b", "qwen3_14b", "granite_3_2_8b_instruct"]
LAB = {"deepseek_r1_llama_8b": "DeepSeek-8B", "phi4_mini_reasoning": "Phi-mini", "qwen3_4b": "Qwen3-4B",
       "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B", "granite_3_2_8b_instruct": "Granite-8B"}

sc = pd.read_csv(os.path.join(BASE, "scenarios.csv"))
S = {}
for r in sc.itertuples():
    c = json.loads(r.candidates_json)
    bids = np.array([x["bid"] for x in c], float)
    dp = np.round(np.array([x["train_p"] for x in c]), 4)
    tu = np.array([x["test_u"] for x in c])
    eu = dp * (r.value - bids)
    srt = np.sort(eu)[::-1]
    S[r.scenario_id] = dict(product=r.category, v=float(r.value), K=r.option_count, bids=bids, dp=dp, tu=tu, eu=eu,
                            fam=f"{r.category}|{r.value}", calc=bids[np.argmax(eu)], ustar=tu.max(),
                            gap12=(srt[0] - srt[1]) / srt[0], hp=bids[np.argmax(dp)])
FAM = sorted({s["fam"] for s in S.values()}); PRODUCTS = sorted({s["product"] for s in S.values()})
FAM_BY_PROD = {p: [f for f in FAM if f.startswith(p)] for p in PRODUCTS}

def q_of(sid, b):
    s = S[sid]; return float(s["tu"][np.isclose(s["bids"], b)][0] / s["ustar"])

def records():
    return pd.read_csv(os.path.join(BASE, "saved_records.csv.gz"))

def family_boot_ids(fams, rng):
    """Product-stratified family bootstrap: resample 18 value families within each product."""
    idx = {f: np.where(fams == f)[0] for f in FAM}
    return np.concatenate([idx[f] for p in PRODUCTS for f in rng.choice(FAM_BY_PROD[p], len(FAM_BY_PROD[p]), replace=True)])

def load_parse(tol):
    """Per-trace parse at a given tolerance, joined with readouts."""
    path = os.path.join(BASE, "trace_first_v2.csv" if tol == 0.02 else f"first_tol{tol}.csv")
    tf = pd.read_csv(path)
    tf["K"] = tf.scenario_id.map(lambda i: S[i]["K"])
    tf["fp"] = tf.first_pos.map(json.loads)
    tf["n_eval"] = tf.fp.map(len)
    tf["complete"] = tf.n_eval == tf.K
    tok = tf.reasoning_tokens.fillna(0).values; nch = np.maximum(tf.n_chars.values, 1)
    tf["comp_tok"] = [max(fp.values()) / n * t if c else np.nan for fp, n, t, c in zip(tf.fp, nch, tok, tf.complete)]
    tf["q_read"] = [q_of(i, b) for i, b in zip(tf.scenario_id, tf.chosen_bid)]
    tf["calc"] = tf.scenario_id.map(lambda i: S[i]["calc"])
    tf["qcalc"] = tf.scenario_id.map(lambda i: q_of(i, S[i]["calc"]))
    def best_eval(i, fp):
        if not len(fp): return np.nan
        s = S[i]; b = np.array([float(x) for x in fp])
        u = np.array([s["eu"][np.isclose(s["bids"], x)][0] for x in b]); return float(b[np.argmax(u)])
    tf["best_eval_bid"] = [best_eval(i, fp) for i, fp in zip(tf.scenario_id, tf.fp)]
    tf["q_be"] = [q_of(i, b) if b == b else q for i, b, q in zip(tf.scenario_id, tf.best_eval_bid, tf.q_read)]
    tf["fam"] = tf.scenario_id.map(lambda i: S[i]["fam"])
    return tf
