"""AuctionNet Table 2 split, through the paper's extractors and functions unchanged:
extract5.py (2% matcher), revision/extract8.py (text-reader segments), read_vals (t10_judge7), hybrid/classify (t10b).
S is rebuilt for AuctionNet scenarios with the same fields the eBay S has. Family = category|value.
Usage: python an_split.py <results_dir> [closed_letters_dir]. Writes results/AN/an_split[_closed].json and an_traces.pkl."""
import json, os, sys, glob, subprocess, tempfile, re
import numpy as np, pandas as pd
B = sys.argv[1]; CL = sys.argv[2] if len(sys.argv) > 2 else None
A = "/tmp/www2027/analysis"; OUT = "/tmp/www2027/results/AN"
MODELS = ["deepseek_r1_llama_8b", "phi4_mini_reasoning", "qwen3_4b", "qwen3_8b", "qwen3_14b", "granite_3_2_8b_instruct"]
SC = pd.read_csv(f"{B}/auctionnet/scenarios.csv", dtype={"scenario_id": str})
S = {}
for _, r in SC.iterrows():
    C = sorted(json.loads(r.candidates_json), key=lambda c: c["bid"]); v = float(r.value)
    bids = np.array([float(c["bid"]) for c in C]); dp = np.array([round(c["train_p"], 4) for c in C])
    tu = np.array([c["test_u"] for c in C]); tr = np.array([c["train_u"] for c in C])
    S[int(r.scenario_id)] = dict(product=r.category, v=v, K=len(C), bids=bids, dp=dp, tu=tu, eu=dp * (v - bids),
                                 fam=f"{r.category}|{v}", calc=float(bids[np.argmax(tr)]), ustar=float(tu.max()), hp=float(bids.max()))
def q_of(sid, b):
    s = S[sid]; return float(s["tu"][np.isclose(s["bids"], b)][0] / s["ustar"])
# paper functions, unchanged source
def _grab(path, name):
    src = open(path).read(); i = src.index(f"def {name}(")
    j = re.search(r"\n(?=\S)", src[i + 1:]); return src[i: i + 1 + j.start()] if j else src[i:]
ns = dict(S=S, np=np)
for p, n in ((f"{A}/revision/t10_judge7.py", "read_vals"), (f"{A}/revision/t10b_hybrid_split.py", "hybrid"), (f"{A}/revision/t10b_hybrid_split.py", "classify")):
    exec(_grab(p, n), ns)
read_vals, hybrid, classify = ns["read_vals"], ns["hybrid"], ns["classify"]
# traces -> extractor input format
tmp = tempfile.mkdtemp(); raw = f"{tmp}/robustness_extensions/server_outputs/raw"; os.makedirs(raw)
for m in MODELS:
    with open(f"{raw}/fixed_{m}_reasoning_greedy.jsonl", "w") as f:
        for rt, bud in (("forced", 768), ("unforced1024", -1)):
            for line in open(f"{B}/auctionnet/decisions/{m}_{rt}.jsonl"):
                r = json.loads(line)
                if rt == "forced" and r["budget"] != 768: continue
                f.write(json.dumps(dict(model=m, scenario_id=int(r["scenario_id"]), repeat_seed=r["order_id"], budget=bud, value=r["value"],
                                        option_order=r["option_order"], reasoning_text=r["reasoning_text"], chosen_bid=r["chosen_bid"],
                                        reasoning_tokens=r["reasoning_tokens"], hit_cap=r["hit_cap"])) + "\n")
disp = {str(i): {str(float(b)): float(p) for b, p in zip(s["bids"], s["dp"])} for i, s in S.items()}
json.dump(disp, open(f"{tmp}/disp.json", "w"))
subprocess.run([sys.executable, f"{A}/extract5.py", tmp, f"{tmp}/disp.json", f"{tmp}/first.csv"], check=True, stdout=subprocess.DEVNULL)
subprocess.run([sys.executable, f"{A}/revision/extract8.py", tmp, f"{tmp}/seg.csv", "historical"], check=True, stdout=subprocess.DEVNULL)
tf = pd.read_csv(f"{tmp}/first.csv")
seg = pd.read_csv(f"{tmp}/seg.csv").rename(columns={"seed": "repeat_seed", "n_chars": "n_chars_seg"})
tf = tf.merge(seg, on=["model", "scenario_id", "repeat_seed", "budget"], how="left")
if CL:  # closed-block letters for the 768 traces
    for f in glob.glob(f"{CL}/*.jsonl"):
        m = os.path.basename(f)[:-6]
        Aalt = {(int(r["scenario_id"]), int(r["repeat_seed"])): float(r["alt_bid"]) for r in map(json.loads, open(f))}
        z = (tf.model == m) & (tf.budget == 768)
        tf.loc[z, "chosen_bid"] = [Aalt[(i, s)] for i, s in zip(tf.loc[z, "scenario_id"], tf.loc[z, "repeat_seed"])]
tf["K"] = tf.scenario_id.map(lambda i: S[i]["K"]); tf["fp"] = tf.first_pos.map(json.loads)
tf["q_read"] = [q_of(i, b) for i, b in zip(tf.scenario_id, tf.chosen_bid)]
tf["calc"] = tf.scenario_id.map(lambda i: S[i]["calc"]); tf["qcalc"] = tf.scenario_id.map(lambda i: q_of(i, S[i]["calc"]))
tf["fam"] = tf.scenario_id.map(lambda i: S[i]["fam"]); tf["cat"] = tf.scenario_id.map(lambda i: S[i]["product"])
tf["segs"] = tf.segments.fillna("[]").map(json.loads)
info_h = {i: dict(zip(S[i]["bids"], S[i]["dp"])) for i in S}
tf["vals"] = [read_vals(sg, S[i]["v"], info_h[i], 600) for sg, i in zip(tf.segs, tf.scenario_id)]
tf["hv"] = [hybrid(vv, fp, i) for vv, fp, i in zip(tf.vals, tf.fp, tf.scenario_id)]
tf["cls"] = [classify(hv, i, K, b) for hv, i, K, b in zip(tf.hv, tf.scenario_id, tf.K, tf.chosen_bid)]
tf["gap"] = tf.qcalc - tf.q_read
tf["finished"] = [len(hv) >= K for hv, K in zip(tf.hv, tf.K)]
CATS = ["not_reached", "arith", "incomplete", "passed"]
FAMS = sorted(set(s["fam"] for s in S.values())); BYC = {}
for f in FAMS: BYC.setdefault(f.split("|")[0], []).append(f)
def fboot(stat, fams, n=2000, seed=20261119):
    rng = np.random.default_rng(seed); idx = {f: np.where(fams == f)[0] for f in FAMS}
    pt = stat(np.arange(len(fams))); bs = []
    for _ in range(n):
        ids = np.concatenate([idx[f] for c in BYC for f in rng.choice(BYC[c], len(BYC[c]))]); bs.append(stat(ids))
    bs = np.array(bs); return dict(pt=np.asarray(pt).tolist(), lo=np.percentile(bs, 2.5, axis=0).tolist(), hi=np.percentile(bs, 97.5, axis=0).tolist())
R = {}
for bud in (768, -1):
    for m in MODELS:
        z = tf[(tf.model == m) & (tf.budget == bud)].reset_index(drop=True)
        g, c = z.gap.values, z.cls.values
        R[f"{m}|{bud}"] = dict(n=len(z), finished=float(z.finished.mean()), q=float(z.q_read.mean()), qcalc=float(z.qcalc.mean()),
                               split=fboot(lambda ids: [g[ids][c[ids] == k].sum() / g[ids].sum() for k in CATS], z.fam.values))
tag = "_closed" if CL else ""
json.dump(R, open(f"{OUT}/an_split{tag}.json", "w"), indent=1)
tf.rename(columns={"segments": "segments_raw", "vals": "vals600"}).drop(columns=["segs"]).to_pickle(f"{OUT}/an_traces{tag}.pkl")
for k, r in R.items():
    s = r["split"]; print(f"{k:32s} fin {100*r['finished']:.0f} q {r['q']:.3f} | " + " ".join(f"{100*p:.0f}[{100*l:.0f},{100*h:.0f}]" for p, l, h in zip(s["pt"], s["lo"], s["hi"])))
