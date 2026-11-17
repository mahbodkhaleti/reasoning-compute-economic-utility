"""AuctionNet closed-block readout (R1) and unforced 4,096 (A3). Same gap and family bootstrap as an_p0_analysis.py.
Writes results/AN/an_overnight_summary.json."""
import json, numpy as np, pandas as pd
B = "/tmp/www2027/results/AN/ov/results/auctionnet"
from an_common import build_S
S = build_S("/tmp/www2027/results/AN/ov/results")
MODELS = ["deepseek_r1_llama_8b", "phi4_mini_reasoning", "qwen3_4b", "qwen3_8b", "qwen3_14b", "granite_3_2_8b_instruct"]
q_of = lambda i, b: float(S[i]["tu"][np.isclose(S[i]["bids"], b)][0] / S[i]["ustar"])
CATS = sorted({s["product"] for s in S.values()}); FAMS = sorted({s["fam"] for s in S.values()})
BYC = {c: [f for f in FAMS if f.startswith(c)] for c in CATS}
def fboot(stat, fams, n=2000, seed=0):
    rng = np.random.default_rng(seed); idx = {f: np.where(fams == f)[0] for f in FAMS}; pt = stat(np.arange(len(fams))); bs = []
    for _ in range(n): bs.append(stat(np.concatenate([idx[f] for c in BYC for f in rng.choice(BYC[c], len(BYC[c]))])))
    return dict(pt=float(pt), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
R = {}
for m in MODELS:
    F = pd.DataFrame([json.loads(l) for l in open(f"{B}/decisions/{m}_forced.jsonl")]); F["sid"] = F.scenario_id.astype(int)
    k = lambda d: list(zip(d.sid, d.order_id))
    f0 = F[F.budget == 0].set_index(["sid", "order_id"]); f7 = F[F.budget == 768].set_index(["sid", "order_id"])
    A = pd.DataFrame([json.loads(l) for l in open(f"{B}/R1_readout_ablation/{m}.jsonl")]); A["sid"] = A.scenario_id.astype(int)
    A = A.set_index(["sid", "repeat_seed"]).reindex(f7.index)
    keys = list(f7.index); fams = np.array([S[i]["fam"] for i, _ in keys])
    q0 = f0.reindex(f7.index).test_utility_ratio.values; q7 = f7.test_utility_ratio.values
    qa = np.array([q_of(i, b) for (i, _), b in zip(keys, A.alt_bid.values)])
    qc = np.array([q_of(i, S[i]["calc"]) for i, _ in keys])
    gr = lambda qq: (lambda ids: (qq[ids].mean() - q0[ids].mean()) / (qc[ids].mean() - q0[ids].mean()))
    r = dict(reproduce=float((A.orig_label == A.saved_label).mean()), gap_open=fboot(gr(q7), fams), gap_closed=fboot(gr(qa), fams),
             gap_diff=fboot(lambda ids: gr(qa)(ids) - gr(q7)(ids), fams), changed=float((A.alt_bid.values != f7.chosen_bid.values).mean()),
             alt_suffix=A.alt_suffix.iloc[0])
    try:
        U = pd.DataFrame([json.loads(l) for l in open(f"{B}/decisions/{m}_unforced4096.jsonl")]); U["sid"] = U.scenario_id.astype(int)
        U = U.set_index(["sid", "order_id"]).reindex(f7.index)
        r["gap_unf4096"] = fboot(gr(U.test_utility_ratio.values), fams)
        r["unf4096_hit_cap"] = float(U.hit_cap.mean()); r["unf4096_tokens_median"] = float(U.reasoning_tokens.median()); r["unf4096_tokens_mean"] = float(U.reasoning_tokens.mean())
    except FileNotFoundError: pass
    R[m] = r
json.dump(R, open("/tmp/www2027/results/AN/an_overnight_summary.json", "w"), indent=1)
G = lambda d: f"{100*d['pt']:.0f} [{100*d['lo']:.0f},{100*d['hi']:.0f}]"
for m, r in R.items():
    print(f"{m:26s} repro {100*r['reproduce']:.1f} open {G(r['gap_open'])} closed {G(r['gap_closed'])} diff {G(r['gap_diff'])} changed {100*r['changed']:.0f}% suffix {r['alt_suffix']!r}" +
          (f" | 4096 {G(r['gap_unf4096'])} cap {100*r['unf4096_hit_cap']:.1f}% med {r['unf4096_tokens_median']:.0f}" if "gap_unf4096" in r else ""))
