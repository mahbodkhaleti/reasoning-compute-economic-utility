"""Is the lean toward the highest bid about the bid's size or its place in the list?
Option orders are randomised (three orders per scenario), so display position is independent of size.
Among wrong answers at 768 forced tokens (saved letters), a conditional logit over the non-calculator options with
features: is the highest bid, is shown first, is shown last. Odds ratios with a scenario-cluster bootstrap (500 draws).
Also the plain shares: highest bid chosen when it is shown first / middle / last. eBay and AuctionNet.
Writes results/position_size/position_vs_size.json."""
import sys, json, numpy as np, pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit
sys.path.insert(0, "/tmp/www2027/analysis/rev25"); sys.path.insert(0, "/tmp/www2027/analysis/auctionnet")
def long_table(rows):
    # rows: (model, sid, seed, chosen, calc, bids_sorted, order)
    L = []
    for g, (m, sid, seed, ch, calc, bids, order) in enumerate(rows):
        top = max(bids)
        for b in order:
            if np.isclose(b, calc): continue
            L.append(dict(g=g, sid=sid, model=m, y=int(np.isclose(b, ch)), high=int(np.isclose(b, top)),
                          first=int(np.isclose(b, order[0])), last=int(np.isclose(b, order[-1]))))
    return pd.DataFrame(L)
def fit(D):
    r = ConditionalLogit(D.y.values, D[["high", "first", "last"]].values.astype(float), groups=D.g.values).fit(disp=0)
    return np.exp(r.params)
def analyse(rows, label, seed=0):
    D = long_table(rows); pt = fit(D)
    sids = D.sid.unique(); rng = np.random.default_rng(seed); bs = []
    X = D[["high", "first", "last"]].values.astype(float); Y = D.y.values; G = D.g.values
    idx = {s: np.where(D.sid.values == s)[0] for s in sids}
    for _ in range(int(__import__("os").environ.get("NB", "300"))):
        pick = rng.choice(sids, len(sids)); ii = [idx[s] for s in pick]
        sel = np.concatenate(ii); gg = np.concatenate([G[a] * 10000 + k for k, a in enumerate(ii)])
        r = ConditionalLogit(Y[sel], X[sel], groups=gg).fit(disp=0); bs.append(np.exp(r.params))
    bs = np.array(bs)
    out = {n: dict(OR=float(pt[j]), ci=[float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]) for j, n in enumerate(["high", "first", "last"])}
    # plain shares
    sh = {}
    for where in ("first", "middle", "last"):
        sel = [(ch, top) for (_, _, _, ch, calc, bids, order) in rows for top in [max(bids)]
               if not np.isclose(top, calc) and (("first" == where and np.isclose(order[0], top)) or ("last" == where and np.isclose(order[-1], top)) or
                   ("middle" == where and not np.isclose(order[0], top) and not np.isclose(order[-1], top)))]
        sh[where] = dict(n=len(sel), share_highest=float(np.mean([np.isclose(c, t) for c, t in sel])) if sel else None)
    out["share_highest_by_position"] = sh; out["n_wrong"] = int(D.g.nunique())
    print(label, {k: (round(v["OR"], 2), [round(x, 2) for x in v["ci"]]) for k, v in out.items() if k in ("high", "first", "last")}, out["n_wrong"],
          {k: (v["n"], round(100 * v["share_highest"])) for k, v in sh.items()})
    return out
R = {}
# eBay
from r_common import tf, S
SUB = ["deepseek_r1_llama_8b", "qwen3_4b", "qwen3_8b", "qwen3_14b"]
z = tf[(tf.budget == 768)]
z = z[~np.isclose(z.chosen_bid, z.calc)]
mk = lambda zz: [(m, i, s, float(b), float(S[i]["calc"]), [float(x) for x in S[i]["bids"]], list(o)) for m, i, s, b, o in zip(zz.model, zz.scenario_id, zz.repeat_seed, zz.chosen_bid, zz.order_list)]
R["ebay_all"] = analyse(mk(z), "eBay all")
R["ebay_deepseek_qwen"] = analyse(mk(z[z.model.isin(SUB)]), "eBay DeepSeek+Qwen")
# AuctionNet
from an_common import build_S
B = "/tmp/www2027/results/AN/ov/results"; SA = build_S(B); rows = []; rows_sub = []
for m in ["deepseek_r1_llama_8b", "phi4_mini_reasoning", "qwen3_4b", "qwen3_8b", "qwen3_14b", "granite_3_2_8b_instruct"]:
    for l in open(f"{B}/auctionnet/decisions/{m}_forced.jsonl"):
        r = json.loads(l)
        if r["budget"] != 768: continue
        i = int(r["scenario_id"]); s = SA[i]
        if np.isclose(r["chosen_bid"], s["calc"]): continue
        row = (m, i, r["order_id"], float(r["chosen_bid"]), s["calc"], [float(x) for x in s["bids"]], [float(x) for x in r["option_order"]])
        rows.append(row); rows_sub += [row] if m in SUB else []
R["auctionnet_all"] = analyse(rows, "AuctionNet all", 1)
R["auctionnet_deepseek_qwen"] = analyse(rows_sub, "AuctionNet DeepSeek+Qwen", 1)
json.dump(R, open("/tmp/www2027/results/position_size/position_vs_size.json", "w"), indent=1)
