"""Batch 3: CPU checks requested in two external review notes.

All analyses reuse saved decisions; no new inference.
  A  routing with signals that do not solve the task; oracle; leave-one-product-out
  B  bootstrap that resamples held-out auctions within product (plus families)
     and a construction-sample scoring sensitivity for the calculator
  C  second-price (proxy-bid) rescoring with a closing-price bracket
  D  product effects: per product, equal weight, without Cartier, value-normalized
  E  near misses and confidence vs dollar regret
  F  calculator with noisy or coarsened probabilities
  G  Bonferroni-adjusted family intervals across six models
  H  majority vote at exactly matched token cost, with runtime
  I  trace coverage, arithmetic slips, position bias, native no-think baseline
"""
import json, itertools, os
import numpy as np, pandas as pd
from scipy import stats

OUT = "out"; os.makedirs(OUT, exist_ok=True)
SEED = 20261107
rng = np.random.default_rng(SEED)
MODELS = ["deepseek_r1_llama_8b", "phi4_mini_reasoning", "qwen3_4b", "qwen3_8b", "qwen3_14b", "granite_3_2_8b_instruct"]
LAB = {"deepseek_r1_llama_8b": "DeepSeek-8B", "phi4_mini_reasoning": "Phi-mini", "qwen3_4b": "Qwen3-4B",
       "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B", "granite_3_2_8b_instruct": "Granite-8B"}
R = {}

d = pd.read_csv("saved_records.csv.gz")
sc = pd.read_csv("scenarios.csv")
auc = pd.read_csv("auction_splits.csv")
PRODUCTS = sorted(sc.category.unique())
price = {(p, s): np.sort(auc[(auc.item == p) & (auc.split == s)].final_price.values) for p in PRODUCTS for s in ["train", "test"]}

S = {}
for r in sc.itertuples():
    c = json.loads(r.candidates_json)
    bids = np.array([x["bid"] for x in c], float)
    trp = np.array([x["train_p"] for x in c]); tep = np.array([x["test_p"] for x in c])
    S[r.scenario_id] = dict(product=r.category, v=float(r.value), K=r.option_count, bids=bids, train_p=trp, test_p=tep,
                            family=f"{r.category}|{r.value}")
# sanity: reproduce held-out probabilities from the auction file (share of prices <= bid)
for sid, s in list(S.items())[:30]:
    pr = price[(s["product"], "test")]
    assert np.allclose(np.searchsorted(pr, s["bids"], side="right") / len(pr), s["test_p"])
SIDS = sorted(S)
FAM = sorted({S[i]["family"] for i in SIDS}); PROD_OF = {f: f.split("|")[0] for f in FAM}
fam_ids = {f: [i for i in SIDS if S[i]["family"] == f] for f in FAM}

def ecdf(sorted_prices, b):
    return np.searchsorted(sorted_prices, b, side="right") / len(sorted_prices)

def calc_bid(s, p=None):
    p = s["train_p"] if p is None else p
    return s["bids"][np.argmax(p * (s["v"] - s["bids"]))]

g = d[d.decoding == "greedy"].copy()
g["fam"] = g.scenario_id.map(lambda i: S[i]["family"])
g["product"] = g.scenario_id.map(lambda i: S[i]["product"])

# ======================= A. routing signals
W = g[g.budget.isin([0, 768])].pivot_table(index=["model", "scenario_id", "repeat_seed"], columns="budget",
                                           values=["test_utility", "chosen_bid", "constrained_option_probability"]).reset_index()
W.columns = ["model", "scenario_id", "order", "b0", "b768", "p0", "p768", "te0", "te768"]
def tr_u(sid, b):
    s = S[sid]; k = np.where(np.isclose(s["bids"], b))[0][0]
    return s["train_p"][k] * (s["v"] - b)
W["tr0"] = [tr_u(i, b) for i, b in zip(W.scenario_id, W.b0)]
W["tr768"] = [tr_u(i, b) for i, b in zip(W.scenario_id, W.b768)]
W["family"] = W.scenario_id.map(lambda i: S[i]["family"]); W["product"] = W.scenario_id.map(lambda i: S[i]["product"])
W["value"] = W.scenario_id.map(lambda i: S[i]["v"])
W["max_surplus"] = W.scenario_id.map(lambda i: S[i]["v"] - S[i]["bids"].min())
W["stakes"] = W.scenario_id.map(lambda i: float(np.max(S[i]["train_p"] * (S[i]["v"] - S[i]["bids"]))))
prod_scale = {p: float(np.median(price[(p, "train")])) for p in PRODUCTS}   # construction-sample median price
W["product_scale"] = W["product"].map(prod_scale)
W["neg_conf"] = -W.p0
# folds: product-stratified 6-fold over value families (same scheme as batch 2)
fold_of = {}
for p in PRODUCTS:
    fl = [f for f in FAM if PROD_OF[f] == p]; perm = rng.permutation(len(fl))
    for i, k in enumerate(perm): fold_of[fl[k]] = i % 6
W["fold"] = W.family.map(fold_of)

def pick_threshold(score, tr0, tr768, target=0.90):
    gain = (tr768 - tr0).sum()
    order = np.argsort(-score, kind="stable"); cum = np.cumsum((tr768 - tr0)[order])
    ok = np.where(cum >= target * gain)[0]; k = ok[0] if len(ok) else len(score) - 1
    return score[order][k]

def area_curve(score, te0, te768):
    order = np.argsort(-score, kind="stable"); cum = np.r_[0, np.cumsum((te768 - te0)[order])]
    n = len(score); grid = np.linspace(0, 1, 101)
    kept = np.array([cum[int(round(q * n))] / cum[-1] for q in grid])
    return float(np.trapezoid(kept, grid)), float(kept[50]), kept

SIGNALS = {"value": "value", "product": "product_scale", "max_surplus": "max_surplus",
           "confidence": "neg_conf", "stakes": "stakes"}
routeA = {}
curvesA = []
for m in MODELS:
    x = W[W.model == m].reset_index(drop=True)
    jit = rng.uniform(0, 1e-7, len(x))     # deterministic tie break
    te0, te768 = x.te0.values, x.te768.values
    full = te768.mean() - te0.mean()
    out = {}
    for name, col in SIGNALS.items():
        sc_ = x[col].values.astype(float) + jit
        esc = np.zeros(len(x), bool)
        for k in range(6):
            trm = (x.fold != k).values
            t = pick_threshold(sc_[trm], x.tr0.values[trm], x.tr768.values[trm])
            esc[~trm] = sc_[~trm] >= t
        # leave-one-product-out: threshold from two products, applied to the third
        esc_lopo = np.zeros(len(x), bool)
        for p in PRODUCTS:
            trm = (x["product"] != p).values
            t = pick_threshold(sc_[trm], x.tr0.values[trm], x.tr768.values[trm])
            esc_lopo[~trm] = sc_[~trm] >= t
        kept = (np.where(esc, te768, te0).mean() - te0.mean()) / full
        kept_lopo = (np.where(esc_lopo, te768, te0).mean() - te0.mean()) / full
        a, k50, kc = area_curve(sc_, te0, te768)
        # family bootstrap of the cross-fitted rule
        bs = []
        for _ in range(2000):
            ids = np.concatenate([np.where(x.family.values == f)[0] for p in PRODUCTS
                                  for f in rng.choice([f for f in FAM if PROD_OF[f] == p], 18, replace=True)])
            g0, g7 = te0[ids].mean(), te768[ids].mean()
            bs.append((np.where(esc[ids], te768[ids], te0[ids]).mean() - g0) / (g7 - g0))
        out[name] = dict(saved=float(1 - esc.mean()), kept=float(kept), kept_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                         saved_lopo=float(1 - esc_lopo.mean()), kept_lopo=float(kept_lopo), area=a, kept50=k50)
        for q, v in zip(np.linspace(0, 1, 101), kc): curvesA.append(dict(model=m, rule=name, token_share=q, gain_kept=v))
    # oracle: escalate exactly the decisions whose held-out gain is positive
    pos = (te768 - te0) > 0
    out["oracle"] = dict(saved=float(1 - pos.mean()), kept=float(((np.where(pos, te768, te0)).mean() - te0.mean()) / full))
    a, k50, kc = area_curve(te768 - te0 + jit, te0, te768)
    out["oracle"].update(area=a, kept50=k50)
    for q, v in zip(np.linspace(0, 1, 101), kc): curvesA.append(dict(model=m, rule="oracle", token_share=q, gain_kept=v))
    routeA[m] = out
pd.DataFrame(curvesA).to_csv(f"{OUT}/routing_curves_v3.csv", index=False)
R["routing"] = routeA

# ======================= B. auction-resampling bootstrap
cell = (g[g.budget.isin([0, 768])].groupby(["model", "scenario_id", "budget"]).chosen_bid
        .apply(lambda b: np.array(b.values, float)).unstack())
calc_b = {i: calc_bid(S[i]) for i in SIDS}
def utilities(test_prices):
    """Return dict sid -> (bids, U per bid, U*)"""
    out = {}
    for i in SIDS:
        s = S[i]; p = ecdf(test_prices[s["product"]], s["bids"]); u = p * (s["v"] - s["bids"])
        out[i] = (s["bids"], u, u.max())
    return out
def score(ut, m):
    rows = []
    for i in SIDS:
        bids, u, ustar = ut[i]
        U0 = np.mean([u[np.isclose(bids, b)][0] for b in cell.loc[(m, i), 0]])
        U7 = np.mean([u[np.isclose(bids, b)][0] for b in cell.loc[(m, i), 768]])
        Uc = u[np.isclose(bids, calc_b[i])][0]
        rows.append((U0, U7, Uc, ustar))
    return np.array(rows)
base_test = {p: price[(p, "test")] for p in PRODUCTS}
base_ut = utilities(base_test)
base = {m: score(base_ut, m) for m in MODELS}
famidx = {f: np.array([SIDS.index(i) for i in fam_ids[f]]) for f in FAM}
def resample_fams():
    return np.concatenate([famidx[f] for p in PRODUCTS for f in rng.choice([f for f in FAM if PROD_OF[f] == p], 18, replace=True)])
NB = 1000
drawsB = {m: {"fam": [], "comb": []} for m in MODELS}
for bdraw in range(NB):
    tp = {p: np.sort(rng.choice(base_test[p], len(base_test[p]), replace=True)) for p in PRODUCTS}
    ut = utilities(tp)
    ids = resample_fams()
    for m in MODELS:
        A_ = score(ut, m)[ids]; B_ = base[m][ids]
        for key, arr in (("comb", A_), ("fam", B_)):
            gain = arr[:, 1].mean() - arr[:, 0].mean()
            dq = (arr[:, 1] / arr[:, 3]).mean() - (arr[:, 0] / arr[:, 3]).mean()
            gap = gain / (arr[:, 2].mean() - arr[:, 0].mean())
            calc_lead = arr[:, 2].mean() - arr[:, 1].mean()
            drawsB[m][key].append((gain, dq, gap, calc_lead))
B = {}
for m in MODELS:
    arr0 = base[m]
    point = dict(gain=arr0[:, 1].mean() - arr0[:, 0].mean(),
                 dq=(arr0[:, 1] / arr0[:, 3]).mean() - (arr0[:, 0] / arr0[:, 3]).mean(),
                 gap=(arr0[:, 1].mean() - arr0[:, 0].mean()) / (arr0[:, 2].mean() - arr0[:, 0].mean()),
                 calc_lead=arr0[:, 2].mean() - arr0[:, 1].mean())
    res = {"point": point}
    for key in ["fam", "comb"]:
        a = np.array(drawsB[m][key])
        res[key] = {n: [float(np.percentile(a[:, j], 2.5)), float(np.percentile(a[:, j], 97.5))]
                    for j, n in enumerate(["gain", "dq", "gap", "calc_lead"])}
        res[key]["share_gain_pos"] = float((a[:, 0] > 0).mean())
        res[key]["share_calc_ahead"] = float((a[:, 3] > 0).mean())
    B[m] = res
# construction-sample sensitivity: resample construction auctions, recompute the calculator's choice
calc_q = []
for _ in range(1000):
    trp = {p: np.sort(rng.choice(price[(p, "train")], len(price[(p, "train")]), replace=True)) for p in PRODUCTS}
    qs = []
    for i in SIDS:
        s = S[i]; p = ecdf(trp[s["product"]], s["bids"]); b = s["bids"][np.argmax(p * (s["v"] - s["bids"]))]
        bids, u, ustar = base_ut[i]; qs.append(u[np.isclose(bids, b)][0] / ustar)
    calc_q.append(np.mean(qs))
B["calculator_construction_resample_q"] = [float(np.percentile(calc_q, 2.5)), float(np.mean(calc_q)), float(np.percentile(calc_q, 97.5))]
R["auction_bootstrap"] = B

# ======================= C. second-price (proxy) rescoring
def inc(pr):
    edges = [(1, .05), (5, .25), (25, .5), (100, 1), (250, 2.5), (500, 5), (1000, 10), (2500, 25), (5000, 50), (np.inf, 100)]
    return np.array([next(i_ for e, i_ in edges if x < e) for x in pr])
def u2(s, b, mode):
    pr = price[(s["product"], "test")]
    beat = pr if mode == "low" else pr + inc(pr)
    win = beat[None, :] <= np.atleast_1d(b)[:, None]
    return (win * (s["v"] - beat[None, :])).mean(axis=1)
C = {}
for mode in ["low", "high"]:
    U2 = {i: u2(S[i], S[i]["bids"], mode) for i in SIDS}
    U2star = {i: U2[i].max() for i in SIDS}
    bidv = {i: float(u2(S[i], np.array([S[i]["v"]]), mode)[0]) for i in SIDS}
    top_bid_q = np.mean([U2[i][np.argmax(S[i]["bids"])] / U2star[i] for i in SIDS])
    calc_q2 = np.mean([U2[i][np.isclose(S[i]["bids"], calc_b[i])][0] / U2star[i] for i in SIDS])
    res = dict(highest_bid_is_optimal_share=float(np.mean([np.argmax(U2[i]) == np.argmax(S[i]["bids"]) for i in SIDS])),
               bid_value_over_menu_opt=float(np.mean([bidv[i] for i in SIDS]) / np.mean([U2star[i] for i in SIDS])),
               highest_bid_q=float(top_bid_q), calculator_q=float(calc_q2), models={})
    for m in MODELS:
        qq = {}
        for bud in [0, 768]:
            vals = []
            for i in SIDS:
                for b in cell.loc[(m, i), bud]:
                    vals.append(U2[i][np.isclose(S[i]["bids"], b)][0] / U2star[i])
            qq[bud] = float(np.mean(vals))
        res["models"][m] = qq
    C[mode] = res
R["second_price"] = C

# ======================= D. product effects
D = {}
cellU = g[g.budget.isin([0, 768])].groupby(["model", "scenario_id", "budget"]).agg(U=("test_utility", "mean"), Q=("test_utility_ratio", "mean")).reset_index()
for m in MODELS:
    c = cellU[cellU.model == m]
    U = c.pivot(index="scenario_id", columns="budget", values="U"); Q = c.pivot(index="scenario_id", columns="budget", values="Q")
    prod = U.index.map(lambda i: S[i]["product"]); v = U.index.map(lambda i: S[i]["v"]).values
    dU = (U[768] - U[0]); dQ = (Q[768] - Q[0])
    per = {p: dict(gain=float(dU[prod == p].mean()), dq=float(dQ[prod == p].mean())) for p in PRODUCTS}
    D[m] = dict(per_product=per, equal_weight_gain=float(np.mean([per[p]["gain"] for p in PRODUCTS])),
                gain_without_cartier=float(dU[prod != "Cartier wristwatch"].mean()),
                dq_without_cartier=float(dQ[prod != "Cartier wristwatch"].mean()),
                gain_per_value_pct=float(100 * (dU.values / v).mean()),
                dq_min_product=float(min(per[p]["dq"] for p in PRODUCTS)))
R["product_effects"] = D

# ======================= E. near misses and confidence vs dollar regret
E = {}
for m in MODELS:
    x = g[(g.model == m) & (g.budget.isin([0, 768]))]
    res = {}
    for bud in [0, 768]:
        y = x[x.budget == bud]
        wrong = ~np.isclose(y.test_utility, y.test_optimal_utility)
        q = y.test_utility_ratio
        res[bud] = dict(wrong_share=float(wrong.mean()), wrong_within_5pct=float((q[wrong] >= 0.95).mean()),
                        wrong_within_1pct=float((q[wrong] >= 0.99).mean()),
                        mean_regret_usd_wrong=float((y.test_optimal_utility - y.test_utility)[wrong].mean()))
    y0 = x[x.budget == 0].set_index(["scenario_id", "repeat_seed"]); y7 = x[x.budget == 768].set_index(["scenario_id", "repeat_seed"]).reindex(y0.index)
    res["spearman_conf_regret"] = float(stats.spearmanr(y0.constrained_option_probability, y0.test_optimal_utility - y0.test_utility).correlation)
    res["spearman_conf_gain"] = float(stats.spearmanr(y0.constrained_option_probability, y7.test_utility.values - y0.test_utility.values).correlation)
    E[m] = res
R["near_miss"] = E

# ======================= F. calculator with noisy or coarsened probabilities
F = {}
base_q = lambda bidf: float(np.mean([base_ut[i][1][np.isclose(S[i]["bids"], bidf(i))][0] / base_ut[i][2] for i in SIDS]))
F["exact"] = base_q(lambda i: calc_b[i])
F["round_2dp"] = base_q(lambda i: calc_bid(S[i], np.round(S[i]["train_p"], 2)))
F["round_1dp"] = base_q(lambda i: calc_bid(S[i], np.round(S[i]["train_p"], 1)))
for s_ in [0.05, 0.10, 0.20, 0.40]:
    qs = []
    for _ in range(300):
        qs.append(base_q(lambda i: calc_bid(S[i], S[i]["train_p"] * (1 + rng.uniform(-s_, s_, S[i]["K"])))))
    F[f"mult_noise_{int(s_*100)}pct"] = float(np.mean(qs))
F["highest_probability_bid"] = base_q(lambda i: S[i]["bids"][np.argmax(S[i]["train_p"])])
R["calculator_noise"] = F

# ======================= G. Bonferroni-adjusted family intervals
G = {}
for m in MODELS:
    c = cellU[cellU.model == m]
    U = c.pivot(index="scenario_id", columns="budget", values="U"); Q = c.pivot(index="scenario_id", columns="budget", values="Q")
    dU = (U[768] - U[0]).reindex(SIDS).values; dQ = (Q[768] - Q[0]).reindex(SIDS).values
    a = []
    for _ in range(10000):
        ids = resample_fams(); a.append((dU[ids].mean(), dQ[ids].mean()))
    a = np.array(a); lo, hi = 100 * 0.05 / 12, 100 - 100 * 0.05 / 12
    G[m] = dict(gain=[float(np.percentile(a[:, 0], lo)), float(np.percentile(a[:, 0], hi))],
                dq=[float(np.percentile(a[:, 1], lo)), float(np.percentile(a[:, 1], hi))])
R["bonferroni"] = G

# ======================= H. majority vote at matched token cost
st = d[d.decoding == "stochastic"].copy()
H = {}
def vote(bids, probs):
    cnt = {}
    for b_, p_ in zip(bids, probs):
        c0, s0 = cnt.get(b_, (0, 0.0)); cnt[b_] = (c0 + 1, s0 + p_)
    return max(cnt.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
for m in MODELS:
    x = st[st.model == m]
    P = {bud: x[x.budget == bud].pivot(index="scenario_id", columns="repeat_seed", values="chosen_bid") for bud in [128, 384, 768]}
    PR = {bud: x[x.budget == bud].pivot(index="scenario_id", columns="repeat_seed", values="constrained_option_probability") for bud in [128, 384, 768]}
    T = {bud: x[x.budget == bud].pivot(index="scenario_id", columns="repeat_seed", values="elapsed_seconds") for bud in [128, 384, 768]}
    seeds = list(P[128].columns)
    def qv(bud, k):
        """scenario-level mean held-out q over all k-subsets of seeds; also mean seconds"""
        qs = np.zeros(len(SIDS)); secs = np.zeros(len(SIDS)); n = 0
        for comb in itertools.combinations(seeds, k):
            n += 1
            for j, i in enumerate(SIDS):
                b = vote(P[bud].loc[i, list(comb)].values, PR[bud].loc[i, list(comb)].values)
                bids, u, ustar = base_ut[i]; qs[j] += u[np.isclose(bids, b)][0] / ustar
                secs[j] += T[bud].loc[i, list(comb)].sum()
        return qs / n, secs / n
    q3_128, s3_128 = qv(128, 3); q1_384, s1_384 = qv(384, 1)
    q2_384, s2_384 = qv(384, 2); q1_768, s1_768 = qv(768, 1)
    res = {}
    for name, (qa, sa, qb, sb) in {"3x128_vs_1x384": (q3_128, s3_128, q1_384, s1_384),
                                   "2x384_vs_1x768": (q2_384, s2_384, q1_768, s1_768)}.items():
        diff = qa - qb
        bs = [diff[resample_fams()].mean() for _ in range(4000)]
        res[name] = dict(q_vote=float(qa.mean()), q_single=float(qb.mean()), diff=float(diff.mean()),
                         ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                         sec_vote=float(sa.mean()), sec_single=float(sb.mean()))
    H[m] = res
R["voting_matched"] = H

# ======================= I. traces, position bias, no-think
tf = pd.read_csv("trace_features_v2.csv", dtype={"scenario_id": str})
tf["calc_bid"] = tf.scenario_id.map(lambda i: calc_b.get(int(i), np.nan) if int(i) in calc_b else np.nan)
I = {}
for m in MODELS:
    y = tf[(tf.model == m) & (tf.chat_mode == "reasoning") & (tf.budget == 768)]
    full = y.n_options_evaluated == y.K
    agree = np.isclose(y.chosen_bid, y.calc_bid)          # matches the exact calculator on displayed numbers
    y0 = tf[(tf.model == m) & (tf.chat_mode == "reasoning") & (tf.budget == 0)]
    pos_share = y0.chosen_pos.value_counts(normalize=True).sort_index()
    # position bias: chi-square of chosen position vs. uniform, separately by K
    chi = {}
    for K in [5, 7]:
        z = y0[y0.K == K].chosen_pos.value_counts().reindex(range(K), fill_value=0).values
        chi[K] = dict(p=float(stats.chisquare(z).pvalue), max_share=float(z.max() / z.sum()), argmax=int(z.argmax()))
    nt = tf[(tf.model == m) & (tf.chat_mode == "no_think")]
    I[m] = dict(full_coverage=float(full.mean()),
                agree_if_full=float(agree[full].mean()) if full.any() else None,
                agree_if_partial=float(agree[~full].mean()) if (~full).any() else None,
                wrong_share_partial=float((~full[~agree]).mean()),
                chosen_not_evaluated_among_wrong=float((y.chosen_evaluated[~agree] == 0).mean()),
                arith_error_rate=float(y.n_products_wrong.sum() / max(y.n_products_checked.sum(), 1)),
                pos_bias=chi,
                no_think_q=float((nt.test_utility / nt.test_optimal_utility).mean()),
                forced0_q=float((y0.test_utility / y0.test_optimal_utility).mean()))
R["traces"] = I

json.dump(R, open(f"{OUT}/batch3_results.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=lambda o: round(float(o), 4))[:200])
