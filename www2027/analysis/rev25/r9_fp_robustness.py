"""Item 4 (round 2): robustness of the first-price equilibrium (FP eq.) scoring.
Variants: value distribution lognormal (paper) / gamma / Weibull; second-highest value = closing price, or closing
price minus eBay's bid increment; number of bidders n, n - 1, n + 1 (fit and rivals). Point estimates of gap recovered
at 768 tokens, seller revenue and bidder surplus change when answers move to the calculator (per 1,000 decisions),
and the overbid share of wrong answers."""
import sys; sys.path.insert(0, "/tmp/www2027/analysis/rev25")
from r_common import *
from scipy import stats, optimize
au = pd.read_csv("/tmp/www2027/analysis/auction_splits.csv")
au["product"] = au["item"].map(lambda it: next(p for p in PRODUCTS if p.split()[0].lower() in it.lower()))
ho = {p: au[(au["product"] == p) & (au.split != "train")] for p in PRODUCTS}
def increment(x):
    """eBay bid increments by current price (US schedule)."""
    for hi, inc in [(1, 0.05), (5, 0.25), (25, 0.5), (100, 1.0), (250, 2.5), (500, 5.0), (1000, 10.0), (2500, 25.0), (5000, 50.0)]:
        if x < hi: return inc
    return 100.0
DISTS = {
    "lognormal": (lambda th: stats.lognorm(s=np.exp(th[1]), scale=np.exp(th[0])), lambda c: [np.log(c).mean(), np.log(np.log(c).std())]),
    "gamma": (lambda th: stats.gamma(a=np.exp(th[0]), scale=np.exp(th[1])), lambda c: [np.log(c.mean() ** 2 / c.var()), np.log(c.var() / c.mean())]),
    "weibull": (lambda th: stats.weibull_min(c=np.exp(th[0]), scale=np.exp(th[1])), lambda c: [np.log(1.5), np.log(c.mean())]),
}
def fit(dist, c, n):
    mk, init = DISTS[dist]
    def nll(th):
        D = mk(th); G = np.clip(D.cdf(c), 1e-12, 1 - 1e-12); lg = D.logpdf(c)
        v = -np.sum(np.log(n * (n - 1)) + (n - 2) * np.log(G) + np.log(1 - G) + lg)
        return v if np.isfinite(v) else 1e18
    r = optimize.minimize(nll, init(c), method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
    return mk(r.x)
def build(dist, inc, dn):
    curves = {}
    for p, h in ho.items():
        c = h.final_price.values.astype(float); n = h.n_bidders.values.astype(float) + dn
        if inc: c = np.maximum(c - np.array([increment(x) for x in c]), 0.01)
        ok = n >= 2; D = fit(dist, c[ok], n[ok])
        lo, hi = D.ppf(1e-6), D.ppf(1 - 1e-9); xs = np.linspace(max(lo, 1e-6), hi, 8000); G = D.cdf(xs)
        ns = np.clip(h.n_bidders.values + dn, 1, None).astype(int); per = {}
        for nn in np.unique(ns):
            N = nn + 1; GN = G ** (N - 1)
            integ = np.concatenate([[0], np.cumsum(0.5 * (GN[1:] + GN[:-1]) * np.diff(xs))])
            beta = np.maximum.accumulate(xs - np.where(GN > 1e-300, integ / np.maximum(GN, 1e-300), 0))
            Hn = G ** nn; dH = np.diff(Hn); bm = 0.5 * (beta[1:] + beta[:-1])
            tail = np.concatenate([np.cumsum((bm * dH)[::-1])[::-1], [0]]); per[nn] = (beta, Hn, tail)
        curves[p] = (per, ns)
    return curves
def tables(curves):
    UF = {}
    for i, s in S.items():
        per, ns = curves[s["product"]]
        pf = np.array([np.mean([np.interp(b, per[n][0], per[n][1]) for n in ns]) for b in s["bids"]])
        rev = []
        for b in s["bids"]:
            r_ = []
            for n in ns:
                beta, Hn, tail = per[n]; k = int(min(max(np.floor(np.interp(b, beta, np.arange(len(beta)))), 0), len(beta) - 1))
                r_.append(np.interp(b, beta, Hn) * b + tail[k])
            rev.append(np.mean(r_))
        u = pf * (s["v"] - s["bids"])
        UF[i] = dict(u=dict(zip(s["bids"], u)), umax=u.max(), opt=s["bids"][np.argmax(u)], rev=dict(zip(s["bids"], rev)))
    return UF
VARS = {"lognormal": ("lognormal", False, 0), "gamma": ("gamma", False, 0), "weibull": ("weibull", False, 0),
        "increment": ("lognormal", True, 0), "n_minus_1": ("lognormal", False, -1), "n_plus_1": ("lognormal", False, 1)}
R = {}
for name, (dist, inc, dn) in VARS.items():
    UF = tables(build(dist, inc, dn)); qF = lambda i, b: UF[i]["u"][float(b)] / UF[i]["umax"] if UF[i]["umax"] > 0 else 1.0
    res = {}
    for m in MODELS:
        z7 = tf[(tf.model == m) & (tf.budget == 768)]; z0 = tf[(tf.model == m) & (tf.budget == 0)]
        q7 = np.mean([qF(i, b) for i, b in zip(z7.scenario_id, z7.chosen_bid)]); q0_ = np.mean([qF(i, b) for i, b in zip(z0.scenario_id, z0.chosen_bid)])
        qc = np.mean([qF(i, S[i]["calc"]) for i in z7.scenario_id])
        rm = np.mean([UF[i]["rev"][float(b)] for i, b in zip(z7.scenario_id, z7.chosen_bid)]); rc = np.mean([UF[i]["rev"][S[i]["calc"]] for i in z7.scenario_id])
        sm = np.mean([UF[i]["u"][float(b)] for i, b in zip(z7.scenario_id, z7.chosen_bid)]); scc = np.mean([UF[i]["u"][S[i]["calc"]] for i in z7.scenario_id])
        opt = np.array([UF[i]["opt"] for i in z7.scenario_id]); w = ~np.isclose(z7.chosen_bid.values, opt)
        res[m] = dict(gap768=float((q7 - q0_) / (qc - q0_)), qcalc=float(qc), rev_calc_minus_model_k=float((rc - rm)), sur_calc_minus_model_k=float((scc - sm)),
                      overbid_of_wrong=float(np.mean(z7.chosen_bid.values[w] > opt[w])))
    R[name] = res
    print(name, {LAB[m]: (round(100 * v["gap768"]), round(v["rev_calc_minus_model_k"], 1), round(v["sur_calc_minus_model_k"], 1), round(v["overbid_of_wrong"], 2)) for m, v in res.items()}, flush=True)
base = R["lognormal"]
R["max_abs_gap_change_points"] = float(max(abs(100 * (R[v][m]["gap768"] - base[m]["gap768"])) for v in VARS for m in MODELS))
R["ranges"] = {k: [float(min(R[v][m][k] for v in VARS for m in MODELS)), float(max(R[v][m][k] for v in VARS for m in MODELS))] for k in ("gap768", "rev_calc_minus_model_k", "sur_calc_minus_model_k", "overbid_of_wrong", "qcalc")}
dump25("r9_fp_robustness", R)
print("max change", round(R["max_abs_gap_change_points"], 1), "ranges", {k: [round(x, 3) for x in v] for k, v in R["ranges"].items()})
