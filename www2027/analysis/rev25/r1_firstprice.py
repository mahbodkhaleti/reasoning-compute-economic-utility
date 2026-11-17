"""Item 1: re-score every saved decision and the calculator under a first-price equilibrium win curve.
Value distribution: eBay proxy auctions close at about the second-highest bidder's value, so per product we fit a
lognormal G by maximum likelihood to held-out (closing price, number of bidders n) pairs with n >= 2 (the paper's
t13 E1 fit). Rivals bid the symmetric first-price equilibrium beta(x) = x - int_0^x G^{N-1} / G(x)^{N-1}, N = n + 1.
A bid b wins with p_FP(b) = mean over held-out auctions of G(beta^{-1}(b))^n.
Outputs: q, gap recovered (family bootstrap), Table 2 split weighted by the FP gap, overbid and highest-bid shares
of wrong answers, and per-1,000-decision seller revenue / bidder surplus / total under first-price pricing
(model answer, audited answer and calculator), with intervals."""
import sys; sys.path.insert(0, "/tmp/www2027/analysis/rev25")
from r_common import *
from scipy import stats, optimize
au = pd.read_csv("/tmp/www2027/analysis/auction_splits.csv")
au["product"] = au["item"].map(lambda it: next(p for p in PRODUCTS if p.split()[0].lower() in it.lower()))
ho = {p: au[(au["product"] == p) & (au.split != "train")] for p in PRODUCTS}
def fit_G(c, n):
    lc = np.log(c)
    def nll(th):
        mu, ls = th; sd = np.exp(ls); z = (lc - mu) / sd
        G = np.clip(stats.norm.cdf(z), 1e-12, 1 - 1e-12); lg = stats.norm.logpdf(z) - np.log(sd) - lc
        return -np.sum(np.log(n * (n - 1)) + (n - 2) * np.log(G) + np.log(1 - G) + lg)
    r = optimize.minimize(nll, [lc.mean(), np.log(lc.std())], method="Nelder-Mead")
    return r.x[0], np.exp(r.x[1])
FIT, CURVE = {}, {}
for p, h in ho.items():
    h2 = h[h.n_bidders >= 2]; mu, sd = fit_G(h2.final_price.values, h2.n_bidders.values.astype(float))
    xs = np.exp(np.linspace(mu - 6 * sd, mu + 6 * sd, 6000)); G = stats.norm.cdf((np.log(xs) - mu) / sd)
    ns = h.n_bidders.clip(lower=1).values
    per_n = {}
    for n in np.unique(ns):
        N = n + 1; GN = G ** (N - 1)
        integ = np.concatenate([[0], np.cumsum(0.5 * (GN[1:] + GN[:-1]) * np.diff(xs))])
        beta = np.maximum.accumulate(xs - np.where(GN > 1e-300, integ / np.maximum(GN, 1e-300), 0))
        Hn = G ** n                                  # cdf of the highest rival value
        dH = np.diff(Hn); bmid = 0.5 * (beta[1:] + beta[:-1])
        tail = np.concatenate([np.cumsum((bmid * dH)[::-1])[::-1], [0]])   # int_{x_k}^inf beta dH
        per_n[n] = (beta, Hn, tail)
    CURVE[p] = (per_n, ns); FIT[p] = dict(mu=float(mu), sd=float(sd), n_fit=int(len(h2)), median_bidders=float(np.median(ns)))
def pFP(prod, b):
    per_n, ns = CURVE[prod]; return float(np.mean([np.interp(b, per_n[n][0], per_n[n][1]) for n in ns]))
def revFP(prod, b):
    """Expected payment to the seller: our bid if we win, else the highest rival's equilibrium bid."""
    per_n, ns = CURVE[prod]; out = []
    for n in ns:
        beta, Hn, tail = per_n[n]; xb = np.interp(b, beta, np.arange(len(beta)))
        k = int(min(max(np.floor(xb), 0), len(beta) - 1)); out.append(np.interp(b, beta, Hn) * b + tail[k])
    return float(np.mean(out))
# scenario-level tables
UF = {}
for i, s in S.items():
    pf = np.array([pFP(s["product"], b) for b in s["bids"]]); u = pf * (s["v"] - s["bids"])
    UF[i] = dict(p=dict(zip(s["bids"], pf)), u=dict(zip(s["bids"], u)), umax=u.max(), opt=s["bids"][np.argmax(u)],
                 rev=dict(zip(s["bids"], [revFP(s["product"], b) for b in s["bids"]])), hi=s["bids"].max())
qF = lambda i, b: UF[i]["u"][float(b)] / UF[i]["umax"] if UF[i]["umax"] > 0 else 1.0
R = {"fit": FIT, "calc_matches_FP_opt": float(np.mean([np.isclose(S[i]["calc"], UF[i]["opt"]) for i in S])), "models": {}}
for m in MODELS:
    res = {}
    zz = {b: tf[(tf.model == m) & (tf.budget == b)].sort_values(["scenario_id", "repeat_seed"]).reset_index(drop=True) for b in (0, 768, -1)}
    z0, z7, zn = zz[0], zz[768], zz[-1]
    assert (z0.scenario_id.values == z7.scenario_id.values).all()
    fams = z7.fam.values
    q0F = np.array([qF(i, b) for i, b in zip(z0.scenario_id, z0.chosen_bid)])
    q7F = np.array([qF(i, b) for i, b in zip(z7.scenario_id, z7.chosen_bid)])
    qnF = np.array([qF(i, b) for i, b in zip(zn.scenario_id, zn.chosen_bid)])
    qcF = np.array([qF(i, S[i]["calc"]) for i in z7.scenario_id])
    aud = np.array([S[i]["calc"] if not np.isclose(pick(vv, 2, b), b) else b for vv, b, i in zip(z7.vals, z7.chosen_bid, z7.scenario_id)])
    qaF = np.array([qF(i, b) for i, b in zip(z7.scenario_id, aud)])
    gr = lambda q: (lambda ids: (q[ids].mean() - q0F[ids].mean()) / (qcF[ids].mean() - q0F[ids].mean()))
    res["q0"], res["q768"], res["q1024run"], res["qcalc"] = map(float, (q0F.mean(), q7F.mean(), qnF.mean(), qcF.mean()))
    res["gap768"] = fboot(gr(q7F), fams)[0]; res["gap1024run"] = fboot(gr(qnF), fams)[0]; res["gap_audit"] = fboot(gr(qaF), fams)[0]
    res["gap768_empirical_for_reference"] = float((z7.q_read.mean() - z0.q_read.mean()) / (z7.qcalc.mean() - z0.q_read.mean()))
    gapF = qcF - q7F; cl = z7.cls.values
    res["split768"] = dict(zip(CATS, fboot(lambda ids: np.array([gapF[ids][cl[ids] == k].sum() / gapF[ids].sum() for k in CATS]), fams)))
    wrong = ~np.isclose(z7.chosen_bid.values, z7.calc.values)
    optF = np.array([UF[i]["opt"] for i in z7.scenario_id]); hi = np.array([UF[i]["hi"] for i in z7.scenario_id])
    wF = ~np.isclose(z7.chosen_bid.values, optF)
    res["overbid_share_of_wrong_FP"] = float(np.mean(z7.chosen_bid.values[wF] > optF[wF]))
    res["highest_bid_share_of_wrong"] = float(np.mean(np.isclose(z7.chosen_bid.values[wrong], hi[wrong])))
    res["loss_share_from_highest_bid_FP"] = float(gapF[wrong & np.isclose(z7.chosen_bid.values, hi)].sum() / gapF.sum())
    # welfare under first-price pricing, per 1,000 decisions
    def econ(bids):
        rev = np.array([UF[i]["rev"][float(b)] for i, b in zip(z7.scenario_id, bids)])
        sur = np.array([UF[i]["u"][float(b)] for i, b in zip(z7.scenario_id, bids)])
        return rev, sur
    (rm, sm), (ra, sa), (rc, sc_) = econ(z7.chosen_bid.values), econ(aud), econ(z7.calc.values)
    W = {}
    for nm, a, b in [("calc_minus_model_rev", rc, rm), ("calc_minus_model_sur", sc_, sm), ("audit_minus_model_rev", ra, rm),
                     ("audit_minus_model_sur", sa, sm), ("calc_minus_model_tot", rc + sc_, rm + sm), ("audit_minus_model_tot", ra + sa, rm + sm)]:
        W[nm] = fboot(lambda ids, a=a, b=b: 1000 * (a[ids] - b[ids]).mean(), fams)[0]
    res["welfare_per_1000"] = W
    res["audit_call_rate"] = float(np.mean(aud != z7.chosen_bid.values))
    R["models"][m] = res
dump25("r1_firstprice", R)
f = lambda x: f"{100*x['pt']:.0f} [{100*x['ci'][0]:.0f}, {100*x['ci'][1]:.0f}]"
k = lambda x: f"{x['pt']/1000:+.1f}k [{x['ci'][0]/1000:.1f}, {x['ci'][1]/1000:.1f}]"
print("fit", FIT, "calc==FP opt", round(R["calc_matches_FP_opt"], 3))
for m, r in R["models"].items():
    print(LAB[m], "qcalc %.3f q0 %.3f q768 %.3f | gap768 %s (emp %.0f) | 1024run %s | audit %s" % (r["qcalc"], r["q0"], r["q768"], f(r["gap768"]), 100 * r["gap768_empirical_for_reference"], f(r["gap1024run"]), f(r["gap_audit"])))
    print("   split", {c: f(v) for c, v in r["split768"].items()}, "| overbid of wrong %.2f highest-bid share %.2f loss from highest %.2f" % (r["overbid_share_of_wrong_FP"], r["highest_bid_share_of_wrong"], r["loss_share_from_highest_bid_FP"]))
    print("   welfare calc-model rev", k(r["welfare_per_1000"]["calc_minus_model_rev"]), "sur", k(r["welfare_per_1000"]["calc_minus_model_sur"]), "tot", k(r["welfare_per_1000"]["calc_minus_model_tot"]),
          "| audit-model rev", k(r["welfare_per_1000"]["audit_minus_model_rev"]), "sur", k(r["welfare_per_1000"]["audit_minus_model_sur"]))
