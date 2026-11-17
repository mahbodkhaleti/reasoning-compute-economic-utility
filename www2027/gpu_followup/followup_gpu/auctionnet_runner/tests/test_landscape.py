import os, sys, tempfile, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from auctionnet_runner import data as D

def fake_csv(path, period=7, n_pv=300, n_adv=48, seed=0):
    r = np.random.default_rng(seed); rows = []
    for pv in range(n_pv):
        bids = r.gamma(2.0, 0.1, n_adv); s = -np.sort(-bids)
        for a in range(n_adv):
            rows.append(dict(deliveryPeriodIndex=period, advertiserNumber=a, advertiserCategoryIndex=a % 8, CPAConstraint=30.0,
                             timeStepIndex=pv // 50, pvIndex=pv, pValue=r.random() * 0.01, bid=bids[a], leastWinningCost=s[3]))
    pd.DataFrame(rows).to_csv(path, index=False)

def test_period_table():
    d = tempfile.mkdtemp(); p = os.path.join(d, "p7.csv"); fake_csv(p)
    tabs, chk = D.period_table(p, lambda *_: None); t = tabs[0]
    raw = pd.read_csv(p)
    third = raw.groupby("pvIndex").bid.apply(lambda x: np.sort(x.values)[::-1][2]).values
    top = raw.groupby("pvIndex").bid.max().values
    assert np.allclose(np.sort(t.m_any.values), np.sort(third), atol=1e-6)
    assert np.allclose(np.sort(t.m_top.values), np.sort(top), atol=1e-6)
    assert chk < 1e-5

def test_missing_column_fails():
    d = tempfile.mkdtemp(); p = os.path.join(d, "bad.csv"); pd.DataFrame({"x": [1]}).to_csv(p, index=False)
    try: D.period_table(p, lambda *_: None); assert False
    except ValueError: pass
