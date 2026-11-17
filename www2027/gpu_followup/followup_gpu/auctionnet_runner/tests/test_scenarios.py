import os, sys, json, numpy as np, pandas as pd, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from auctionnet_runner import scenarios as SC, data as D

def fake_T(seed=1):
    r = np.random.default_rng(seed); parts = []
    for k, (prod, scale) in enumerate(zip(D.PRODUCTS, (80, 200, 450))):
        for split, n in (("train", 3000), ("test", 1000)):
            parts.append(pd.DataFrame({"product": prod, "split": split, "m_any_cpm": r.gamma(3.0, scale / 3.0, n), "m_top_cpm": r.gamma(3.0, scale / 2.0, n)}))
    return pd.concat(parts, ignore_index=True)

def test_build_mirrors_ebay_design():
    C = yaml.safe_load(open(os.path.join(ROOT, "auctionnet_runner", "config.yaml")))
    S, info = SC.build(fake_T(), C, "m_any_cpm", lambda *_: None)
    assert len(S) == 3 * 18 * 2 * 2, len(S)
    assert S.scenario_id.is_unique
    for _, s in S.iterrows():
        c = json.loads(s.candidates_json); b = [x["bid"] for x in c]
        assert len(b) == s.option_count and b == sorted(b) and max(b) < s.value
        assert all(0 <= x["train_p"] <= 1 for x in c)
    chk = SC.calculator_check(S); assert 0.5 < chk["q_calc"] <= 1.0 and chk["q_random"] < chk["q_calc"]

def test_empirical_probability_is_the_ebay_estimator():
    import prepare_scenarios as PS
    x = np.array([1.0, 2.0, 3.0, 4.0]); assert PS.empirical_probability(x, 2.0) == 0.5
