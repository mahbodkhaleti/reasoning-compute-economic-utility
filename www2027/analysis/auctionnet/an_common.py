"""AuctionNet scenario dict S with the same fields as the eBay S in common_rev.py (calc = argmax displayed expected profit)."""
import json, numpy as np, pandas as pd
def build_S(B):
    SC = pd.read_csv(f"{B}/auctionnet/scenarios.csv", dtype={"scenario_id": str}); S = {}
    for _, r in SC.iterrows():
        C = sorted(json.loads(r.candidates_json), key=lambda c: c["bid"]); v = float(r.value)
        bids = np.array([float(c["bid"]) for c in C]); dp = np.round(np.array([c["train_p"] for c in C]), 4)
        tu = np.array([c["test_u"] for c in C]); eu = dp * (v - bids); srt = np.sort(eu)[::-1]
        S[int(r.scenario_id)] = dict(product=r.category, v=v, K=len(C), bids=bids, dp=dp, tu=tu, eu=eu, fam=f"{r.category}|{v}",
                                     calc=float(bids[np.argmax(eu)]), ustar=float(tu.max()), gap12=float((srt[0] - srt[1]) / srt[0]), hp=float(bids[np.argmax(dp)]))
    return S
