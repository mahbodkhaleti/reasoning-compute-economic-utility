"""Downstream analyses with the closed-block letters (R1) in place of the saved letters, 768-token eBay traces.
Usage: python r1_closed_downstream.py open|closed. 'open' reruns the same code on the saved letters so every closed
value has an open-block value computed by the identical path. Reuses r5 (regression) and r4 (audit) code unchanged,
with their dump/print tails cut off so no existing results file is written. Output: results/R1/downstream_<mode>.json."""
import sys, json, glob
sys.path.insert(0, "/tmp/www2027/analysis/rev25")
from r_common import *
MODE = sys.argv[1]
if MODE == "closed":
    for f in sorted(glob.glob("/tmp/www2027/results/R1/results/extra/R1_readout_ablation/*.jsonl")):
        m = f.split("/")[-1][:-6]
        A = {(int(r["scenario_id"]), int(r["repeat_seed"])): float(r["alt_bid"]) for r in map(json.loads, open(f))}
        idx = tf.index[(tf.model == m) & (tf.budget == 768)]
        tf.loc[idx, "chosen_bid"] = [A[(i, s)] for i, s in zip(tf.loc[idx, "scenario_id"], tf.loc[idx, "repeat_seed"])]
    z = tf.budget == 768
    tf.loc[z, "q_read"] = [q_of(i, b) for i, b in zip(tf.loc[z, "scenario_id"], tf.loc[z, "chosen_bid"])]
    tf.loc[z, "cls"] = [classify(hv, i, K, b) for hv, i, K, b in zip(tf.loc[z, "hv"], tf.loc[z, "scenario_id"], tf.loc[z, "K"], tf.loc[z, "chosen_bid"])]
    tf["gap"] = tf.qcalc - tf.q_read
OUT = {"mode": MODE, "models": {}}
z7 = tf[tf.budget == 768]
for m in MODELS:
    z = z7[z7.model == m].reset_index(drop=True); fams = z.fam.values
    w = z[~np.isclose(z.chosen_bid, z.calc)]
    top = np.array([float(max(S[i]["bids"])) for i in w.scenario_id])
    hi = float(np.mean(np.isclose(w.chosen_bid.values, top)))
    hi_chance = float(np.mean([np.mean([np.isclose(b, max(S[i]["bids"])) for b in S[i]["bids"] if not np.isclose(b, S[i]["calc"])]) for i in w.scenario_id]))
    # pay-your-bid, empirical held-out win curve: bidder surplus of model answers minus calculator's, per 1,000 decisions
    sm = np.array([HELD[i][float(b)]["test_p"] * (S[i]["v"] - float(b)) for i, b in zip(z.scenario_id, z.chosen_bid)])
    sc = np.array([HELD[i][float(S[i]["calc"])]["test_p"] * (S[i]["v"] - float(S[i]["calc"])) for i in z.scenario_id])
    gap = z.gap.values; cl = z.cls.values
    OUT["models"][m] = dict(n=int(len(z)), n_wrong=int(len(w)), highest_share_of_wrong=hi, highest_chance=hi_chance,
        surplus_loss_per_1000=float(1000 * (sc - sm).mean()),
        split=dict(zip(CATS, fboot(lambda ids: np.array([gap[ids][cl[ids] == k].sum() / gap[ids].sum() for k in CATS]), fams))))
# regression (r5 code, unchanged)
_s = open("/tmp/www2027/analysis/rev25/r5_passover_regression.py").read()
exec(_s[_s.index("z = tf[tf.budget == 768].copy()"):_s.index("# per-model slopes")])
OUT["regression"] = dict(n=R["n"], events=R["events"], best_is_high=R["pooled"]["best_is_high"], finished=R["pooled"]["finished"], K7=R["pooled"]["K7"])
# audit (r4 code, unchanged)
_s = open("/tmp/www2027/analysis/rev25/r4_audit_oos.py").read()
exec(_s[_s.index("_t10 = open"):_s.index("dump25(")])
OUT["audit"] = dict(nested_lpo={m: R["nested_leave_product_out"][m]["gap_audit"] for m in MODELS},
                    nested_fam={m: R["nested_family_6fold"][m]["gap_audit"] for m in MODELS},
                    syn_to_ebay={m: R["syn_to_ebay"][m]["gap_audit"] for m in MODELS},
                    answer={m: R["syn_to_ebay"][m]["gap_answer"] for m in MODELS},
                    calls_lpo={m: R["nested_leave_product_out"][m]["calls"] for m in MODELS},
                    at20=R["curve_at"]["audit"][0.2], random_at20=R["curve_at"]["random"][0.2])
json.dump(OUT, open(f"/tmp/www2027/results/R1/downstream_{MODE}.json", "w"), indent=1, default=float)
print(json.dumps(OUT, indent=1, default=float)[:200])
