"""Item 2b: Table 2 split with written values taken from the code reader alone (every value the reader reads that
differs from the truth counts as an arithmetic error: an upper bound on Arith.), vs the paper's hybrid rule (lower
bound: a value counts as miscomputed only if the true value appears nowhere). Family-bootstrap intervals."""
import sys; sys.path.insert(0, "/tmp/www2027/analysis/rev25")
from r_common import *
def cls_reader(vals, i, K, chosen):
    s = S[i]; calc = s["calc"]; eu = lambda o: s["eu"][np.isclose(s["bids"], o)][0]
    if np.isclose(chosen, calc): return "right"
    if calc not in vals: return "not_reached"
    if any(vals[o][0] >= vals[calc][0] for o in vals if o != calc and eu(o) < eu(calc)): return "arith"
    return "incomplete" if len(vals) < K else "passed"
R = {}
for m in MODELS:
    z = tf[(tf.model == m) & (tf.budget == 768)].reset_index(drop=True)
    cr = np.array([cls_reader(v, i, K, b) for v, i, K, b in zip(z.vals, z.scenario_id, z.K, z.chosen_bid)])
    gap = z.gap.values; fams = z.fam.values
    R[m] = dict(reader_only=dict(zip(CATS, fboot(lambda ids: np.array([gap[ids][cr[ids] == k].sum() / gap[ids].sum() for k in CATS]), fams))))
dump25("r2b_reader_only_split", R)
for m, r in R.items():
    print(LAB[m], {k: f"{100*v['pt']:.0f} [{100*v['ci'][0]:.0f}, {100*v['ci'][1]:.0f}]" for k, v in r["reader_only"].items()})
