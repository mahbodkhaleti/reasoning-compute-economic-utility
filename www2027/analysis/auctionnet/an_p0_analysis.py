"""AuctionNet P0 (A1 forced, A2 unforced 1,024) + eBay vLLM baseline for Phi-mini and Granite-8B.
Reads the saved decision files; writes results/AN/an_p0_summary.json. Family = (category, value), resampled within category."""
import json, sys, numpy as np, pandas as pd
B = sys.argv[1] if len(sys.argv) > 1 else "/tmp/www2027/results/AN/ov/results"
MODELS = ["deepseek_r1_llama_8b", "phi4_mini_reasoning", "qwen3_4b", "qwen3_8b", "qwen3_14b", "granite_3_2_8b_instruct"]
BUD = [0, 64, 128, 256, 512, 768]
rng = np.random.default_rng(0)
def load(ds, m, rt):
    p = f"{B}/{ds}/decisions/{m}_{rt}.jsonl"
    return pd.DataFrame([json.loads(l) for l in open(p)])
def scen_table(ds):
    if ds != "auctionnet": return None
    S = pd.read_csv(f"{B}/auctionnet/scenarios.csv", dtype={"scenario_id": str})
    rows = []
    for _, s in S.iterrows():
        C = json.loads(s.candidates_json); tu = np.array([c["test_u"] for c in C]); tr = np.array([c["train_u"] for c in C])
        rows.append(dict(scenario_id=str(s.scenario_id).zfill(12), category=s.category, value=s.value, q_calc=tu[int(np.argmax(tr))] / tu.max(),
                         q_rand=float(np.mean(tu / tu.max())), u_calc=tu[int(np.argmax(tr))], u_max=tu.max(), top_is_calc=float(C[int(np.argmax(tr))]["bid"] == max(c["bid"] for c in C))))
    return pd.DataFrame(rows)
def fboot(df, stat, n=2000):
    fam = df.groupby(["category", "value"]).ngroup().values; cats = df.category.values
    groups = {c: np.unique(fam[cats == c]) for c in np.unique(cats)}
    idx_of = {f: np.where(fam == f)[0] for f in np.unique(fam)}
    pt = stat(np.arange(len(df))); out = []
    for _ in range(n):
        ids = np.concatenate([idx_of[f] for c, fs in groups.items() for f in rng.choice(fs, len(fs))])
        out.append(stat(ids))
    out = np.array(out); return dict(pt=float(pt), lo=float(np.percentile(out, 2.5)), hi=float(np.percentile(out, 97.5)))
OUT = {}
for ds in ("auctionnet", "extra/ebay_vllm"):
    ST = scen_table(ds); res = {}
    ms = MODELS if ds == "auctionnet" else ["phi4_mini_reasoning", "granite_3_2_8b_instruct"]
    for m in ms:
        F = load(ds, m, "forced"); U = load(ds, m, "unforced1024")
        F["scenario_id"] = F.scenario_id.astype(str); U["scenario_id"] = U.scenario_id.astype(str)
        key = ["scenario_id", "order_id"]
        W = F.pivot_table(index=key, columns="budget", values="test_utility_ratio").reset_index()
        Wu = F.pivot_table(index=key, columns="budget", values="test_utility").reset_index()
        W = W.merge(U[key + ["test_utility_ratio", "test_utility", "reasoning_tokens", "hit_cap"]].rename(columns={"test_utility_ratio": "q_unf", "test_utility": "u_unf"}), on=key)
        W = W.merge(Wu.rename(columns={b: f"u{b}" for b in BUD}), on=key)
        if ST is not None:
            W = W.merge(ST, on="scenario_id")
        else:
            opt = F[F.budget == 0][key + ["test_optimal_utility", "category", "value"]]
            W = W.merge(opt, on=key)
            W["q_calc"] = np.nan  # eBay: compare q only (calculator scoring lives in the paper pipeline)
        r = dict(q={b: float(W[b].mean()) for b in BUD}, q_unf=float(W.q_unf.mean()),
                 unf_mean_tokens=float(U.reasoning_tokens.mean()), unf_hit_cap=float(U.hit_cap.mean()),
                 valid=float(F.valid_output.mean()))
        if ST is not None:
            r["q_calc"] = float(W.q_calc.mean()); r["q_rand"] = float(W.q_rand.mean())
            g = lambda col: (lambda ids: (W[col].values[ids].mean() - W[0].values[ids].mean()) / (W.q_calc.values[ids].mean() - W[0].values[ids].mean()))
            r["gap768"] = fboot(W, g(768)); r["gap_unf1024"] = fboot(W, g("q_unf"))
            r["gap_by_budget"] = {b: float((W[b].mean() - W[0].mean()) / (W.q_calc.mean() - W[0].mean())) for b in BUD}
            r["gap768_dollar"] = float((W.u768.sum() - W.u0.sum()) / (W.u_calc.sum() - W.u0.sum()))
            r["gap768_by_cat"] = {c: float((d[768].mean() - d[0].mean()) / (d.q_calc.mean() - d[0].mean())) for c, d in W.groupby("category")}
            # wrong answers at 768 that are the highest bid
            f7 = F[F.budget == 768].merge(ST[["scenario_id"]], on="scenario_id")
            S = pd.read_csv(f"{B}/auctionnet/scenarios.csv", dtype={"scenario_id": str}); S["scenario_id"] = S.scenario_id.str.zfill(12)
            menu = {s: json.loads(c) for s, c in zip(S.scenario_id, S.candidates_json)}
            calc = {s: max(C, key=lambda c: c["train_u"])["bid"] for s, C in menu.items()}
            wr = f7[[not np.isclose(b, calc[s]) for s, b in zip(f7.scenario_id, f7.chosen_bid)]]
            r["n_wrong768"] = int(len(wr))
            r["highest_share_wrong768"] = float(np.mean([np.isclose(b, max(c["bid"] for c in menu[s])) for s, b in zip(wr.scenario_id, wr.chosen_bid)]))
            r["highest_chance"] = float(np.mean([np.mean([np.isclose(c["bid"], max(x["bid"] for x in menu[s])) for c in menu[s] if not np.isclose(c["bid"], calc[s])]) for s in wr.scenario_id]))
        res[m] = r
    OUT[ds] = res
json.dump(OUT, open("/tmp/www2027/results/AN/an_p0_summary.json", "w"), indent=1)
for ds, res in OUT.items():
    print("==", ds)
    for m, r in res.items():
        line = f"{m:26s} q0 {r['q'][0]:.3f} q768 {r['q'][768]:.3f} unf {r['q_unf']:.3f} tok {r['unf_mean_tokens']:.0f} cap {r['unf_hit_cap']:.3f}"
        if "gap768" in r:
            G = lambda d: f"{100*d['pt']:.0f} [{100*d['lo']:.0f},{100*d['hi']:.0f}]"
            line += f" | calc {r['q_calc']:.3f} rand {r['q_rand']:.3f} gap768 {G(r['gap768'])} unf {G(r['gap_unf1024'])} $ {100*r['gap768_dollar']:.0f} hi {100*r['highest_share_wrong768']:.0f}/{100*r['highest_chance']:.0f} nw {r['n_wrong768']}"
            line += " bud " + " ".join(f"{100*v:.0f}" for v in r["gap_by_budget"].values())
        print(line)
