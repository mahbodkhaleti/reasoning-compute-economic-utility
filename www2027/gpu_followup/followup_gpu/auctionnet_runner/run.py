"""CLI. Usage: python -m auctionnet_runner.run <cmd> [...]   (run from the followup_gpu folder)
  prepare-data | build-scenarios | run --dataset {auctionnet,ebay} --model M --run-type {forced,unforced1024,unforced4096,readout_ablation,equiv,prefix_test}
  status | estimate      Add --smoke for 2 scenarios x 1 order."""
from __future__ import annotations
import argparse, gzip, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "followup"))
from auctionnet_runner import engine as E   # noqa: E402

MODELS = ["phi4_mini_reasoning", "qwen3_4b", "qwen3_8b", "qwen3_14b", "deepseek_r1_llama_8b", "granite_3_2_8b_instruct"]
def cfg():
    import yaml
    return yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
def py_hf(model): return os.path.join(ROOT, "env_b" if model == "qwen3_14b" else "env_a", "bin", "python")
def py_v(): return os.path.join(ROOT, "env_v", "bin", "python")
def log(msg):
    print(msg, flush=True)
def paths(dataset, model, run, smoke):
    base = os.path.join(ROOT, "results", "smoke" if smoke else "", dataset if dataset != "ebay" else "extra/ebay_vllm")
    return {k: os.path.join(base, k, f"{model}_{run}.jsonl") for k in ("prep", "gen", "decisions")} | {"runlog": os.path.join(base, "runlog.jsonl"), "base": base}
def scen_csv(dataset): return os.path.join(ROOT, "results", "auctionnet", "scenarios.csv") if dataset == "auctionnet" else os.path.join(ROOT, "data", "scenarios.csv")
def sub(py, *args):
    r = subprocess.run([py, "-m", "auctionnet_runner.run", *args], cwd=ROOT, env={**os.environ, "PATH": os.path.join(ROOT, "env_v", "bin") + os.pathsep + os.environ.get("PATH", "")})
    if r.returncode: sys.exit(r.returncode)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    ap.add_argument("--dataset", default="auctionnet"); ap.add_argument("--model"); ap.add_argument("--run-type")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--phase"); ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args(); C = cfg()
    if a.cmd == "prepare-data":
        from auctionnet_runner import data as D
        T, diag = D.build_landscapes(C, log); out = os.path.join(ROOT, C["paths"]["out_dir"]); os.makedirs(out, exist_ok=True)
        T[["period", "split", "product", "m_any_cpm", "m_top_cpm", "pvalue_mean"]].to_parquet(os.path.join(out, "landscapes.parquet"))
        json.dump(diag, open(os.path.join(out, "data_diagnostics.json"), "w"), indent=1, default=float); return
    if a.cmd == "build-scenarios":
        import pandas as pd
        from auctionnet_runner import scenarios as SC
        out = os.path.join(ROOT, C["paths"]["out_dir"]); T = pd.read_parquet(os.path.join(out, "landscapes.parquet"))
        for col, name in (("m_any_cpm", "scenarios.csv"), ("m_top_cpm", "scenarios_m_top.csv")):
            S, info = SC.build(T, C, col, log); S.to_csv(os.path.join(out, name), index=False)
            chk = SC.calculator_check(S); log(f"{name}: {chk}")
            json.dump(dict(info=info, check=chk), open(os.path.join(out, name.replace(".csv", "_info.json")), "w"), indent=1, default=float)
        return
    if a.cmd == "status":
        import glob
        for f in sorted(glob.glob(os.path.join(ROOT, "results", "**", "decisions", "*.jsonl"), recursive=True)):
            rows = E.read_jsonl(f); by = {}
            for r in rows: by[r["run_type"]] = by.get(r["run_type"], 0) + 1
            log(f"{os.path.relpath(f, ROOT)}: {by}  (expected 648 per run_type)")
        return
    if a.cmd == "estimate":
        rl = os.path.join(ROOT, "results", "smoke", "auctionnet", "runlog.jsonl")
        for r in E.read_jsonl(rl):
            if r["phase"] == "G" and r["tokens"]:
                tps = r["tokens"] / r["wall_s"]
                full_forced = 648 * (768 + 6 * 20); full_unf = 648 * 1024
                log(f"{r['model']}: smoke {r['tokens']} tokens in {r['wall_s']:.0f}s ({tps:.0f} tok/s at smoke batch size; full batches are faster). "
                    f"Upper bounds at this rate: forced {full_forced/tps/60:.0f} min, unforced1024 {full_unf/tps/60:.0f} min, unforced4096 {4*full_unf/tps/60:.0f} min")
        return
    if a.cmd != "run": sys.exit("unknown command")
    m, rt, ds = a.model, a.run_type, a.dataset; P = paths(ds, m, rt, a.smoke); sc = scen_csv(ds)
    limit = 2 if a.smoke else None; seeds = (11,) if a.smoke else tuple(C["eval"]["repeat_seeds"])
    if a.phase == "P":
        only = None
        if rt in ("equiv", "prefix_test"):
            ref = E.read_jsonl(os.path.join(ROOT, "data", "ebay_raw768", f"{m}.jsonl.gz"))
            n = a.n or (40 if rt == "equiv" else 20)
            only = {(r["scenario_id"], int(r["repeat_seed"])) for r in sorted(ref, key=lambda r: (r["scenario_id"], r["repeat_seed"]))[:n]}
            limit, seeds = None, (11, 23, 37)
        meta = E.phase_prepare(m, sc, P["prep"], limit=limit, seeds=seeds, only=only); log(f"prepared {meta['n']} decisions"); return
    if a.phase == "G":
        cap = {"unforced1024": 1024, "unforced4096": 4096}.get(rt)
        E.phase_generate(P["prep"], P["gen"], forced_max=None if cap else 768, unforced_cap=cap, vcfg=C["eval"]["vllm"], runlog=P["runlog"],
                         direct_budgets=[256, 512] if rt == "prefix_test" else None, budgets=(768,) if rt == "equiv" else tuple(C["eval"]["budgets"]),
                         prefix_reuse=bool(C["eval"].get("prefix_reuse", True)))
        return
    if a.phase == "R":
        cap = {"unforced1024": 1024, "unforced4096": 4096}.get(rt)
        E.phase_readout(m, P["prep"], P["gen"], sc, P["decisions"], ds, rt, unforced_cap=cap, runlog=P["runlog"],
                        budgets=(768,) if rt == "equiv" else tuple(C["eval"]["budgets"])); return
    if a.phase == "ABL":
        from auctionnet_runner import ablation as AB
        import harness_ext as ext, pandas as pd
        common, cfgp = ext.load_pipeline(m); conf = common.load_config(cfgp); spec = common.find_model(conf, m)
        S = {str(s["scenario_id"]).zfill(12): s for s in pd.read_csv(sc, dtype={"scenario_id": str}).to_dict("records")}
        if ds == "auctionnet":
            # AuctionNet: the saved 768-token forced traces from P0 (decisions file); order_id plays the role of repeat_seed
            dec = os.path.join(ROOT, "results", "auctionnet", "decisions", f"{m}_forced.jsonl")
            ref = [dict(scenario_id=r["scenario_id"], repeat_seed=int(r["order_id"]), option_order=r["option_order"], reasoning_text=r["reasoning_text"],
                        constrained_option_label=r["chosen_label"]) for r in E.read_jsonl(dec) if r["run_type"] == "forced_c768"]
            if len(ref) != 648 and not a.smoke: sys.exit(f"{m}: expected 648 forced_c768 decisions, found {len(ref)}")
            sub_dir = os.path.join("auctionnet", "R1_readout_ablation")
        else:
            ref = E.read_jsonl(os.path.join(ROOT, "data", "ebay_raw768", f"{m}.jsonl.gz"))
            sub_dir = os.path.join("extra", "R1_readout_ablation")
        if a.smoke: ref = ref[:2]
        out = os.path.join(ROOT, "results", "smoke" if a.smoke else "", sub_dir, f"{m}.jsonl")
        done = {(r["scenario_id"], r["repeat_seed"]) for r in E.read_jsonl(out)}
        h = common.ModelHarness(spec, conf); t0 = time.time(); batch = []
        for r in ref:
            if (r["scenario_id"], r["repeat_seed"]) in done: continue
            s = dict(S[r["scenario_id"]]); s["scenario_id"] = r["scenario_id"]; cands = common.shuffled_candidates(s, int(r["repeat_seed"]))
            assert [c["bid"] for c in cands] == r["option_order"]
            prompt = common.auction_prompt(float(s["value"]), cands)
            orig = h.constrained_bid_readout(prompt, r["reasoning_text"], cands, "reasoning")
            sfx = AB.alt_suffix(m, r["reasoning_text"]); alt = AB.readout_with_suffix(h, prompt, r["reasoning_text"], cands, sfx)
            batch.append(dict(model=m, scenario_id=r["scenario_id"], repeat_seed=r["repeat_seed"], saved_label=r["constrained_option_label"],
                              orig_label=orig["option_label"], orig_bid=orig["chosen_bid"], alt_suffix=sfx, alt_label=alt["label"], alt_bid=alt["bid"],
                              alt_probs=alt["probs"], orig_probs=orig["option_probabilities"], **{f"alt_{k}": v for k, v in common.score_bid(alt["bid"], cands).items()}))
            if len(batch) >= 50: E.append(out, batch); batch = []
        E.append(out, batch); log(f"R1 {m}: done in {time.time()-t0:.0f}s"); return
    # orchestrate phases in the right environments
    if rt == "readout_ablation":
        sub(py_hf(m), "run", "--dataset", ds, "--model", m, "--run-type", rt, "--phase", "ABL", *(["--smoke"] if a.smoke else [])); return
    common_args = ["run", "--dataset", ds, "--model", m, "--run-type", rt] + (["--smoke"] if a.smoke else []) + (["--n", str(a.n)] if a.n else [])
    if not os.path.exists(P["prep"]): sub(py_hf(m), *common_args, "--phase", "P")
    sub(py_v(), *common_args, "--phase", "G")
    if rt == "prefix_test":
        G = E.read_jsonl(P["gen"]); f = {(g["scenario_id"], g["order_id"]): g["ids"] for g in G if g["kind"] == "forced"}
        res = {c: sum(f[(g["scenario_id"], g["order_id"])][:c] == g["ids"] for g in G if g["kind"] == "direct" and g["budget"] == c) for c in (256, 512)}
        n = sum(1 for g in G if g["kind"] == "direct" and g["budget"] == 256)
        log(f"PREFIX TEST {m}: identical prefixes at 256: {res[256]}/{n}, at 512: {res[512]}/{n}")
        E.append(os.path.join(ROOT, "results", "extra", "prefix_test.jsonl"), [dict(model=m, n=n, same256=res[256], same512=res[512])]); return
    sub(py_hf(m), *common_args, "--phase", "R")
    if rt == "equiv":
        ref = {(r["scenario_id"], int(r["repeat_seed"])): r for r in E.read_jsonl(os.path.join(ROOT, "data", "ebay_raw768", f"{m}.jsonl.gz"))}
        D = [d for d in E.read_jsonl(P["decisions"]) if d["budget"] == 768]
        same_l = sum(d["chosen_label"] == ref[(d["scenario_id"], d["order_id"])]["constrained_option_label"] for d in D)
        same_t = sum(d["reasoning_text"] == ref[(d["scenario_id"], d["order_id"])]["reasoning_text"] for d in D)
        log(f"EQUIVALENCE {m}: same letter {same_l}/{len(D)}, identical trace {same_t}/{len(D)}" + ("   <-- BELOW 90%: STOP AND REPORT" if same_l < 0.9 * len(D) else ""))
        E.append(os.path.join(ROOT, "results", "extra", "equivalence.jsonl"), [dict(model=m, n=len(D), same_letter=same_l, same_trace=same_t)])

if __name__ == "__main__":
    main()
