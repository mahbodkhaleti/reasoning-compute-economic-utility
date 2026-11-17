"""Generation in three phases so that everything except the batched decoding is the eBay code itself.
  P (HF env, CPU): build decisions with common.shuffled_candidates / common.auction_prompt, and prompt token ids with
     ModelHarness._chat_text + tokenizer(prefix) exactly as auction_decision does; record the end-of-sequence ids.
  G (vLLM env, GPU): from token ids only (no tokenizer): one forced greedy trace per decision (min=max tokens, which
     bans every end-of-sequence id like bad_words_ids + min_new_tokens), the 20-token FINAL BID answer for every budget
     (prefix reuse), and unforced traces (max_tokens = cap, stopping on the generation-config end-of-sequence ids).
  R (HF env, GPU): decode with skip_special_tokens, letter readout with ModelHarness.constrained_bid_readout,
     parse_final_bid and score_bid, one JSONL row per decision and budget, keyed and resumable."""
from __future__ import annotations
import json, os, sys, time, gzip
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "followup"))

def read_jsonl(p):
    if not os.path.exists(p): return []
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f: return [json.loads(x) for x in f if x.strip()]

def append(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

def key(r): return (r["dataset"], r["model"], r["run_type"], r["scenario_id"], int(r["order_id"]), int(r.get("seed", 0)))

# ------------------------------------------------------------------ phase P
class _Stub:  # carries what ModelHarness._chat_text needs
    def __init__(self, tok, spec): self.tokenizer, self.spec = tok, spec

def phase_prepare(model, scen_csv, out, limit=None, seeds=(11, 23, 37), only=None):
    import harness_ext as ext, pandas as pd
    from transformers import AutoTokenizer, GenerationConfig
    common, cfgp = ext.load_pipeline(model); conf = common.load_config(cfgp); spec = common.find_model(conf, model)
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec.get("tokenizer_revision", spec.get("revision")), trust_remote_code=True)
    try: gen_eos = GenerationConfig.from_pretrained(spec["hf_id"], revision=spec.get("revision")).eos_token_id
    except Exception: gen_eos = None
    eos = set([tok.eos_token_id] if isinstance(tok.eos_token_id, int) else (tok.eos_token_id or []))
    eos |= set([gen_eos] if isinstance(gen_eos, int) else (gen_eos or []))
    S = pd.read_csv(scen_csv, dtype={"scenario_id": str}).to_dict("records")
    for s in S: s["scenario_id"] = str(s["scenario_id"]).zfill(12)
    if limit: S = S[:limit]
    stub = _Stub(tok, spec); rows = []
    for s in S:
        for sd in seeds:
            if only is not None and (s["scenario_id"], sd) not in only: continue
            cands = common.shuffled_candidates(s, sd)
            prompt = common.auction_prompt(float(s["value"]), cands)
            prefix = common.ModelHarness._chat_text(stub, prompt, "reasoning")
            rows.append(dict(scenario_id=s["scenario_id"], order_id=sd, prompt=prompt, prompt_ids=tok(prefix)["input_ids"],
                             option_order=[c["bid"] for c in cands]))
    suffix = tok("\n\nNow give the required answer. FINAL BID:", add_special_tokens=False)["input_ids"]
    meta = dict(model=model, hf_id=spec["hf_id"], revision=spec.get("revision"), eos_ids=sorted(int(x) for x in eos),
                answer_suffix_ids=suffix, n=len(rows))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(json.dumps(meta) + "\n")
        for r in rows: f.write(json.dumps(r) + "\n")
    return meta

# ------------------------------------------------------------------ phase G
def phase_generate(prep, out, forced_max=768, budgets=(0, 64, 128, 256, 512, 768), unforced_cap=None, answer_tokens=20,
                   vcfg=None, runlog=None, direct_budgets=None, prefix_reuse=True):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    lines = read_jsonl(prep); meta, rows = lines[0], lines[1:]
    done = {(r["scenario_id"], r["order_id"], r["kind"], r.get("budget")) for r in read_jsonl(out)}
    vcfg = vcfg or {}
    need = [("forced", forced_max)] if forced_max else []
    need += [("forced", c) for c in budgets if forced_max and not prefix_reuse and 0 < c < forced_max]
    need += [("answer", c) for c in budgets] if forced_max else []
    need += [("direct", c) for c in (direct_budgets or [])] + ([("unforced", unforced_cap)] if unforced_cap else [])
    if all((r["scenario_id"], r["order_id"], k, c) in done for r in rows for k, c in need):
        print("generation already complete; skipping vLLM"); return
    t0 = time.time()
    llm = LLM(model=meta["hf_id"], revision=meta["revision"], dtype="bfloat16", skip_tokenizer_init=True, seed=0,
              generation_config="vllm", max_num_seqs=vcfg.get("max_num_seqs", 128),
              gpu_memory_utilization=vcfg.get("gpu_memory_utilization", 0.88), max_model_len=vcfg.get("max_model_len", 6144))
    load_s = time.time() - t0; eos = meta["eos_ids"]; ntok = 0; t1 = time.time()
    def greedy(n, minn): return SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=n, min_tokens=minn,
                                               stop_token_ids=eos, detokenize=False, seed=0)
    if forced_max:
        todo = [r for r in rows if (r["scenario_id"], r["order_id"], "forced", forced_max) not in done]
        if todo:
            outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in todo], greedy(forced_max, forced_max))
            gen = [dict(scenario_id=r["scenario_id"], order_id=r["order_id"], kind="forced", budget=forced_max, ids=list(o.outputs[0].token_ids)) for r, o in zip(todo, outs)]
            ntok += sum(len(g["ids"]) for g in gen); append(out, gen)
        if not prefix_reuse:   # vLLM batching broke prefix consistency: generate every budget separately
            for c in [b for b in budgets if 0 < b < forced_max]:
                todo = [r for r in rows if (r["scenario_id"], r["order_id"], "forced", c) not in done]
                if todo:
                    outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in todo], greedy(c, c))
                    gen = [dict(scenario_id=r["scenario_id"], order_id=r["order_id"], kind="forced", budget=c, ids=list(o.outputs[0].token_ids)) for r, o in zip(todo, outs)]
                    ntok += sum(len(g["ids"]) for g in gen); append(out, gen)
        allg = read_jsonl(out)
        forced = {(g["scenario_id"], g["order_id"], g["budget"]): g["ids"] for g in allg if g["kind"] == "forced"}
        def trace(r, c):
            k = (r["scenario_id"], r["order_id"])
            return [] if c == 0 else (forced[k + (c,)] if (k + (c,)) in forced else forced[k + (forced_max,)][:c])
        # 20-token FINAL BID answers for every budget
        jobs = [(r, c) for r in rows for c in budgets if (r["scenario_id"], r["order_id"], "answer", c) not in done]
        if jobs:
            outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"] + trace(r, c) + meta["answer_suffix_ids"]) for r, c in jobs],
                                SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=answer_tokens, stop_token_ids=eos, detokenize=False, seed=0))
            gen = [dict(scenario_id=r["scenario_id"], order_id=r["order_id"], kind="answer", budget=c, ids=list(o.outputs[0].token_ids)) for (r, c), o in zip(jobs, outs)]
            ntok += sum(len(g["ids"]) for g in gen); append(out, gen)
    for c in (direct_budgets or []):   # used only by the prefix-equivalence test
        todo = [r for r in rows if (r["scenario_id"], r["order_id"], "direct", c) not in done]
        outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in todo], greedy(c, c))
        append(out, [dict(scenario_id=r["scenario_id"], order_id=r["order_id"], kind="direct", budget=c, ids=list(o.outputs[0].token_ids)) for r, o in zip(todo, outs)])
    if unforced_cap:
        todo = [r for r in rows if (r["scenario_id"], r["order_id"], "unforced", unforced_cap) not in done]
        if todo:
            outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in todo], greedy(unforced_cap, 0))
            gen = [dict(scenario_id=r["scenario_id"], order_id=r["order_id"], kind="unforced", budget=unforced_cap,
                        ids=list(o.outputs[0].token_ids), finish=o.outputs[0].finish_reason) for r, o in zip(todo, outs)]
            ntok += sum(len(g["ids"]) for g in gen); append(out, gen)
    if runlog:
        import torch, vllm
        append(runlog, [dict(phase="G", model=meta["model"], gpu=torch.cuda.get_device_name(0), vllm=vllm.__version__, torch=torch.__version__,
                             max_num_seqs=vcfg.get("max_num_seqs", 128), tokens=ntok, wall_s=time.time() - t1, load_s=load_s,
                             forced_max=forced_max, unforced_cap=unforced_cap, n=len(rows), time=time.strftime("%Y-%m-%dT%H:%M:%S"))])

# ------------------------------------------------------------------ phase R
def phase_readout(model, prep, gen, scen_csv, out, dataset, run_type, budgets=(0, 64, 128, 256, 512, 768), unforced_cap=None, runlog=None):
    import harness_ext as ext, pandas as pd, torch
    common, cfgp = ext.load_pipeline(model); conf = common.load_config(cfgp); spec = common.find_model(conf, model)
    lines = read_jsonl(prep); rows = lines[1:]
    S = {str(s["scenario_id"]).zfill(12): s for s in pd.read_csv(scen_csv, dtype={"scenario_id": str}).to_dict("records")}
    G = {}
    for g in read_jsonl(gen): G[(g["scenario_id"], g["order_id"], g["kind"], g["budget"])] = g
    done = {key(r) for r in read_jsonl(out)}
    h = common.ModelHarness(spec, conf); tok = h.tokenizer; t0 = time.time(); n = 0
    for r in rows:
        s = dict(S[r["scenario_id"]]); s["scenario_id"] = r["scenario_id"]; cands = common.shuffled_candidates(s, r["order_id"])
        assert [c["bid"] for c in cands] == r["option_order"]
        settings = [("unforced", unforced_cap)] if unforced_cap else [("forced", b) for b in budgets]
        batch = []
        for kind, b in settings:
            k = dict(dataset=dataset, model=model, run_type=run_type if kind == "unforced" else f"{run_type}_c{b}", scenario_id=r["scenario_id"], order_id=r["order_id"], seed=0)
            if key(k) in done: continue
            if kind == "forced":
                gk = (r["scenario_id"], r["order_id"], "forced", b)
                ids = [] if b == 0 else (G[gk]["ids"] if gk in G else G[(r["scenario_id"], r["order_id"], "forced", 768)]["ids"][:b])
                text = tok.decode(ids, skip_special_tokens=True); ans = G[(r["scenario_id"], r["order_id"], "answer", b)]["ids"]
                final = tok.decode(ans, skip_special_tokens=True); hit = None
            else:
                gg = G[(r["scenario_id"], r["order_id"], "unforced", b)]; ids = gg["ids"]
                text = tok.decode(ids, skip_special_tokens=True); final = text; hit = gg.get("finish") == "length"
            ro = h.constrained_bid_readout(r["prompt"], text, cands, "reasoning")
            k.update(category=s["category"], value=float(s["value"]), option_count=int(s["option_count"]), budget=b if kind == "forced" else -1,
                     unforced_cap=b if kind == "unforced" else None, option_order=r["option_order"], reasoning_tokens=len(ids), hit_cap=hit,
                     reasoning_text=text, freeform_final_text=final, freeform_chosen_bid=common.parse_final_bid(final, cands),
                     chosen_label=ro["option_label"], option_probabilities=ro["option_probabilities"],
                     **common.score_bid(ro["chosen_bid"], cands))
            k["chosen_bid"] = ro["chosen_bid"]
            batch.append(k); n += 1
        append(out, batch)
    if runlog:
        append(runlog, [dict(phase="R", model=model, gpu=torch.cuda.get_device_name(0), rows=n, wall_s=time.time() - t0, run_type=run_type)])
