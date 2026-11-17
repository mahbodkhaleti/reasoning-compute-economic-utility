#!/usr/bin/env python3
"""Runner for the GPU follow-up experiments.

Every decision goes through the paper's own code: common.shuffled_candidates
(same option orders and seeds), common.auction_prompt, ModelHarness.auction_decision
(same chat template, thinking flag, greedy decoding, EOS ban for forced budgets),
ModelHarness.constrained_bid_readout, common.parse_final_bid and common.score_bid.

Runs
  drift768   paper protocol at a forced 768 budget; compare with the paper's saved traces
  run1       natural stopping: one unforced generation with cap --cap (4096, or 8192 for reruns)
  run4       prompt baseline: one added instruction; forced 768 and unforced (cap 1024)
  run2a      letters drawn at random, independent of position; forced 768 and unforced (cap 1024)

Output: results/followup/<run_name>/<model>.jsonl (one row per decision), plus
<model>_manifest.json. With --smoke: results/followup/smoke/<run_name>/ and only
the first 10 scenarios with order seed 11. Rerunning the same command resumes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import harness_ext as ext  # noqa: E402

RUN_MODELS = {
    "drift768": list(ext.MODEL_PACKAGE),
    "run1": ["qwen3_4b", "qwen3_8b", "qwen3_14b", "deepseek_r1_llama_8b"],
    "run4": ["phi4_mini_reasoning", "granite_3_2_8b_instruct", "deepseek_r1_llama_8b", "qwen3_8b"],
    "run2a": ["phi4_mini_reasoning", "granite_3_2_8b_instruct", "deepseek_r1_llama_8b"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def settings_for(run: str, cap: int) -> list[tuple[str, int, int]]:
    """(setting name, budget passed to auction_decision, unforced cap)."""
    if run == "drift768":
        return [("forced768", 768, 1024)]
    if run == "run1":
        return [(f"unforced{cap}", -1, cap)]
    if run in ("run4", "run2a"):
        return [("forced768", 768, 1024), ("unforced1024", -1, 1024)]
    raise ValueError(run)


def run_name_for(run: str, cap: int) -> str:
    return {
        "drift768": "drift768",
        "run1": f"run1_nat{cap}",
        "run4": "run4_prompt",
        "run2a": "run2a_randletters",
    }[run]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(RUN_MODELS))
    ap.add_argument("--model", required=True, choices=list(ext.MODEL_PACKAGE))
    ap.add_argument("--cap", type=int, default=4096, help="Run 1 unforced token cap")
    ap.add_argument("--only-capped-from", default=None,
                    help="Run 1 rerun: only decisions that hit the cap in this earlier JSONL")
    ap.add_argument("--smoke", action="store_true", help="10 scenarios x order seed 11, separate output folder")
    ap.add_argument("--results-root", default=str(ROOT / "results" / "followup"))
    ap.add_argument("--letters-file", default=None, help="Run 2a: JSON from check_letters.py")
    ap.add_argument("--allow-version-mismatch", action="store_true")
    ap.add_argument("--allow-any-model", action="store_true")
    args = ap.parse_args()

    if args.model not in RUN_MODELS[args.run] and not args.allow_any_model:
        sys.exit(f"{args.model} is not part of {args.run} in the spec; pass --allow-any-model to override")

    import pandas as pd
    import torch

    versions = {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
                "accelerate": importlib.metadata.version("accelerate"), "python": platform.python_version(),
                "cuda_build": torch.version.cuda}
    expected = ext.EXPECTED_VERSIONS[args.model]
    mismatch = {k: (versions[k], v) for k, v in expected.items() if not versions[k].startswith(v)}
    if mismatch and not args.allow_version_mismatch:
        sys.exit(f"Library versions differ from the paper run for {args.model}: {mismatch}. "
                 f"Use the environment from setup_pod.sh (env_a for five models, env_b for qwen3_14b).")

    common, config_path = ext.load_pipeline(args.model)
    config = common.load_config(config_path)
    spec = common.find_model(config, args.model)
    run_name = run_name_for(args.run, args.cap)
    out_dir = Path(args.results_root) / ("smoke" if args.smoke else "") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}.jsonl"
    manifest_path = out_dir / f"{args.model}_manifest.json"

    scenarios_path = ROOT / "data" / "scenarios.csv"
    scenarios = pd.read_csv(scenarios_path, dtype={"scenario_id": str}).to_dict("records")
    for s in scenarios:
        s["scenario_id"] = str(s["scenario_id"]).zfill(12)
    repeat_seeds = list(config["evaluation"]["repeat_seeds"])
    if args.smoke:
        scenarios, repeat_seeds = scenarios[:10], [11]
    settings = settings_for(args.run, args.cap)

    only_keys = None
    if args.only_capped_from:
        prior = common.read_jsonl(args.only_capped_from)
        only_keys = {(r["scenario_id"], int(r["repeat_seed"])) for r in prior if r.get("hit_cap")}
        print(f"Rerun limited to {len(only_keys)} decisions that hit the cap in {args.only_capped_from}")

    allowed_letters = None
    if args.run == "run2a":
        letters_file = Path(args.letters_file or (Path(args.results_root) / "letters.json"))
        if not letters_file.exists():
            sys.exit(f"Run 2a needs {letters_file}; run check_letters.py first")
        allowed_letters = json.loads(letters_file.read_text())["allowed_letters"]

    required = [(name, s["scenario_id"], seed) for name, _, _ in settings for s in scenarios for seed in repeat_seeds
                if only_keys is None or (s["scenario_id"], seed) in only_keys]
    done = {(r["setting"], r["scenario_id"], int(r["repeat_seed"])) for r in common.read_jsonl(out_path)}
    todo = [k for k in required if k not in done]
    print(f"{run_name} {args.model}: {len(required)} decisions, {len(required) - len(todo)} already done")

    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "run_name": run_name, "model": args.model, "hf_id": spec["hf_id"],
        "requested_model_revision": spec.get("revision"),
        "requested_tokenizer_revision": spec.get("tokenizer_revision", spec.get("revision")),
        "pipeline_package": ext.MODEL_PACKAGE[args.model],
        "pipeline_common_sha256": sha256(Path(common.__file__)),
        "pipeline_config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)), "harness_ext_sha256": sha256(HERE / "harness_ext.py"),
        "scenarios_sha256": sha256(scenarios_path), "settings": settings, "repeat_seeds": repeat_seeds,
        "smoke": args.smoke, "first_started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        **versions,
    }
    manifest.setdefault("starts_utc", []).append(dt.datetime.now(dt.timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if not todo:
        print("Nothing to do.")
        return

    for name, budget, cap in settings:
        if budget < 0:
            config["evaluation"]["unconstrained_max_tokens"] = cap  # read by auction_decision
    harness = common.ModelHarness(spec, config)
    if harness.model_revision and spec.get("revision") and harness.model_revision != spec["revision"]:
        sys.exit(f"Resolved revision {harness.model_revision} != pinned {spec['revision']}")
    manifest.update({"gpu": torch.cuda.get_device_name(0), "resolved_model_revision": harness.model_revision,
                     "resolved_tokenizer_revision": harness.tokenizer_revision})
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Loaded {spec['hf_id']} on {manifest['gpu']}")

    todo_set = set(todo)
    t_run = time.time()
    n_done = 0
    for name, budget, cap in settings:
        config["evaluation"]["unconstrained_max_tokens"] = cap
        for scenario in scenarios:
            for seed in repeat_seeds:
                key = (name, scenario["scenario_id"], seed)
                if key not in todo_set:
                    continue
                candidates = common.shuffled_candidates(scenario, seed)
                if allowed_letters is not None:
                    labels = ext.random_labels(scenario["scenario_id"], seed, len(candidates), allowed_letters, common.stable_int)
                else:
                    labels = ext.default_labels(len(candidates))
                variant = "list_profits" if args.run == "run4" else "paper"
                prompt = ext.build_prompt(common, float(scenario["value"]), candidates, labels, variant)
                gen_seed = common.stable_int(config["seed"], args.model, "reasoning", scenario["scenario_id"], budget, seed)
                t0 = time.time()
                gen = harness.auction_decision(prompt, budget, gen_seed, "reasoning", sample=False)
                if labels == ext.default_labels(len(candidates)):
                    ro = harness.constrained_bid_readout(prompt, gen["reasoning_text"], candidates, "reasoning")
                    method = "constrained_option_letter_next_token"
                else:
                    ro = ext.labeled_readout(harness, prompt, gen["reasoning_text"], candidates, labels, "reasoning", common.GPUMeter)
                    method = "constrained_random_letter_next_token"
                wall = time.time() - t0
                chosen = ro["chosen_bid"]
                freeform_chosen = common.parse_final_bid(gen["final_text"], candidates)
                gm, rm = common.meter_fields(gen["meter"], config), common.meter_fields(ro["meter"], config)
                energies = [x for x in (gm["energy_joules"], rm["energy_joules"]) if isinstance(x, float) and math.isfinite(x)]
                hit_cap = budget < 0 and gen["reasoning_tokens"] >= cap
                row = {
                    "run": run_name, "setting": name, "model": args.model, "hf_id": spec["hf_id"],
                    "model_revision": spec.get("revision"), "prompt_variant": variant,
                    "scenario_id": scenario["scenario_id"], "category": scenario["category"],
                    "value": float(scenario["value"]), "option_count": int(scenario["option_count"]),
                    "candidate_set_index": int(scenario["candidate_set_index"]),
                    "repeat_seed": int(seed), "option_order_seed": int(seed), "generation_seed": int(gen_seed),
                    "budget": int(budget), "unforced_cap": int(cap) if budget < 0 else None,
                    "option_order": [c["bid"] for c in candidates], "option_labels": labels,
                    "label_of_bid": {f"{c['bid']:.2f}": l for c, l in zip(candidates, labels)},
                    "reasoning_tokens": gen["reasoning_tokens"], "hit_cap": bool(hit_cap),
                    "stopped_naturally": bool(budget < 0 and not hit_cap),
                    "reasoning_text": gen["reasoning_text"], "freeform_final_text": gen["final_text"],
                    "freeform_chosen_bid": freeform_chosen, "freeform_valid": freeform_chosen is not None,
                    "readout_method": method, "chosen_label": ro["option_label"],
                    "chosen_label_position": labels.index(ro["option_label"]),
                    "option_probabilities": ro["option_probabilities"], "option_token_ids": ro["option_token_ids"],
                    "readout_context_tokens": ro["context_tokens"],
                    **common.score_bid(chosen, candidates),
                    "wall_seconds": wall,
                    "generation_seconds": gm["elapsed_seconds"], "readout_seconds": rm["elapsed_seconds"],
                    "generation_energy_joules": gm["energy_joules"], "readout_energy_joules": rm["energy_joules"],
                    "energy_joules": sum(energies) if energies else math.nan,
                    "peak_memory_bytes": max(gm["peak_memory_bytes"], rm["peak_memory_bytes"]),
                    "gpu": manifest["gpu"], "torch": versions["torch"], "transformers": versions["transformers"],
                }
                common.append_jsonl(out_path, row)
                n_done += 1
                rate = (time.time() - t_run) / n_done
                print(f"[{n_done}/{len(todo)}] {name} {scenario['scenario_id']} seed={seed} tokens={gen['reasoning_tokens']}"
                      f"{' CAP' if hit_cap else ''} q={row['test_utility_ratio']:.3f} {wall:.1f}s "
                      f"eta={rate * (len(todo) - n_done) / 3600:.1f}h", flush=True)

    manifest["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
