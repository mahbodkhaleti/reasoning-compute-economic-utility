#!/usr/bin/env python3
"""Run 2a helper: find the letters A-Z that are single tokens for every Run 2a model.

Loads tokenizers only (no GPU). Writes results/followup/letters.json, which
run_followup.py --run run2a reads. Run with env_a.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness_ext as ext  # noqa: E402

MODELS = ["phi4_mini_reasoning", "granite_3_2_8b_instruct", "deepseek_r1_llama_8b"]


def main() -> None:
    from transformers import AutoTokenizer

    per_model = {}
    for slug in MODELS:
        common, cfg_path = ext.load_pipeline(slug)
        spec = common.find_model(common.load_config(cfg_path), slug)
        tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec.get("tokenizer_revision", spec.get("revision")),
                                            trust_remote_code=True)
        per_model[slug] = ext.single_token_letters(tok)
        print(slug, "".join(per_model[slug]))
    allowed = sorted(set.intersection(*(set(v) for v in per_model.values())))
    excluded = sorted(set(chr(c) for c in range(65, 91)) - set(allowed))
    out = HERE.parent / "results" / "followup" / "letters.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"allowed_letters": allowed, "excluded_letters": excluded, "per_model": per_model}, indent=2))
    print("allowed:", "".join(allowed), "| excluded:", "".join(excluded) or "none", "->", out)


if __name__ == "__main__":
    main()
