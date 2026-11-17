"""Small additions on top of the paper's pipeline (pipeline/orig or pipeline/new common.py).

Nothing here changes how the paper generated traces. Generation always goes
through the pipeline's own ModelHarness.auction_decision, and the default
letter readout goes through ModelHarness.constrained_bid_readout. The helpers
below only cover what the follow-up runs need and the paper code cannot do:

* which pipeline package and environment a model belongs to;
* an NVML meter that reads the GPU this process actually uses;
* prompt variants (Run 4 instruction, Run 2a letter labels);
* a letter readout over arbitrary labels (Run 2a). It is the paper's
  constrained_bid_readout with the label list as a parameter, nothing else.
"""
from __future__ import annotations

import importlib
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Which copy of the paper code ran each model, and with which config file.
# Phi, Qwen3-4B, DeepSeek and Qwen3-8B ran from robustness_extensions/scripts;
# Granite and Qwen3-14B ran from new_models_full_server_package/src.
MODEL_PACKAGE = {
    "phi4_mini_reasoning": "orig",
    "qwen3_4b": "orig",
    "deepseek_r1_llama_8b": "orig",
    "qwen3_8b": "orig",
    "granite_3_2_8b_instruct": "new",
    "qwen3_14b": "new",
}
PACKAGE_CONFIG = {"orig": "config.json", "new": "experiment.json"}

# Library versions recorded in the paper's run metadata.
EXPECTED_VERSIONS = {
    "phi4_mini_reasoning": {"torch": "2.13.0", "transformers": "5.15.0"},
    "qwen3_4b": {"torch": "2.13.0", "transformers": "5.15.0"},
    "deepseek_r1_llama_8b": {"torch": "2.13.0", "transformers": "5.15.0"},
    "qwen3_8b": {"torch": "2.13.0", "transformers": "5.15.0"},
    "granite_3_2_8b_instruct": {"torch": "2.13.0", "transformers": "5.15.0"},
    "qwen3_14b": {"torch": "2.9.1", "transformers": "5.1.0"},
}


def load_pipeline(model_slug: str):
    """Import the paper's common.py for this model and return (module, config path)."""
    package = MODEL_PACKAGE[model_slug]
    package_dir = ROOT / "pipeline" / package
    sys.path.insert(0, str(package_dir))
    if "common" in sys.modules:
        del sys.modules["common"]
    common = importlib.import_module("common")
    _patch_meter(common)
    return common, package_dir / PACKAGE_CONFIG[package]


def _nvml_index() -> int:
    explicit = os.environ.get("FOLLOWUP_NVML_INDEX")
    if explicit is not None:
        return int(explicit)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(visible) if visible.isdigit() else 0


def _patch_meter(common) -> None:
    """The paper's GPUMeter always reads NVML device 0. NVML ignores
    CUDA_VISIBLE_DEVICES, so on a multi-GPU pod a process on GPU 1 would log
    GPU 0's power. Point the handle at the GPU this process uses. On a
    one-GPU pod this changes nothing."""
    base = common.GPUMeter

    class FollowupGPUMeter(base):  # type: ignore[misc, valid-type]
        def __init__(self, poll_seconds: float = 0.1):
            super().__init__(poll_seconds)
            if getattr(self, "_nvml", None) is not None:
                try:
                    self._handle = self._nvml.nvmlDeviceGetHandleByIndex(_nvml_index())
                except Exception:
                    pass

    common.GPUMeter = FollowupGPUMeter


# ---------------------------------------------------------------- prompts

LIST_PROFITS_INSTRUCTION = (
    "End by listing the expected profit of each option, then answer with the option "
    "that has the largest."
)
_END_ANCHOR = "End exactly with: FINAL BID: <number>"


def default_labels(n: int) -> list[str]:
    return [chr(65 + i) for i in range(n)]


def build_prompt(common, value: float, candidates: list[dict], labels: list[str] | None, variant: str) -> str:
    """Start from the paper's auction_prompt and apply at most two edits."""
    text = common.auction_prompt(value, candidates)
    if labels is not None and labels != default_labels(len(candidates)):
        lines = text.split("\n")
        index = 0
        for line_no, line in enumerate(lines):
            expected = f"Option {chr(65 + index)}: bid "
            if index < len(candidates) and line.startswith(expected):
                lines[line_no] = f"Option {labels[index]}: bid " + line[len(expected):]
                index += 1
        if index != len(candidates):
            raise RuntimeError("Could not relabel every option line")
        text = "\n".join(lines)
    if variant == "list_profits":
        if _END_ANCHOR not in text:
            raise RuntimeError("Prompt anchor for the Run 4 instruction not found")
        text = text.replace(_END_ANCHOR, LIST_PROFITS_INSTRUCTION + "\n" + _END_ANCHOR)
    elif variant != "paper":
        raise ValueError(variant)
    return text


def random_labels(scenario_id: str, repeat_seed: int, n: int, allowed: list[str], stable_int) -> list[str]:
    """Letters drawn independently of position, fixed per (scenario, order) so
    every model and setting sees the same labels."""
    rng = random.Random(stable_int(scenario_id, repeat_seed, "followup-random-letters"))
    return rng.sample(sorted(allowed), n)


# ---------------------------------------------------------------- readout

def labeled_readout(harness, prompt: str, reasoning_text: str, candidates: list[dict], labels: list[str], chat_mode: str, meter_cls) -> dict[str, Any]:
    """ModelHarness.constrained_bid_readout with the label list as a parameter.
    With labels = A, B, C, ... it computes exactly what the paper's method computes."""
    import torch

    tok = harness.tokenizer
    device = harness.input_device
    prefix = harness._chat_text(prompt, chat_mode)
    prefix_ids = tok(prefix, return_tensors="pt")["input_ids"].to(device)
    if reasoning_text:
        reasoning_ids = tok(reasoning_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    else:
        reasoning_ids = torch.empty((1, 0), dtype=prefix_ids.dtype, device=device)
    suffix_ids = tok(
        "\n\nUse the reasoning above and select the best displayed option. "
        "Return one option letter only. FINAL OPTION:",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)
    input_ids = torch.cat([prefix_ids, reasoning_ids, suffix_ids], dim=1)

    label_token_ids: list[int] = []
    for label in labels:
        variants = [tok.encode(f" {label}", add_special_tokens=False), tok.encode(label, add_special_tokens=False)]
        one_token = next((ids for ids in variants if len(ids) == 1), None)
        if one_token is None:
            raise RuntimeError(f"Option label {label!r} is not a single token: {variants}")
        label_token_ids.append(int(one_token[0]))
    if len(set(label_token_ids)) != len(label_token_ids):
        raise RuntimeError(f"Option labels have duplicate token IDs: {label_token_ids}")

    meter = meter_cls(harness.config["cost"]["power_poll_seconds"])
    start = meter.start()
    with torch.inference_mode():
        output = harness.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    measured = meter.stop(start)
    logits = output.logits[0, -1].float()
    candidate_logits = torch.stack([logits[t] for t in label_token_ids])
    probabilities = torch.softmax(candidate_logits, dim=0)
    selected = int(torch.argmax(candidate_logits).item())
    return {
        "chosen_bid": float(candidates[selected]["bid"]),
        "option_label": labels[selected],
        "option_probability": float(probabilities[selected].item()),
        "option_probabilities": {label: float(probabilities[i].item()) for i, label in enumerate(labels)},
        "option_token_ids": dict(zip(labels, label_token_ids)),
        "context_tokens": int(input_ids.numel()),
        "meter": measured,
    }


def single_token_letters(tokenizer) -> list[str]:
    """Letters whose ' X' (or 'X') form is one token, as the paper's readout requires."""
    good = []
    for code in range(65, 91):
        letter = chr(code)
        variants = [tokenizer.encode(f" {letter}", add_special_tokens=False), tokenizer.encode(letter, add_special_tokens=False)]
        if any(len(v) == 1 for v in variants):
            good.append(letter)
    return good

