from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dirs(output_dir: str | Path) -> dict[str, Path]:
    base = Path(output_dir)
    paths = {
        "base": base,
        "data": base / "data",
        "raw": base / "raw",
        "tables": base / "tables",
        "figures": base / "figures",
        "direct": base / "direct_games",
        "logs": base / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def stable_int(*parts: Any, modulo: int = 2**31 - 1) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.append(json.loads(line))
    return result


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def find_model(config: dict[str, Any], slug: str) -> dict[str, Any]:
    for spec in config["models"]:
        if spec["slug"] == slug:
            return spec
    raise KeyError(f"Unknown model slug {slug!r}")


def parse_budget_list(text: str | None, defaults: list[int]) -> list[int]:
    if not text:
        return list(defaults)
    values: list[int] = []
    for part in text.split(","):
        part = part.strip().lower()
        values.append(-1 if part == "unconstrained" else int(part))
    return values


def parse_candidate_json(text: str) -> list[dict[str, float]]:
    raw = json.loads(text)
    return [
        {
            "bid": float(x["bid"]),
            "train_p": float(x["train_p"]),
            "train_u": float(x["train_u"]),
            "test_p": float(x["test_p"]),
            "test_u": float(x["test_u"]),
        }
        for x in raw
    ]


def shuffled_candidates(scenario: dict[str, Any], repeat_seed: int, agent: str = "") -> list[dict[str, float]]:
    candidates = parse_candidate_json(scenario["candidates_json"])
    rng = random.Random(stable_int(scenario["scenario_id"], repeat_seed, agent, "option-order"))
    rng.shuffle(candidates)
    return candidates


def auction_prompt(value: float, candidates: list[dict[str, float]], strategic_history: str = "") -> str:
    options = "\n".join(
        f"Option {chr(65 + i)}: bid ${c['bid']:.2f}; estimated win probability {c['train_p']:.4f}"
        for i, c in enumerate(candidates)
    )
    history = f"\nRecent public market history:\n{strategic_history}\n" if strategic_history else ""
    return f"""You are a bidder in a first-price auction.
Your private value is ${value:.2f}.
{history}
Available bids are shown in randomized order:
{options}

For any bid b, expected profit under the supplied estimate is
(private value - b) * estimated win probability.
Choose one listed bid. Do not infer the answer from option position.
End exactly with: FINAL BID: <number>
"""


def budget_choice_prompt(
    value: float,
    allowed: list[int],
    frontier_text: str,
    opponent_history: str,
    gpu_hour_usd: float,
    estimated_seconds_per_token: float,
) -> str:
    cost_per_token = gpu_hour_usd * estimated_seconds_per_token / 3600.0
    return f"""You must choose an inference reasoning budget before seeing the bid options.
Your private auction value is ${value:.2f}.
Allowed reasoning-token budgets: {allowed}.
Measured validation performance for this model:
{frontier_text}
Recent opponent behavior:
{opponent_history or 'No previous rounds.'}
Measured rental-cost estimate is ${cost_per_token:.10f} per generated token
at ${gpu_hour_usd:.4f} per GPU-hour. More tokens may improve the later bid but cost money.
Choose strategically to maximize expected private auction profit net of compute cost.
    Return only: COMPUTE BUDGET: <one allowed integer>
"""


def verbal_confidence_prompt(auction_text: str, selected_bid: float | None) -> str:
    bid_text = "invalid/no bid" if selected_bid is None else f"${selected_bid:.2f}"
    return f"""Review the auction decision below.

{auction_text}

The selected answer was {bid_text}.
Estimate the probability that this selected bid maximizes expected utility among the listed options.
Do not change the bid and do not use knowledge unavailable in the auction prompt.
Return only: CONFIDENCE: <integer from 0 to 100>
"""


FINAL_BID_PATTERNS = [
    r"FINAL\s+BID\s*:\s*\$?\s*\\boxed\{\s*([0-9]+(?:\.[0-9]+)?)\s*\}",
    r"FINAL\s+BID\s*:\s*\$?\s*([0-9]+(?:\.[0-9]+)?)",
]


def parse_final_bid(text: str, candidates: list[dict[str, float]], tolerance: float = 1e-4) -> float | None:
    value: float | None = None
    for pattern in FINAL_BID_PATTERNS:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            value = float(matches[-1].group(1))
            break
    if value is None:
        return None
    valid = [c["bid"] for c in candidates if abs(c["bid"] - value) <= tolerance]
    return float(valid[0]) if valid else None


def parse_compute_budget(text: str, allowed: list[int]) -> int | None:
    matches = list(re.finditer(r"COMPUTE\s+BUDGET\s*:\s*([0-9]+)", text, flags=re.IGNORECASE))
    if not matches:
        return None
    value = int(matches[-1].group(1))
    return value if value in allowed else None


def parse_verbal_confidence(text: str) -> float | None:
    matches = list(re.finditer(r"CONFIDENCE\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE))
    if not matches:
        return None
    value = float(matches[-1].group(1))
    return value / 100.0 if 0.0 <= value <= 100.0 else None


def score_bid(chosen: float | None, candidates: list[dict[str, float]]) -> dict[str, Any]:
    train_best = max(candidates, key=lambda x: (x["train_u"], -x["bid"]))
    test_best = max(candidates, key=lambda x: (x["test_u"], -x["bid"]))
    selected = next((x for x in candidates if chosen is not None and abs(x["bid"] - chosen) < 1e-4), None)
    train_u = 0.0 if selected is None else selected["train_u"]
    test_u = 0.0 if selected is None else selected["test_u"]
    return {
        "valid_output": selected is not None,
        "chosen_bid": chosen,
        "train_utility": train_u,
        "test_utility": test_u,
        "train_optimal_bid": train_best["bid"],
        "test_optimal_bid": test_best["bid"],
        "train_optimal_utility": train_best["train_u"],
        "test_optimal_utility": test_best["test_u"],
        "train_regret": train_best["train_u"] - train_u,
        "test_regret": test_best["test_u"] - test_u,
        "train_utility_ratio": train_u / train_best["train_u"] if train_best["train_u"] > 0 else math.nan,
        "test_utility_ratio": test_u / test_best["test_u"] if test_best["test_u"] > 0 else math.nan,
        "train_correct": selected is not None and abs(selected["bid"] - train_best["bid"]) < 1e-4,
        "test_correct": selected is not None and abs(selected["bid"] - test_best["bid"]) < 1e-4,
    }


@dataclass
class MeterResult:
    seconds: float
    energy_joules: float
    average_power_watts: float
    peak_memory_bytes: int
    nvml_status: str
    nvml_error: str | None


class GPUMeter:
    def __init__(self, poll_seconds: float = 0.1):
        self.poll_seconds = poll_seconds
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None
        self.nvml_status = "unavailable"
        self.nvml_error: str | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml_status = "available"
        except Exception as exc:
            self._nvml = None
            self.nvml_error = f"{type(exc).__name__}: {exc}"

    def _poll(self) -> None:
        assert self._nvml is not None and self._handle is not None
        while not self._stop.is_set():
            try:
                watts = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                self.samples.append((time.perf_counter(), watts))
            except Exception:
                pass
            self._stop.wait(self.poll_seconds)

    def start(self) -> float:
        import torch

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        if self._nvml is not None:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return start

    def stop(self, start: float) -> MeterResult:
        import torch

        torch.cuda.synchronize()
        end = time.perf_counter()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        energy = math.nan
        avg_power = math.nan
        if self.samples:
            if len(self.samples) == 1:
                energy = self.samples[0][1] * (end - start)
            else:
                energy = sum(
                    (t2 - t1) * (p1 + p2) / 2.0
                    for (t1, p1), (t2, p2) in zip(self.samples[:-1], self.samples[1:])
                )
            avg_power = energy / max(end - start, 1e-9)
        return MeterResult(
            seconds=end - start,
            energy_joules=energy,
            average_power_watts=avg_power,
            peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
            nvml_status=self.nvml_status,
            nvml_error=self.nvml_error,
        )


def monetary_cost(meter: MeterResult, gpu_hour_usd: float | None, electricity_usd_per_kwh: float | None, gpu_count: int = 1) -> dict[str, float]:
    rental = math.nan if gpu_hour_usd is None else meter.seconds * gpu_hour_usd * gpu_count / 3600.0
    energy = (
        math.nan
        if electricity_usd_per_kwh is None or math.isnan(meter.energy_joules)
        else meter.energy_joules / 3_600_000.0 * electricity_usd_per_kwh
    )
    total = rental
    if math.isnan(total) and not math.isnan(energy):
        total = energy
    elif not math.isnan(total) and not math.isnan(energy):
        total += energy
    return {"rental_cost_usd": rental, "energy_cost_usd": energy, "total_compute_cost_usd": total}


class ModelHarness:
    def __init__(self, spec: dict[str, Any], config: dict[str, Any]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required")
        cache = os.environ.get("ROBUSTNESS_HF_CACHE", config.get("hf_cache"))
        self.spec = spec
        self.config = config
        model_revision = spec.get("revision")
        tokenizer_revision = spec.get("tokenizer_revision", model_revision)
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec["hf_id"],
            revision=tokenizer_revision,
            cache_dir=cache,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                spec["hf_id"],
                revision=model_revision,
                cache_dir=cache,
                dtype=torch.bfloat16,
                device_map=spec.get("device_map", "auto"),
                trust_remote_code=bool(spec.get("trust_remote_code", True)),
                low_cpu_mem_usage=True,
            )
        except torch.cuda.OutOfMemoryError as exc:
            visible = [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
            raise RuntimeError(
                f"BF16 model load ran out of GPU memory; visible bytes={visible}. "
                "The package will not silently quantize. Use a larger GPU or explicit multi-GPU device_map=auto."
            ) from exc
        self.model.eval()
        self.input_device = next(self.model.parameters()).device
        # Transformers may omit its private _commit_hash metadata. Both
        # artifacts were loaded using immutable commit SHAs above.
        self.model_revision = getattr(self.model.config, "_commit_hash", None) or model_revision
        self.tokenizer_revision = self.tokenizer.init_kwargs.get("_commit_hash") or tokenizer_revision

    def _chat_text(self, prompt: str, chat_mode: str) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        template_argument = self.spec.get("chat_template_argument")
        if template_argument:
            kwargs[template_argument] = chat_mode == "reasoning"
        try:
            return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)
        except TypeError as exc:
            if template_argument and self.spec.get("template_argument_required", False):
                raise RuntimeError(
                    f"Pinned tokenizer rejected required chat-template argument {template_argument!r}"
                ) from exc
            kwargs.pop(template_argument, None)
            return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)

    def _bad_eos(self) -> list[list[int]]:
        eos = self.tokenizer.eos_token_id
        if eos is None:
            return []
        ids = eos if isinstance(eos, list) else [eos]
        return [[int(x)] for x in ids]

    def generate_text(self, prompt: str, max_tokens: int, seed: int, chat_mode: str, sample: bool = True) -> tuple[str, int, MeterResult]:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        text = self._chat_text(prompt, chat_mode)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.input_device)
        meter = GPUMeter(self.config["cost"]["power_poll_seconds"])
        start = meter.start()
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if sample:
            generation_kwargs.update(
                temperature=self.config["evaluation"]["temperature"],
                top_p=self.config["evaluation"]["top_p"],
            )
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        measured = meter.stop(start)
        new_ids = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True), int(new_ids.numel()), measured

    def confidence_query(self, prompt: str, seed: int) -> dict[str, Any]:
        """Elicit a short confidence value without allowing a reasoning preamble.

        Some reasoning checkpoints (notably Phi-4-mini-reasoning) ignore or do
        not implement the chat template's ``enable_thinking=False`` argument.
        A normal generation can therefore exhaust its token cap inside a
        reasoning trace.  Prefilling the required response label makes this a
        constrained readout, analogous to the zero-reasoning auction answer
        stage, and keeps its cost separate from the economic decision.
        """
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        prefix = self._chat_text(prompt, "no_think")
        inputs = self.tokenizer(prefix, return_tensors="pt").to(self.input_device)
        suffix = self.tokenizer(
            "\n\nCONFIDENCE:",
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.input_device)
        answer_input = torch.cat([inputs["input_ids"], suffix], dim=1)
        meter = GPUMeter(self.config["cost"]["power_poll_seconds"])
        start = meter.start()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=answer_input,
                attention_mask=torch.ones_like(answer_input),
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        measured = meter.stop(start)
        new_ids = output[0, answer_input.shape[1] :]
        text = "CONFIDENCE:" + self.tokenizer.decode(new_ids, skip_special_tokens=True)
        tokens = int(new_ids.numel())
        return {
            "verbal_confidence_text": text,
            "verbal_confidence": parse_verbal_confidence(text),
            "verbal_confidence_tokens": tokens,
            "verbal_confidence_meter": measured,
        }

    def constrained_bid_readout(
        self,
        prompt: str,
        reasoning_text: str,
        candidates: list[dict[str, float]],
        chat_mode: str,
    ) -> dict[str, Any]:
        """Select a displayed option using constrained next-token letter logits."""
        import torch

        prefix = self._chat_text(prompt, chat_mode)
        prefix_ids = self.tokenizer(prefix, return_tensors="pt")["input_ids"].to(self.input_device)
        if reasoning_text:
            reasoning_ids = self.tokenizer(
                reasoning_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(self.input_device)
        else:
            reasoning_ids = torch.empty(
                (1, 0), dtype=prefix_ids.dtype, device=self.input_device
            )
        suffix_ids = self.tokenizer(
            "\n\nUse the reasoning above and select the best displayed option. "
            "Return one option letter only. FINAL OPTION:",
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.input_device)
        input_ids = torch.cat([prefix_ids, reasoning_ids, suffix_ids], dim=1)

        labels = [chr(65 + index) for index in range(len(candidates))]
        label_token_ids: list[int] = []
        for label in labels:
            variants = [
                self.tokenizer.encode(f" {label}", add_special_tokens=False),
                self.tokenizer.encode(label, add_special_tokens=False),
            ]
            one_token = next((ids for ids in variants if len(ids) == 1), None)
            if one_token is None:
                raise RuntimeError(
                    f"Option label {label!r} is not a single token: {variants}"
                )
            label_token_ids.append(int(one_token[0]))
        if len(set(label_token_ids)) != len(label_token_ids):
            raise RuntimeError(f"Option labels have duplicate token IDs: {label_token_ids}")

        meter = GPUMeter(self.config["cost"]["power_poll_seconds"])
        start = meter.start()
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
        measured = meter.stop(start)
        logits = output.logits[0, -1].float()
        candidate_logits = torch.stack([logits[token_id] for token_id in label_token_ids])
        probabilities = torch.softmax(candidate_logits, dim=0)
        selected_index = int(torch.argmax(candidate_logits).item())
        return {
            "chosen_bid": float(candidates[selected_index]["bid"]),
            "option_label": labels[selected_index],
            "option_probability": float(probabilities[selected_index].item()),
            "option_probabilities": {
                label: float(probabilities[index].item())
                for index, label in enumerate(labels)
            },
            "option_token_ids": dict(zip(labels, label_token_ids)),
            "context_tokens": int(input_ids.numel()),
            "meter": measured,
        }

    def auction_decision(
        self,
        prompt: str,
        budget: int,
        seed: int,
        chat_mode: str,
        sample: bool = False,
    ) -> dict[str, Any]:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        prefix = self._chat_text(prompt, chat_mode)
        inputs = self.tokenizer(prefix, return_tensors="pt").to(self.input_device)
        eval_cfg = self.config["evaluation"]
        meter = GPUMeter(self.config["cost"]["power_poll_seconds"])
        start = meter.start()
        if budget < 0:
            unconstrained_kwargs = {
                "max_new_tokens": eval_cfg["unconstrained_max_tokens"],
                "do_sample": sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "use_cache": True,
            }
            if sample:
                unconstrained_kwargs.update(
                    temperature=eval_cfg["temperature"],
                    top_p=eval_cfg["top_p"],
                )
            with torch.inference_mode():
                output = self.model.generate(**inputs, **unconstrained_kwargs)
            new_ids = output[0, inputs["input_ids"].shape[1] :]
            decoded = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            measured = meter.stop(start)
            return {"reasoning_text": decoded, "final_text": decoded, "reasoning_tokens": int(new_ids.numel()), "answer_tokens": 0, "meter": measured}

        if budget == 0:
            reason_output = inputs["input_ids"]
            reasoning_ids = reason_output[0, 0:0]
        else:
            reasoning_kwargs = {
                "min_new_tokens": budget,
                "max_new_tokens": budget,
                "do_sample": sample,
                "bad_words_ids": self._bad_eos() or None,
                "pad_token_id": self.tokenizer.pad_token_id,
                "use_cache": True,
            }
            if sample:
                reasoning_kwargs.update(
                    temperature=eval_cfg["temperature"],
                    top_p=eval_cfg["top_p"],
                )
            with torch.inference_mode():
                reason_output = self.model.generate(**inputs, **reasoning_kwargs)
            reasoning_ids = reason_output[0, inputs["input_ids"].shape[1] :]
        suffix = self.tokenizer("\n\nNow give the required answer. FINAL BID:", add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.input_device)
        answer_input = torch.cat([reason_output[0].unsqueeze(0), suffix], dim=1)
        with torch.inference_mode():
            answer_output = self.model.generate(
                input_ids=answer_input,
                attention_mask=torch.ones_like(answer_input),
                max_new_tokens=eval_cfg["answer_tokens"],
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        answer_sequence = answer_output.sequences
        answer_ids = answer_sequence[0, answer_input.shape[1] :]
        transition_scores = self.model.compute_transition_scores(
            answer_sequence,
            answer_output.scores,
            normalize_logits=True,
        )[0]
        finite_scores = transition_scores[torch.isfinite(transition_scores)]
        if finite_scores.numel():
            answer_mean_logprob = float(finite_scores.mean().item())
            answer_min_logprob = float(finite_scores.min().item())
            answer_geomean_probability = float(torch.exp(finite_scores.mean()).item())
        else:
            answer_mean_logprob = math.nan
            answer_min_logprob = math.nan
            answer_geomean_probability = math.nan
        measured = meter.stop(start)
        reasoning_text = self.tokenizer.decode(reasoning_ids, skip_special_tokens=True)
        answer_text = "FINAL BID:" + self.tokenizer.decode(answer_ids, skip_special_tokens=True)
        return {
            "reasoning_text": reasoning_text,
            "final_text": answer_text,
            "reasoning_tokens": int(reasoning_ids.numel()),
            "answer_tokens": int(answer_ids.numel()),
            "answer_mean_logprob": answer_mean_logprob,
            "answer_min_logprob": answer_min_logprob,
            "answer_geomean_probability": answer_geomean_probability,
            "meter": measured,
        }


def meter_fields(meter: MeterResult, config: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "elapsed_seconds": meter.seconds,
        "energy_joules": meter.energy_joules,
        "average_power_watts": meter.average_power_watts,
        "peak_memory_bytes": meter.peak_memory_bytes,
        "nvml_status": meter.nvml_status,
        "nvml_error": meter.nvml_error,
    }
    fields.update(monetary_cost(meter, config["cost"].get("gpu_hour_usd"), config["cost"].get("electricity_usd_per_kwh"), int(config["cost"].get("gpu_count_billed", 1))))
    return fields
