"""R1: readout-template ablation on the saved eBay 768-token traces (prefill only, no new generation).
Original readout: chat prefix + re-tokenized trace + "\n\nUse the reasoning above and select the best displayed option.
Return one option letter only. FINAL OPTION:" (no end-of-thinking tag; the eBay code does not close the block).
Alternative: close the reasoning block with the model's own marker, then a neutral line, then read the letter:
  <think> models (Qwen3, DeepSeek-R1-Distill, Phi-4-mini-reasoning): "\n</think>\n\nFinal answer: Option"
  Granite-3.2 (thinking format "Here is my thought process: ... Here is my response:"): "\n\nHere is my response:\nFinal answer: Option"
If the trace already contains the marker, only "\n\nFinal answer: Option" is appended. The label token rule (first
single-token variant of " X" / "X") is the eBay rule. Both readouts are recomputed so the original can be checked
against the saved letter."""
from __future__ import annotations
import time

def alt_suffix(model, text):
    if model.startswith("granite"):
        return "\n\nFinal answer: Option" if "Here is my response:" in text else "\n\nHere is my response:\nFinal answer: Option"
    return "\n\nFinal answer: Option" if "</think>" in text else "\n</think>\n\nFinal answer: Option"

def readout_with_suffix(h, prompt, text, cands, suffix):
    import torch
    tok = h.tokenizer; dev = h.input_device
    ids = tok(h._chat_text(prompt, "reasoning"))["input_ids"] + (tok(text, add_special_tokens=False)["input_ids"] if text else []) + tok(suffix, add_special_tokens=False)["input_ids"]
    labels = [chr(65 + i) for i in range(len(cands))]
    lt = []
    for L in labels:
        v = [tok.encode(f" {L}", add_special_tokens=False), tok.encode(L, add_special_tokens=False)]
        one = next((x for x in v if len(x) == 1), None)
        if one is None: raise RuntimeError(f"label {L} not single token")
        lt.append(int(one[0]))
    x = torch.tensor([ids], device=dev)
    with torch.inference_mode():
        lg = h.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0, -1].float()
    cl = torch.stack([lg[t] for t in lt]); p = torch.softmax(cl, 0); j = int(torch.argmax(cl))
    return dict(label=labels[j], bid=float(cands[j]["bid"]), probs={L: float(p[i]) for i, L in enumerate(labels)})
