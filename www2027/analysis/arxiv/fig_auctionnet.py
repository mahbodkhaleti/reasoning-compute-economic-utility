"""Figure: AuctionNet held-out utility ratio by forced budget, open vs closed readout at 768, and unforced runs.
Reads results/AN/an_p0_summary.json and an_overnight_summary.json (and the decision files for unforced token use)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Linux Libertine O", "DejaVu Serif"], "font.size": 8.5,
 "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e", "pdf.fonttype": 42})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
order = ["DeepSeek-8B", "Qwen3-14B", "Qwen3-4B", "Qwen3-8B", "Granite-8B", "Phi-mini"]
col = dict(zip(order, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"])); mk = dict(zip(order, ["o", "s", "^", "D", "v", "P"]))
key = {"deepseek_r1_llama_8b": "DeepSeek-8B", "qwen3_14b": "Qwen3-14B", "qwen3_4b": "Qwen3-4B", "qwen3_8b": "Qwen3-8B", "granite_3_2_8b_instruct": "Granite-8B", "phi4_mini_reasoning": "Phi-mini"}
A = json.load(open("/tmp/www2027/results/AN/an_p0_summary.json"))["auctionnet"]; O = json.load(open("/tmp/www2027/results/AN/an_overnight_summary.json"))
B = "/tmp/www2027/results/AN/ov/results/auctionnet/decisions"
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
for a in ax: a.grid(color=GRID, lw=0.5); a.set_axisbelow(True); [a.spines[s].set_visible(False) for s in ("top", "right")]
a = ax[0]; qc = A["qwen3_4b"]["q_calc"]; qr = A["qwen3_4b"]["q_rand"]
a.axhline(qc, color=INK, lw=1, ls="--"); a.text(10, qc + 0.004, f"calculator {qc:.3f}", fontsize=7)
a.axhline(qr, color=INK2, lw=0.8, ls=":"); a.text(1900, qr + 0.004, f"random {qr:.3f}", fontsize=7, color=INK2, ha="right")
for m, l in key.items():
    b = [0, 64, 128, 256, 512, 768]; a.plot(b, [A[m]["q"][str(x)] for x in b], color=col[l], lw=1.4, marker=mk[l], ms=3.2, mec="#fcfcfb", mew=0.5, label=l)
    for rt, ms_ in (("unforced1024", 4), ("unforced4096", 6.5)):
        try: D = [json.loads(x) for x in open(f"{B}/{m}_{rt}.jsonl")]
        except FileNotFoundError: continue
        a.plot(np.mean([d["reasoning_tokens"] for d in D]), np.mean([d["test_utility_ratio"] for d in D]), marker=mk[l], ms=ms_, mfc="none", mec=col[l], mew=1)
a.set_xlabel("Reasoning tokens"); a.set_ylabel("Held-out utility ratio"); a.set_title("(a) AuctionNet: budget and unforced runs", fontsize=8.5)
a.set_xlim(-15, 1950)
a = ax[1]; ms_ = list(key)
x = np.arange(len(ms_)); w = 0.36
op = [100 * O[m]["gap_open"]["pt"] for m in ms_]; cl = [100 * O[m]["gap_closed"]["pt"] for m in ms_]
a.bar(x - w / 2, op, w, color=[col[key[m]] for m in ms_], alpha=0.45, label="open block")
a.bar(x + w / 2, cl, w, color=[col[key[m]] for m in ms_], label="closed block")
for i, m in enumerate(ms_):
    for dx, k in ((-w / 2, "gap_open"), (w / 2, "gap_closed")):
        lo, hi = O[m][k]["lo"], O[m][k]["hi"]; a.plot([i + dx] * 2, [100 * lo, 100 * hi], color=INK, lw=0.7)
a.set_xticks(x); a.set_xticklabels([key[m] for m in ms_], rotation=30, ha="right", fontsize=7.5)
a.set_ylabel("Gap recovered at 768 tokens (%)"); a.set_ylim(0, 118); a.set_yticks([0, 25, 50, 75, 100]); a.set_title("(b) AuctionNet: open vs closed readout", fontsize=8.5)
from matplotlib.patches import Patch
a.legend(handles=[Patch(color="#888", alpha=0.45, label="open block (paper readout)"), Patch(color="#888", label="closed block")], frameon=False, fontsize=7, loc="upper left", ncol=2)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig("/tmp/www2027/results/arxiv/figures/fig_auctionnet.pdf", bbox_inches="tight")
