import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Linux Libertine O","DejaVu Serif"],"font.size":8.5,
 "axes.edgecolor":"#52514e","axes.linewidth":0.6,"xtick.color":"#52514e","ytick.color":"#52514e","pdf.fonttype":42})
INK,INK2,GRID="#0b0b0b","#52514e","#e6e5e0"
order=["DeepSeek-8B","Qwen3-14B","Qwen3-4B","Qwen3-8B","Granite-8B","Phi-mini"]
col=dict(zip(order,["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300"]))
mk=dict(zip(order,["o","s","^","D","v","P"]))
key={"deepseek_r1_llama_8b":"DeepSeek-8B","qwen3_14b":"Qwen3-14B","qwen3_4b":"Qwen3-4B","qwen3_8b":"Qwen3-8B","granite_3_2_8b_instruct":"Granite-8B","phi4_mini_reasoning":"Phi-mini"}
fig,ax=plt.subplots(1,3,figsize=(7.0,2.45),gridspec_kw=dict(width_ratios=[1.05,1,1]))
def style(a):
    a.grid(color=GRID,lw=0.5); a.set_axisbelow(True)
    for sp in ["top","right"]: a.spines[sp].set_visible(False)
# (a) frontier
f=pd.read_csv("../six_model_deterministic_frontier.csv"); a=ax[0]
a.axhline(0.981,color=INK,lw=1,ls="--"); a.text(10,0.986,"calculator 0.981",fontsize=7,color=INK)
a.axhline(0.711,color=INK2,lw=0.8,ls=":"); a.text(10,0.690,"random choice 0.711",fontsize=7,color=INK2)
for l in order:
    s=f[f.model_label==l].sort_values("budget")
    a.plot(s.budget,s.q,color=col[l],lw=1.4,marker=mk[l],ms=3.2,mec="#fcfcfb",mew=0.5,label=l)
_nat=pd.read_csv("../trace_first_v2.csv"); _nat=_nat[_nat.budget==-1]
_rec=pd.read_csv("../saved_records.csv.gz"); _rec=_rec[(_rec.decoding=="greedy")&(_rec.budget==-1)]
for m_,l_ in key.items():
    xx=_nat[_nat.model==m_].reasoning_tokens.mean(); yy=_rec[_rec.model==m_].test_utility_ratio.mean()
    a.plot([xx],[yy],marker=mk[l_],ms=4,mfc="none",mec=col[l_],mew=1.2,ls="none")
import json as _jj
_R1=_jj.load(open("/tmp/www2027/results/followup/run1_summary.json"))
for m_,r_ in _R1.items():
    l_=key[m_]; x0=_nat[_nat.model==m_].reasoning_tokens.mean(); y0=_rec[_rec.model==m_].test_utility_ratio.mean()
    a.plot([x0,r_["tokens_mean"]],[y0,r_["q_natural"]["pt"]],color=col[l_],lw=0.6,ls=":")
    a.plot([r_["tokens_mean"]],[r_["q_natural"]["pt"]],marker=mk[l_],ms=6,mfc="none",mec=col[l_],mew=1.2,ls="none")
a.text(1640,0.64,"hollow: no forcing, cap 1,024\n(small) or 4,096 (large)",fontsize=6.5,ha="right",color=INK2)
a.set_xticks([0,512,1024,1536]); a.set_ylim(0.62,1.01); a.set_xlim(-15,1660)
a.set_xlabel("Reasoning tokens"); a.set_ylabel("Held-out utility ratio"); style(a); a.grid(axis="x",visible=False)
a.set_title("(a) eBay price data",fontsize=8.5,loc="left")
# (b) cumulative incidence: share of ALL unconstrained traces that have evaluated every option by token t
import json as _j
A3=_j.load(open("out/t9_judge6.json"))["A3"]
a=ax[1]; grid=np.arange(0,1025,8)
for m,l in key.items(): a.plot(grid,A3[m]["all"]["curve"],color=col[l],lw=1.4)
a.axvline(768,color=INK,lw=0.8,ls="--"); a.text(760,0.03,"forced\nbudget",fontsize=7,ha="right",color=INK)
a.axvline(1024,color=INK2,lw=0.8,ls=":")
a.set_xticks([0,256,512,768,1024]); a.set_xlim(0,1040); a.set_ylim(0,1.02)
a.set_xlabel("Reasoning tokens"); a.set_ylabel("Share of traces finished"); style(a)
a.set_title("(b) Tokens needed to finish",fontsize=8.5,loc="left")
d=pd.read_csv("../saved_records.csv.gz")
# (c) routing curves
import json as _js
V=_js.load(open("out/t10_fig_series.json")); a=ax[2]
def series(k):
    ys=[]
    for m_ in key:
        pts=V[m_]; xs=[p["budget"] for p in pts]; ys.append([p[k] for p in pts])
    return np.array(xs), np.array(ys)
a.axhline(0.981,color=INK,lw=1,ls="--"); a.text(10,0.986,"calculator",fontsize=7,color=INK)
for k_,cc,lab_,ls_,band in [("q_read","#52514e","Model's answer","-",True),("q_reader","#2a78d6","Text reader","-",True),("q_oracle","#1baf7a","Oracle reader",":",False)]:
    xs,ys=series(k_)
    if band: a.fill_between(xs,ys.min(0),ys.max(0),color=cc,alpha=0.12,lw=0)
    a.plot(xs,ys.mean(0),color=cc,lw=1.6 if band else 1.2,ls=ls_,label=lab_)
a.set_xticks([0,256,512,768]); a.set_xlim(-15,785); a.set_ylim(0.62,1.01)
a.set_xlabel("Forced reasoning tokens"); a.set_ylabel("Held-out utility ratio"); style(a)
a.legend(fontsize=6.8,frameon=False,loc="upper left",bbox_to_anchor=(0,0.93),handlelength=1.6)
a.set_title("(c) Reading out the trace",fontsize=8.5,loc="left")
h,l=ax[0].get_legend_handles_labels()
fig.legend(h,l,ncol=6,fontsize=7.5,frameon=False,loc="upper center",bbox_to_anchor=(0.5,1.0),handlelength=1.6,columnspacing=1.2)
fig.tight_layout(pad=0.25,rect=(0,0,1,0.92),w_pad=1.0)
fig.savefig("/tmp/www2027/results/followup/fig_main_run1.pdf"); fig.savefig("/tmp/www2027/results/followup/fig_main_run1.png",dpi=220)
