import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Linux Libertine O","DejaVu Serif"],"font.size":7.5,
 "axes.edgecolor":"#52514e","axes.linewidth":0.6,"xtick.color":"#52514e","ytick.color":"#52514e","pdf.fonttype":42})
INK,INK2,GRID="#0b0b0b","#52514e","#e6e5e0"
order=["DeepSeek-8B","Qwen3-14B","Qwen3-4B","Qwen3-8B","Granite-8B","Phi-mini"]
col=dict(zip(order,["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300"]))
mk=dict(zip(order,["o","s","^","D","v","P"]))
key={"deepseek_r1_llama_8b":"DeepSeek-8B","qwen3_14b":"Qwen3-14B","qwen3_4b":"Qwen3-4B","qwen3_8b":"Qwen3-8B","granite_3_2_8b_instruct":"Granite-8B","phi4_mini_reasoning":"Phi-mini"}
fig,ax=plt.subplots(1,3,figsize=(7.0,2.05),gridspec_kw=dict(width_ratios=[1.05,1,1]))
def style(a):
    a.grid(color=GRID,lw=0.5); a.set_axisbelow(True)
    for sp in ["top","right"]: a.spines[sp].set_visible(False)
# (a) frontier
f=pd.read_csv("six_model_deterministic_frontier.csv"); a=ax[0]
a.axhline(0.981,color=INK,lw=1,ls="--"); a.text(10,0.986,"calculator 0.981",fontsize=6.3,color=INK)
a.axhline(0.711,color=INK2,lw=0.8,ls=":"); a.text(10,0.690,"random choice 0.711",fontsize=6.3,color=INK2)
for l in order:
    s=f[f.model_label==l].sort_values("budget")
    a.plot(s.budget,s.q,color=col[l],lw=1.4,marker=mk[l],ms=3.2,mec="#fcfcfb",mew=0.5,label=l)
a.set_xticks([0,128,256,512,768]); a.set_ylim(0.62,1.01); a.set_xlim(-15,785)
a.set_xlabel("Forced reasoning tokens"); a.set_ylabel("Held-out utility ratio"); style(a); a.grid(axis="x",visible=False)
a.set_title("(a) eBay price data",fontsize=7.5,loc="left")
# (b) Kaplan-Meier share of traces that have evaluated every option by token t (natural run, censored at stop or cap)
import json as _j
tf=pd.read_csv("trace_first_v2.csv"); sc_=pd.read_csv("scenarios.csv"); Kmap=dict(zip(sc_.scenario_id,sc_.option_count))
tf=tf[tf.budget==-1].copy(); tf["K"]=tf.scenario_id.map(Kmap); tf["fp"]=tf.first_pos.map(_j.loads)
tf["complete"]=tf.fp.map(len)==tf.K
tf["ct"]=[max(f.values())/max(n,1)*t if c else np.nan for f,n,t,c in zip(tf.fp,tf.n_chars,tf.reasoning_tokens,tf.complete)]
def km(time,event,grid):
    o=np.argsort(time); time=np.asarray(time,float)[o]; event=np.asarray(event)[o]; n=len(time); s=1.0; pts=[]; ar=n; i=0
    while i<n:
        tt=time[i]; dd=0; cc=0
        while i<n and time[i]==tt: dd+=event[i]; cc+=1-event[i]; i+=1
        if ar>0 and dd>0: s*=(1-dd/ar)
        pts.append((tt,1-s)); ar-=dd+cc
    pts=np.array(pts); return np.array([pts[pts[:,0]<=gg,1].max() if (pts[:,0]<=gg).any() else 0 for gg in grid])
a=ax[1]; grid=np.arange(0,1025,8)
for m,l in key.items():
    x=tf[tf.model==m]; time=np.where(x.complete,x.ct,x.reasoning_tokens)
    a.plot(grid,km(time,x.complete.astype(int).values,grid),color=col[l],lw=1.4)
a.axvline(768,color=INK,lw=0.8,ls="--"); a.text(760,0.03,"forced\nbudget",fontsize=6,ha="right",color=INK)
a.axvline(1024,color=INK2,lw=0.8,ls=":")
a.set_xticks([0,256,512,768,1024]); a.set_xlim(0,1040); a.set_ylim(0,1.02)
a.set_xlabel("Reasoning tokens"); a.set_ylabel("Share of traces finished"); style(a)
a.set_title("(b) Tokens needed to finish",fontsize=7.5,loc="left")
d=pd.read_csv("saved_records.csv.gz")
# (c) routing curves
r=pd.read_csv("out/routing_curves_v3.csv"); a=ax[2]
rc={"value":("#2a78d6","Value, \\$ gain"),"confidence":("#eb6834","Confidence, \\$ gain"),"random":("#52514e","Random")}
for k,(cc,lab) in rc.items():
    z=(r[r.rule==k].pivot(index="token_share",columns="model",values="gain_kept") if k!="random" else pd.DataFrame({"x":np.linspace(0,1,101)},index=np.linspace(0,1,101)))
    if k!="random": a.fill_between(z.index,z.min(axis=1),z.max(axis=1),color=cc,alpha=0.15,lw=0)
    a.plot(z.index,z.mean(axis=1),color=cc,lw=1.5 if k!="random" else 0.9,ls="-" if k!="random" else ":",label=lab)
# value ranking scored on the utility-ratio gain (dashed)
gq=d[(d.decoding=="greedy")&d.budget.isin([0,768])].pivot_table(index=["model","scenario_id","repeat_seed"],columns="budget",values="test_utility_ratio").reset_index()
gq["v"]=gq.scenario_id.map(dict(zip(pd.read_csv("scenarios.csv").scenario_id,pd.read_csv("scenarios.csv").value)))
cur=[]
for m in key:
    x=gq[gq.model==m]; jit=np.random.default_rng(1).uniform(0,1e-7,len(x))
    o=np.argsort(-(x.v.values+jit),kind="stable"); dg=(x[768]-x[0]).values[o]; cum=np.r_[0,np.cumsum(dg)]
    cur.append([cum[int(round(q*len(dg)))]/cum[-1] for q in np.linspace(0,1,101)])
cur=np.array(cur); a.plot(np.linspace(0,1,101),cur.mean(0),color="#2a78d6",lw=1.3,ls="--",label="Value, ratio gain")
a.set_xlim(0,1); a.set_ylim(-0.42,1.08); a.set_yticks([0,0.25,0.5,0.75,1.0])
a.set_xlabel("Share of decisions given 768 tokens",fontsize=7); a.set_ylabel("Held-out gain kept"); style(a)
a.legend(fontsize=5.6,frameon=False,loc="lower right",ncol=2,handlelength=1.6,columnspacing=0.6,borderaxespad=0.1)
a.set_title("(c) Routing compute",fontsize=7.5,loc="left")
h,l=ax[0].get_legend_handles_labels()
fig.legend(h,l,ncol=6,fontsize=6.5,frameon=False,loc="upper center",bbox_to_anchor=(0.5,1.0),handlelength=1.6,columnspacing=1.2)
fig.tight_layout(pad=0.25,rect=(0,0,1,0.92),w_pad=1.0)
fig.savefig("/tmp/www2027/paper/figures/main_figure.pdf"); fig.savefig("/tmp/main.png",dpi=220)
