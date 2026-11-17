import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Linux Libertine O","Libertinus Serif","DejaVu Serif"],"font.size":7.5,
 "axes.edgecolor":"#52514e","axes.linewidth":0.6,"xtick.color":"#52514e","ytick.color":"#52514e","pdf.fonttype":42})
f=pd.read_csv("six_model_deterministic_frontier.csv")
order=["DeepSeek-8B","Qwen3-14B","Qwen3-4B","Qwen3-8B","Granite-8B","Phi-mini"]
col=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300"]
mk=["o","s","^","D","v","P"]
fig,ax=plt.subplots(figsize=(3.33,1.95))
ax.axhline(0.981,color="#0b0b0b",lw=1,ls="--"); ax.text(10,0.986,"calculator 0.981",fontsize=6.5,color="#0b0b0b")
ax.axhline(0.725,color="#52514e",lw=0.8,ls=":"); ax.text(10,0.700,"median-position rule 0.725",fontsize=6.5,color="#52514e")
for lab,c,m in zip(order,col,mk):
    s=f[f.model_label==lab].sort_values("budget")
    ax.plot(s.budget,s.q,color=c,lw=1.5,marker=m,ms=3.5,mec="#fcfcfb",mew=0.6,label=lab)
ax.set_xticks([0,64,128,256,512,768]); ax.set_xticklabels(["0","64","128","256","512","768"])
ax.set_xlabel("Forced reasoning tokens per decision"); ax.set_ylabel("Held-out utility ratio")
ax.set_ylim(0.68,1.0); ax.set_xlim(-15,785)
ax.grid(axis="y",color="#e6e5e0",lw=0.5); ax.set_axisbelow(True)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
ax.legend(ncol=3,fontsize=6.2,frameon=False,loc="upper left",bbox_to_anchor=(0,0.93),handlelength=1.6,columnspacing=0.8)
fig.tight_layout(pad=0.2); fig.savefig("/tmp/www2027/paper/figures/frontier.pdf"); fig.savefig("/tmp/frontier.png",dpi=220)
