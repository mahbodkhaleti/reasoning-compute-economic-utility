import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Linux Libertine O","DejaVu Serif"],"font.size":7.5,
 "axes.edgecolor":"#52514e","axes.linewidth":0.6,"xtick.color":"#52514e","ytick.color":"#52514e","pdf.fonttype":42})
d=pd.read_csv("saved_records.csv.gz"); g=d[(d.decoding=="greedy")&d.budget.isin([0,768])]
c=g.groupby(["model","scenario_id","budget"]).test_utility.mean().unstack()
gain=(c[768]-c[0]).reset_index(name="gain")
lab={"deepseek_r1_llama_8b":"DeepSeek-8B","qwen3_14b":"Qwen3-14B","qwen3_4b":"Qwen3-4B","qwen3_8b":"Qwen3-8B","granite_3_2_8b_instruct":"Granite-8B","phi4_mini_reasoning":"Phi-mini"}
col=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300"]
fig,ax=plt.subplots(figsize=(3.33,1.95))
for (m,l),cc in zip(lab.items(),col):
    x=np.sort(gain[gain.model==m].gain.values); y=np.arange(1,len(x)+1)/len(x)
    ax.step(x,y,where="post",color=cc,lw=1.3,label=l)
ax.set_xscale("symlog",linthresh=0.01,linscale=0.6)
ax.axvline(0.0034,color="#0b0b0b",lw=0.9,ls="--")
ax.text(0.0045,0.50,"cost of 768 tokens\n(max \\$0.0034)",fontsize=6.2,color="#0b0b0b")
ax.axvline(0,color="#52514e",lw=0.6,ls=":")
ax.set_xlim(-200,400); ax.set_ylim(0,1)
ax.set_xticks([-100,-1,-0.01,0,0.01,1,100]); ax.set_xticklabels(["$-$100","$-$1","","0","0.01","1","100"])
ax.set_xlabel("Expected profit gain per scenario, 0 to 768 tokens (\\$)"); ax.set_ylabel("Share of scenarios")
ax.grid(color="#e6e5e0",lw=0.5); ax.set_axisbelow(True)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
ax.legend(ncol=3,fontsize=6,frameon=False,loc="lower left",bbox_to_anchor=(0,1.0),handlelength=1.4,columnspacing=0.8,borderaxespad=0.1)
fig.tight_layout(pad=0.2); fig.savefig("/tmp/www2027/paper/figures/gain_ecdf.pdf"); fig.savefig("/tmp/gain.png",dpi=220)
print(gain.groupby("model").gain.apply(lambda v:((v>-0.0001)&(v<0.0001)).mean()))
