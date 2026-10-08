"""Exploratory, CPU-only reanalysis of frozen auction inference records."""
from pathlib import Path
import argparse,itertools,json,math,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'deepseek_r1_llama_8b':'DeepSeek-8B','phi4_mini_reasoning':'Phi-mini','qwen3_4b':'Qwen3-4B','qwen3_8b':'Qwen3-8B','qwen3_14b':'Qwen3-14B','granite_3_2_8b_instruct':'Granite-8B'}
FIELDS=['model','chat_mode','decoding','scenario_id','category','value','option_count','candidate_set_index','repeat_seed','budget','chosen_bid','test_utility','test_optimal_utility','test_utility_ratio','test_regret','test_correct','train_correct','train_utility','train_optimal_utility','elapsed_seconds','constrained_readout_elapsed_seconds']

def merge_float_cuts(values):
    result=[]
    for x in sorted(values):
        if not result or abs(x-result[-1])>max(1e-15,abs(x)*1e-10):result.append(x)
    return result

def load_records(root):
    paths=list((root/'robustness_extensions/server_outputs/raw').glob('fixed_*.jsonl'))
    for m in ['qwen3_14b','granite_3_2_8b_instruct']:
        paths.extend((root/'preserved_outputs'/m/'historical/raw').glob('fixed_*.jsonl'))
    rows=[]
    for p in sorted(paths):
        with p.open() as f:
            for line in f:
                r=json.loads(line)
                rows.append({k:r.get(k) for k in FIELDS})
    d=pd.DataFrame(rows);d['scenario_id']=d['scenario_id'].astype(str).str.zfill(12)
    assert not d.duplicated(['model','chat_mode','decoding','scenario_id','budget','repeat_seed']).any()
    assert len(d)==50544
    for m,g in d.groupby('model'):
        assert len(g)==8424
    return d

def simplex_equilibrium_vertices(P,c,k):
    # Original symmetric Tullock game is strategically equivalent to zero sum:
    # adding the rival's cost to own payoff gives A=P-k*c_i+k*c_j, A+A.T=1.
    # R=A-.5 is skew symmetric. Every optimal strategy lies in {x>=0, sum x=1, R x<=0}.
    R=P-.5-k*(c[:,None]-c[None,:]);n=len(c)
    assert np.max(np.abs(R+R.T))<1e-12
    G=np.vstack([R,-np.eye(n)])
    vertices=[]
    for active in itertools.combinations(range(2*n),n-1):
        M=np.vstack([np.ones(n),G[list(active)]])
        if np.linalg.matrix_rank(M,tol=1e-12)<n:continue
        x=np.linalg.solve(M,np.r_[1.,np.zeros(n-1)])
        if np.min(x)<-1e-8 or np.max(R@x)>1e-9:continue
        x[np.abs(x)<1e-10]=0;x=x/x.sum()
        if not any(np.max(np.abs(x-v))<1e-7 for v in vertices):vertices.append(x)
    assert vertices,'No minimax-polytope vertices'
    residual=max(float(np.max(R@v)) for v in vertices)
    assert residual<1e-9
    return np.array(vertices),residual

def contest_analysis(stochastic,out):
    c=np.array([128,384,768],float);results=[];regimes=[]
    for m,g in stochastic.groupby('model',sort=True):
        q=g.groupby('budget')['test_utility_ratio'].mean().reindex(c.astype(int)).to_numpy()
        accuracy=g.groupby('budget')['test_correct'].mean().reindex(c.astype(int)).to_numpy()
        t=(c-c[0])/(c[-1]-c[0])
        shapes={'measured':q,'flat':np.repeat(q[0],3),'linear':q[0]+(q[-1]-q[0])*t,'concave_sqrt':q[0]+(q[-1]-q[0])*np.sqrt(t),'convex_square':q[0]+(q[-1]-q[0])*t*t,'saturating':q[0]+(q[-1]-q[0])*(1-np.exp(-3*t))/(1-np.exp(-3)),'shift_plus_half':q+.5,'scale_times_two':2*q,'accuracy':accuracy}
        for name,s in shapes.items():
            for gamma in [0.5,1,2,4]:
                z=s**gamma;P=z[:,None]/(z[:,None]+z[None,:]);Q=P*s[:,None]+(1-P)*s[None,:];C=c[:,None]+c[None,:]
                upper=min((.5-P[a,-1])/(c[-1]-c[a]) for a in range(2))
                lower=max((P[a,0]-.5)/(c[a]-c[0]) for a in range(1,3))
                first=min((s[-1]-Q[a,b])/(2*c[-1]-C[a,b]) for a in range(3) for b in range(3) if C[a,b]<2*c[-1])
                results.append({'model':m,'mapping':name,'gamma':gamma,'q128':s[0],'q384':s[1],'q768':s[2],'high_symmetric_NE_ceases_at_kappa':upper,'low_symmetric_NE_starts_at_kappa':lower,'planner_high_high_ceases_at_kappa_over_rho':first})
                if name!='measured':continue
                cuts={0.}
                for j in range(3):
                    for a,b in itertools.combinations(range(3),2):
                        v=(P[a,j]-P[b,j])/(c[a]-c[b])
                        if v>0:cuts.add(float(v))
                cuts=merge_float_cuts(cuts);cuts.append(max(0.0012,2*max(cuts)))
                for left,right in zip(cuts,cuts[1:]):
                    k=(left+right)/2;verts,residual=simplex_equilibrium_vertices(P,c,k)
                    A=P-k*c[:,None];B=1-P-k*c[None,:]
                    pure=[]
                    for a,b in itertools.product(range(3),repeat=2):
                        if A[a,b]>=max(A[:,b])-1e-11 and B[a,b]>=max(B[a,:])-1e-11:pure.append([int(c[a]),int(c[b])])
                    W=Q-k*C # rho=1 consequence; not anchored economic welfare.
                    losses=[]
                    for x,y in itertools.product(verts,repeat=2):
                        assert max(A@y)-float(x@A@y)<1e-9
                        assert max(x@B)-float(x@B@y)<1e-9
                        losses.append(float(np.max(W)-x@W@y))
                    regimes.append({'model':m,'gamma':gamma,'left_kappa':left,'right_kappa':right,'evaluation_kappa':k,'pure_equilibria':json.dumps(pure),'equilibrium_strategy_vertices':json.dumps(verts.tolist()),'n_strategy_vertices':len(verts),'min_expected_total_compute':2*float(np.min(verts@c)),'max_expected_total_compute':2*float(np.max(verts@c)),'best_equilibrium_welfare_loss_rho1':min(losses),'worst_equilibrium_welfare_loss_rho1':max(losses),'deviation_residual':residual,'interval_type':'open'} )
                # Report exact best-response indifference boundaries too; full equilibrium
                # sets at those points can include asymmetric and mixed profiles.
                for k in cuts[:-1]:
                    verts,residual=simplex_equilibrium_vertices(P,c,k)
                    regimes.append({'model':m,'gamma':gamma,'left_kappa':k,'right_kappa':k,'evaluation_kappa':k,'pure_equilibria':'boundary: strategy polytope below','equilibrium_strategy_vertices':json.dumps(verts.tolist()),'n_strategy_vertices':len(verts),'min_expected_total_compute':2*float(np.min(verts@c)),'max_expected_total_compute':2*float(np.max(verts@c)),'best_equilibrium_welfare_loss_rho1':np.nan,'worst_equilibrium_welfare_loss_rho1':np.nan,'deviation_residual':residual,'interval_type':'boundary'})
    pd.DataFrame(results).to_csv(out/'contest_mapping_sensitivity.csv',index=False)
    pd.DataFrame(regimes).to_csv(out/'complete_equilibrium_regimes.csv',index=False)
    return pd.DataFrame(results),pd.DataFrame(regimes)

def policy_intervals(c,actual,proxy):
    cuts={0.}
    for r in [actual,proxy]:
        for a,b in itertools.combinations(range(len(c)),2):
            v=(r[a]-r[b])/(c[a]-c[b])
            if v>0:cuts.add(float(v))
    cuts=merge_float_cuts(cuts);cuts.append(2*max(cuts))
    result=[]
    for lo,hi in zip(cuts,cuts[1:]):
        k=(lo+hi)/2;i=int(np.argmax(actual-k*c));j=int(np.argmax(proxy-k*c))
        regret=(actual[i]-k*c[i])-(actual[j]-k*c[j])
        maxreg=max(float(np.max(actual-x*c)-(actual[j]-x*c[j])) for x in [lo,hi])
        result.append({'left_lambda':lo,'right_lambda':hi,'true_profit_budget':int(c[i]),'ratio_proxy_budget':int(c[j]),'point_profit_regret_usd':float(regret),'interval_max_profit_regret_usd':maxreg,'hypothetical_cost_domain':True})
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--draws',type=int,default=10000);args=p.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    d=load_records(args.snapshot);print('Loaded 50,544 historical rows',flush=True)
    greedy=d[(d.chat_mode=='reasoning')&(d.decoding=='greedy')&(d.budget>=0)].copy()
    stochastic=d[(d.chat_mode=='reasoning')&(d.decoding=='stochastic')].copy()
    budgets=np.array([0,64,128,256,512,768]);models=sorted(LABELS)
    scenarios=pd.read_csv(args.snapshot/'new_models_full_server_package/data/historical/scenarios.csv',dtype={'scenario_id':str}).sort_values('scenario_id')
    ids=scenarios.scenario_id.astype(str).str.zfill(12).tolist();scenarios['scenario_id']=ids
    categories=sorted(scenarios.category.unique());family=[(r.category,float(r.value)) for r in scenarios.itertuples()]
    assert len(set(family))==54
    family_ids={cat:sorted({f for f in family if f[0]==cat}) for cat in categories}
    arrays=[];summary=[];policy=[];scoring=[];cost=[]
    for m in models:
        g=greedy[greedy.model==m];cell=g.groupby(['scenario_id','budget']).mean(numeric_only=True)
        cols=['test_utility_ratio','test_utility','test_correct']
        ar=np.stack([cell[col].unstack('budget').reindex(index=ids,columns=budgets).to_numpy() for col in cols],axis=-1)
        assert np.isfinite(ar).all();arrays.append(ar)
        oracle=cell['test_optimal_utility'].unstack().reindex(index=ids,columns=budgets).to_numpy()
        assert np.max(np.ptp(oracle,axis=1))<1e-10
        oraclemean=float(oracle[:,0].mean())
        mean=ar.mean(axis=0)
        for b,r in zip(budgets,mean):
            summary.append({'model':m,'budget':int(b),'mean_utility_ratio':r[0],'mean_bidder_profit_usd':r[1],'mean_accuracy':r[2],'mean_oracle_profit_usd':oraclemean,'aggregate_profit_ratio':r[1]/oraclemean,'dollar_scaled_equal_ratio_proxy':r[0]*oraclemean})
        for r in policy_intervals(budgets,mean[:,1],oraclemean*mean[:,0]):policy.append({'model':m,**r})
        for b,gb in g.groupby('budget'):
            scoring.append({'model':m,'budget':int(b),'construction_optimality':float(gb.train_correct.mean()),'heldout_optimality':float(gb.test_correct.mean()),'construction_utility_ratio':float((gb.train_utility/gb.train_optimal_utility).mean()),'heldout_utility_ratio':float(gb.test_utility_ratio.mean())})
        for dec in ['greedy','stochastic']:
            cg=d[(d.model==m)&(d.chat_mode=='reasoning')&(d.decoding==dec)&(d.budget>=0)]
            x=cg.budget.to_numpy();elapsed=cg.elapsed_seconds.to_numpy();readout=cg.constrained_readout_elapsed_seconds.fillna(0).to_numpy()
            slope=lambda y:float(np.dot(x-x.mean(),y-y.mean())/np.dot(x-x.mean(),x-x.mean()))
            cost.append({'model':m,'decoding':dec,'legacy_decision_seconds_per_token':slope(elapsed),'constrained_readout_seconds_per_token':slope(readout),'combined_audit_pipeline_seconds_per_token':slope(elapsed+readout),'mean_constrained_readout_seconds':float(readout.mean()),'pricing_note':'Audit pipeline includes discarded free-form answer; not isolated deployment decision cost'})
    A=np.stack(arrays) # model, scenario, budget, metric
    rng=np.random.default_rng(20261006);scenario_draw=np.zeros((args.draws,216));family_draw=np.zeros_like(scenario_draw)
    for cat in categories:
        ix=np.flatnonzero(scenarios.category.to_numpy()==cat)
        assert len(ix)==72
        selected=rng.integers(0,72,(args.draws,72));counts=np.zeros((args.draws,72))
        np.add.at(counts,(np.arange(args.draws)[:,None],selected),1);scenario_draw[:,ix]=counts/216
        f=family_ids[cat];assert len(f)==18
        selected=rng.integers(0,18,(args.draws,18));counts=np.zeros((args.draws,18))
        np.add.at(counts,(np.arange(args.draws)[:,None],selected),1)
        for j,key in enumerate(f):
            positions=[i for i,x in enumerate(family) if x==key];assert len(positions)==4
            family_draw[:,positions]=counts[:,[j]]/216
    assert np.allclose(scenario_draw.sum(axis=1),1) and np.allclose(family_draw.sum(axis=1),1)
    point=A.mean(axis=1);sdraw=np.einsum('ds,msbk->dmbk',scenario_draw,A);fdraw=np.einsum('ds,msbk->dmbk',family_draw,A)
    contrasts=list(zip(range(5),range(1,6)))+[(0,5)]
    family_delta=np.stack([fdraw[:,:,b,:2]-fdraw[:,:,a,:2] for a,b in contrasts],axis=2)
    point_delta=np.stack([point[:,b,:2]-point[:,a,:2] for a,b in contrasts],axis=1)
    sd=family_delta.std(axis=0,ddof=1);safe=np.where(sd>1e-14,sd,np.inf)
    maxstat=np.max(np.abs((family_delta-point_delta)/safe),axis=(1,2,3));critical=float(np.quantile(maxstat,.95))
    comparisons=[]
    for mi,m in enumerate(models):
        for ci,(a,b) in enumerate(contrasts):
            r={'model':m,'low_budget':int(budgets[a]),'high_budget':int(budgets[b]),'point_reversal':bool(point_delta[mi,ci,0]*point_delta[mi,ci,1]<0)}
            for j,metric in enumerate(['utility_ratio','profit_usd']):
                r[f'delta_{metric}']=float(point_delta[mi,ci,j])
                fs=family_delta[:,mi,ci,j];ss=sdraw[:,mi,b,j]-sdraw[:,mi,a,j]
                for label,values in [('family',fs),('scenario',ss)]:
                    low,high=np.quantile(values,[.025,.975]);r[f'{metric}_{label}_ci_low']=low;r[f'{metric}_{label}_ci_high']=high
                r[f'{metric}_simultaneous_low']=point_delta[mi,ci,j]-critical*sd[mi,ci,j]
                r[f'{metric}_simultaneous_high']=point_delta[mi,ci,j]+critical*sd[mi,ci,j]
            r['pointwise_family_supported_reversal']=bool((r['utility_ratio_family_ci_low']>0 and r['profit_usd_family_ci_high']<0)or(r['utility_ratio_family_ci_high']<0 and r['profit_usd_family_ci_low']>0))
            r['simultaneous_supported_reversal']=bool((r['utility_ratio_simultaneous_low']>0 and r['profit_usd_simultaneous_high']<0)or(r['utility_ratio_simultaneous_high']<0 and r['profit_usd_simultaneous_low']>0))
            comparisons.append(r)
    summary=pd.DataFrame(summary);comparisons=pd.DataFrame(comparisons);policy=pd.DataFrame(policy)
    summary.to_csv(out/'metric_frontiers.csv',index=False);comparisons.to_csv(out/'paired_metric_contrasts.csv',index=False)
    policy.to_csv(out/'metric_policy_intervals.csv',index=False);pd.DataFrame(scoring).to_csv(out/'construction_vs_heldout.csv',index=False);pd.DataFrame(cost).to_csv(out/'cost_pipeline_audit.csv',index=False)
    print('Metric bootstrap and policy envelopes complete',flush=True)
    maps,regimes=contest_analysis(stochastic,out)
    # Plot conditional measurement outcomes, not total auction welfare.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axs=plt.subplots(2,3,figsize=(12,7.5),layout='constrained')
    for ax,m in zip(axs.flat,models):
        g=summary[summary.model==m];ax.plot(g.budget,g.mean_utility_ratio,marker='o',label='Mean scenario profit ratio');ax.plot(g.budget,g.aggregate_profit_ratio,marker='s',label='Aggregate dollar profit / oracle');ax.set_title(LABELS[m]);ax.set_ylim(.55,1.01);ax.set_xlabel('Assigned reasoning tokens');ax.set_ylabel('Held-out profit ratio');ax.grid(alpha=.2)
    axs.flat[0].legend(fontsize=8);fig.suptitle('The averaging rule changes the economic frontier (point estimates)',fontsize=15)
    fig.savefig(out/'metric_frontiers.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12.5,5.0))
    colors={'measured':'#176d9c','linear':'#d58430','concave_sqrt':'#47854b','convex_square':'#a84b66','shift_plus_half':'#7764a3'}
    for name,color in colors.items():
        g=maps[(maps.mapping==name)&(maps.gamma==1)].set_index('model').reindex(models)
        offset=(list(colors).index(name)-2)*.10
        axs[0].scatter(g.high_symmetric_NE_ceases_at_kappa,np.arange(6)+offset,s=36,label=name.replace('_',' '),color=color)
        axs[1].scatter(g.planner_high_high_ceases_at_kappa_over_rho,np.arange(6)+offset,s=36,color=color)
    for ax,title in zip(axs,['Private maximum-budget boundary','Planner maximum-budget boundary']):
        ax.set_yticks(range(6),[LABELS[m] for m in models]);ax.invert_yaxis();ax.set_title(title);ax.set_xscale('log');ax.grid(alpha=.2)
    axs[0].set_xlabel('κ threshold (γ = 1)');axs[1].set_xlabel('κ/ρ threshold (γ = 1)')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=5,fontsize=9,frameon=False)
    fig.suptitle('Boundary locations depend on curve shape and quality origin',fontsize=15)
    fig.subplots_adjust(left=.14,right=.98,top=.84,bottom=.20,wspace=.43)
    fig.savefig(out/'contest_boundary_sensitivity.png');plt.close(fig)
    report={'date':'2026-10-06','inference':'none','historical_rows':len(d),'scenario_count':216,'category_value_families':54,'fixed_products':3,'bootstrap_draws':args.draws,'bootstrap_seed':20261006,'uncertainty':'Both bootstraps stratified by fixed product; family units keep four candidate variants together. Conditional on prompts, fixed CDFs and fixed products. No independent confirmation.','simultaneous_contrast_count':72,'simultaneous_max_standardized_critical_value':critical,'point_reversals':int(comparisons.point_reversal.sum()),'pointwise_family_supported_reversals':int(comparisons.pointwise_family_supported_reversal.sum()),'simultaneous_supported_reversals':int(comparisons.simultaneous_supported_reversal.sum()),'open_interval_regimes':int((regimes.interval_type=='open').sum()),'mixed_only_open_intervals':int(((regimes.interval_type=='open')&(regimes.pure_equilibria=='[]')).sum()),'max_deviation_residual':float(regimes.deviation_residual.max()),'indifference_cut_merge_tolerance':'max(1e-15,1e-10*abs(cut)); eliminates machine-precision duplicate intersections','plots':['metric_frontiers.png','contest_boundary_sensitivity.png']}
    (out/'analysis_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
