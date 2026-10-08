from pathlib import Path
import csv,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
MODELS=['deepseek_r1_llama_8b','phi4_mini_reasoning','qwen3_4b','qwen3_8b','qwen3_14b','granite_3_2_8b_instruct']
LABELS=['DeepSeek-8B','Phi-mini','Qwen3-4B','Qwen3-8B','Qwen3-14B','Granite-8B'];BUDGETS=[0,64,128,256,512,768]
def main():
 out=ROOT/'results';out.mkdir(exist_ok=True)
 scenarios=pd.read_csv(ROOT/'data/scenarios.csv',dtype={'scenario_id':str});scenarios['scenario_id']=scenarios.scenario_id.str.zfill(12);scenarios=scenarios.sort_values('scenario_id');ids=scenarios.scenario_id.tolist()
 info={};baseline=[]
 for r in scenarios.to_dict('records'):
  options=json.loads(r['candidates_json']);v=float(r['value']);calc=max(options,key=lambda x:(x['train_u'],-x['bid']));display=max(options,key=lambda x:((v-x['bid'])*float(f"{x['train_p']:.4f}"),-x['bid']));oracle=max(x['test_u'] for x in options)
  for x in options:assert np.isclose(x['train_u'],(v-x['bid'])*x['train_p']) and np.isclose(x['test_u'],(v-x['bid'])*x['test_p'])
  info[r['scenario_id']]={'options':{x['bid']:x for x in options},'calc':calc,'display_calc':display,'display_max':(v-display['bid'])*float(f"{display['train_p']:.4f}"),'oracle':oracle}
  baseline.append({'scenario_id':r['scenario_id'],'category':r['category'],'value':v,'calculator_bid':calc['bid'],'calculator_test_profit':calc['test_u'],'calculator_ratio':calc['test_u']/oracle,'display_calculator_test_profit':display['test_u'],'calculator_regret':oracle-calc['test_u'],'display_calculator_bid':display['bid'],'rounding_changes_calculator':display['bid']!=calc['bid'],'calculator_heldout_optimal':abs(oracle-calc['test_u'])<1e-9})
 base=pd.DataFrame(baseline);base.to_csv(out/'calculator_scenarios.csv',index=False)
 archived=pd.read_csv(ROOT/'data/archived_analytic_baselines.csv',dtype={'scenario_id':str});archived['scenario_id']=archived.scenario_id.str.zfill(12);archived=archived[archived.model=='calculator_train'].groupby('scenario_id').test_utility.mean().reindex(ids)
 assert np.allclose(archived.to_numpy(),base.calculator_test_profit.to_numpy(),atol=1e-10)
 d=pd.read_csv(ROOT/'data/saved_greedy_decisions.csv',dtype={'scenario_id':str});d['scenario_id']=d.scenario_id.str.zfill(12)
 assert len(d)==23328 and not d.duplicated(['model','scenario_id','budget','repeat_seed']).any()
 rows=[]
 for r in d.to_dict('records'):
  z=info[r['scenario_id']];x=z['options'][r['chosen_bid']];du=(r['value']-x['bid'])*float(f"{x['train_p']:.4f}");prompt_opt=z['display_max']-du<1e-9;test_opt=z['oracle']-x['test_u']<1e-9
  assert abs(x['test_u']-r['test_utility'])<1e-9 and abs(x['train_u']-r['train_utility'])<1e-9
  cls=('both_optimal' if prompt_opt and test_opt else 'display_optimal_heldout_suboptimal' if prompt_opt else 'display_suboptimal_heldout_optimal' if test_opt else 'both_suboptimal')
  gap=x['test_u']-z['calc']['test_u'];regret=z['oracle']-x['test_u'];calc_regret=z['oracle']-z['calc']['test_u'];assert abs(regret-(calc_regret-gap))<1e-8
  rows.append({**r,'calculator_test_profit':z['calc']['test_u'],'profit_gap_vs_calculator':gap,'profit_gap_vs_display_calculator':x['test_u']-z['display_calc']['test_u'],'displayed_objective_regret':z['display_max']-du,'displayed_optimal':prompt_opt,'heldout_utility_optimal':test_opt,'error_class':cls,'calculator_regret':calc_regret,'calculator_ratio':z['calc']['test_u']/z['oracle'],'ratio_gap_vs_calculator':r['test_utility_ratio']-z['calc']['test_u']/z['oracle'],'beats_calculator_on_heldout':gap>1e-9})
 audit=pd.DataFrame(rows);audit.to_csv(out/'decision_error_audit.csv',index=False)
 metrics=['profit_gap_vs_calculator','ratio_gap_vs_calculator','displayed_objective_regret','displayed_optimal','heldout_utility_optimal','beats_calculator_on_heldout','profit_gap_vs_display_calculator']
 a=[];class_arrays={}
 for m in MODELS:
  g=audit[audit.model==m];cell=g.groupby(['scenario_id','budget'])[metrics].mean()
  a.append(np.stack([cell[k].unstack('budget').reindex(index=ids,columns=BUDGETS).to_numpy() for k in metrics],axis=-1))
  class_arrays[m]={c:g.assign(rate=(g.error_class==c).astype(float)).groupby(['scenario_id','budget']).rate.mean().unstack('budget').reindex(index=ids,columns=BUDGETS).to_numpy() for c in ['both_optimal','display_optimal_heldout_suboptimal','display_suboptimal_heldout_optimal','both_suboptimal']}
 A=np.array(a);assert np.isfinite(A).all();point=A.mean(axis=1);rng=np.random.default_rng(20261009);weights=np.zeros((10000,216))
 for cat in sorted(scenarios.category.unique()):
  vals=sorted(scenarios[scenarios.category==cat].value.unique());draw=rng.integers(0,18,(10000,18));counts=np.array([(draw==i).sum(axis=1) for i in range(18)]).T
  for i,value in enumerate(vals):
   ix=np.flatnonzero((scenarios.category.to_numpy()==cat)&(scenarios.value.to_numpy()==value));assert len(ix)==4;weights[:,ix]=counts[:,[i]]/216
 assert np.allclose(weights.sum(axis=1),1)
 boot=np.einsum('ds,msbk->dmbk',weights,A);ci=lambda x:np.quantile(x,[.025,.975])
 high=boot[:,:,5,0];sd=high.std(axis=0,ddof=1);critical=np.quantile(np.max(np.abs((high-point[:,5,0])/sd),axis=1),.95);bands=np.column_stack([point[:,5,0]-critical*sd,point[:,5,0]+critical*sd])
 display_high=boot[:,:,5,6];display_sd=display_high.std(axis=0,ddof=1);display_critical=np.quantile(np.max(np.abs((display_high-point[:,5,6])/display_sd),axis=1),.95);display_bands=np.column_stack([point[:,5,6]-display_critical*display_sd,point[:,5,6]+display_critical*display_sd])
 summary=[]
 for mi,m in enumerate(MODELS):
  for bi,b in enumerate(BUDGETS):
   g=audit[(audit.model==m)&(audit.budget==b)];lo,hi=ci(boot[:,mi,bi,0]);delta=boot[:,mi,bi,0]-boot[:,mi,0,0]
   r={'model':m,'label':LABELS[mi],'budget':b,'mean_profit_gap_vs_calculator':point[mi,bi,0],'profit_gap_family_ci_low':lo,'profit_gap_family_ci_high':hi,'mean_ratio_gap_vs_calculator':point[mi,bi,1],'displayed_objective_regret_usd':point[mi,bi,2],'displayed_optimal_rate':point[mi,bi,3],'heldout_utility_optimal_rate':point[mi,bi,4],'heldout_beats_calculator_rate':point[mi,bi,5],'mean_llm_profit_usd':g.test_utility.mean(),'mean_gap_vs_display_calculator':point[mi,bi,6],'mean_calculator_profit_usd':g.calculator_test_profit.mean(),'llm_heldout_regret_usd':(g.test_optimal_utility-g.test_utility).mean(),'calculator_heldout_regret_usd':g.calculator_regret.mean()}
   r.update({c+'_rate':class_arrays[m][c][:,bi].mean() for c in class_arrays[m]})
   assert np.isclose(sum(r[c+'_rate'] for c in class_arrays[m]),1)
   if b==768:r.update({'display_calculator_simultaneous_gap_low':display_bands[mi,0],'display_calculator_simultaneous_gap_high':display_bands[mi,1],'simultaneous_profit_gap_low':bands[mi,0],'simultaneous_profit_gap_high':bands[mi,1],'profit_improvement_0_to_768':point[mi,bi,0]-point[mi,0,0],'improvement_family_ci_low':ci(delta)[0],'improvement_family_ci_high':ci(delta)[1]})
   summary.append(r)
 pd.DataFrame(summary).to_csv(out/'calculator_comparison.csv',index=False)
 manifest={'rows':len(audit),'scenarios':216,'value_families':54,'products':3,'budgets':BUDGETS,'draws':10000,'bootstrap_seed':20261009,'design':'Primary greedy frontier, averaged over three option orders; within-product value-family paired bootstrap retains four candidate variants and complete budget vectors.','prompt_probabilities':'Exactly reconstructed four-decimal display from archived formatter; stored full-precision train_u remains the published calculator baseline.','rounding_changed_baseline_scenarios':int(base.rounding_changes_calculator.sum()),'calculator_mean_profit_usd':float(base.calculator_test_profit.mean()),'calculator_mean_ratio':float(base.calculator_ratio.mean()),'display_calculator_mean_profit_usd':float(base.display_calculator_test_profit.mean()),'all_six_display_calculator_simultaneous_gap_upper_bounds_negative':bool(np.all(display_bands[:,1]<0)),'calculator_mean_regret_usd':float(base.calculator_regret.mean()),'calculator_heldout_optimal_scenarios':int(base.calculator_heldout_optimal.sum()),'simultaneous_panel':'Six model-minus-calculator dollar-profit gaps at 768; exploratory 95% maximum-standardized-deviation bands.','simultaneous_critical_value':float(critical),'all_six_simultaneous_profit_gap_upper_bounds_negative':bool(np.all(bands[:,1]<0)),'new_inference':False,'limits':'Conditional on fixed products, prompts, construction/scoring CDFs and order design. Choice categories describe observable objective agreement, not an internal reasoning mechanism. Calculator runtime/cost placeholders are not measurements; no measured cost-effectiveness or net deployment-payoff comparison is claimed.','input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*') if p.is_file()}}
 (out/'audit_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2));print(pd.DataFrame(summary).query('budget==768').to_string(index=False))
if __name__=='__main__':main()
