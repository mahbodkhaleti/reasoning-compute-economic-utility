"""CPU-only paired family/seed uncertainty and service-selection sensitivity."""
from pathlib import Path
import argparse,csv,hashlib,json,itertools
import numpy as np
BUDGETS=np.array([128,384,768],float)
MODELS=['deepseek_r1_llama_8b','phi4_mini_reasoning','qwen3_4b','qwen3_8b','qwen3_14b','granite_3_2_8b_instruct']
LABELS=['DeepSeek-8B','Phi-mini','Qwen3-4B','Qwen3-8B','Qwen3-14B','Granite-8B']
SEEDS=[1,7,21,42,99]
def load(root):
 scenarios=list(csv.DictReader((root/'new_models_full_server_package/data/historical/scenarios.csv').open()))
 scenarios.sort(key=lambda x:x['scenario_id'].zfill(12));ids=[r['scenario_id'].zfill(12) for r in scenarios];sid={x:i for i,x in enumerate(ids)}
 paths=list((root/'robustness_extensions/server_outputs/raw').glob('*_reasoning_stochastic.jsonl'))
 for m in ['qwen3_14b','granite_3_2_8b_instruct']:paths+=list((root/'preserved_outputs'/m/'historical/raw').glob('*_reasoning_stochastic.jsonl'))
 a=np.full((6,216,5,3,3),np.nan);oracle=np.full(216,np.nan);seen=set()
 for p in sorted(paths):
  for line in p.open():
   r=json.loads(line);m=MODELS.index(r['model']);s=sid[str(r['scenario_id']).zfill(12)];seed=SEEDS.index(r['repeat_seed']);b=list(BUDGETS).index(r['budget']);key=(m,s,seed,b)
   assert key not in seen;seen.add(key)
   a[m,s,seed,b]=[r['test_utility_ratio'],float(r['test_correct']),r['test_utility']]
   if np.isfinite(oracle[s]):assert abs(oracle[s]-r['test_optimal_utility'])<1e-9
   oracle[s]=r['test_optimal_utility']
 assert len(seen)==19440 and np.isfinite(a).all() and np.all(oracle>0)
 families=[]
 for cat in sorted(set(r['category'] for r in scenarios)):
  groups=[]
  for v in sorted(set(float(r['value']) for r in scenarios if r['category']==cat)):
   ix=[i for i,r in enumerate(scenarios) if r['category']==cat and float(r['value'])==v];assert len(ix)==4;groups.append(ix)
  assert len(groups)==18;families.append(np.array(groups))
 return a,oracle,families,paths

def probability(q,kind='tullock',scale=None):
 if kind=='tullock':return q[..., :,None]/(q[..., :,None]+q[...,None,:])
 diff=(q[..., :,None]-q[...,None,:])/np.asarray(scale)[...,None,None]
 return 1/(1+np.exp(-diff))
def high_threshold(P):
 return np.minimum((.5-P[...,0,2])/640,(.5-P[...,1,2])/384)
def dominance_gap(P,k,chosen):
 pay=P-k*BUDGETS[:,None]
 return np.min(np.stack([pay[...,chosen,:]-pay[...,i,:] for i in range(3) if i!=chosen],axis=-2),axis=(-2,-1))
def ci(x):return [float(v) for v in np.quantile(x,[.025,.975])]
def dump_csv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze(root,out,draws,seed):
 out.mkdir(parents=True,exist_ok=True);a,oracle,families,paths=load(root);rng=np.random.default_rng(seed)
 point=a.mean(axis=(1,2));q=point[:,:,0];accuracy=point[:,:,1];profit=point[:,:,2]/oracle.mean();qref=q.mean(axis=1)
 bs=np.empty((draws,6,3,3));sampled_oracle=np.empty(draws)
 for d in range(draws):
  ix=np.concatenate([g[rng.integers(0,18,18)].ravel() for g in families]);sampled_oracle[d]=oracle[ix].mean()
  for m in range(6):
   si=rng.integers(0,5,216);bs[d,m]=a[m,ix,si].mean(axis=0)
  if (d+1)%1000==0:print(f'Paired draws: {d+1}/{draws}',flush=True)
 qb=bs[:,:,:,0];ab=bs[:,:,:,1];pb=bs[:,:,:,2]/sampled_oracle[:,None,None]
 linear=lambda x:x[...,:1]+(x[...,-1:]-x[...,:1])*(BUDGETS-128)/640
 ps={'utility':probability(q),'accuracy':probability(accuracy),'linear':probability(linear(q)),'profit':probability(profit),'logit_half':probability(q,'logit',2*qref),'logit_matched':probability(q,'logit',qref),'logit_double':probability(q,'logit',qref/2)}
 bps={'utility':probability(qb),'accuracy':probability(ab),'linear':probability(linear(qb)),'profit':probability(pb),'logit_half':probability(qb,'logit',2*qref[None,:]),'logit_matched':probability(qb,'logit',qref[None,:]),'logit_double':probability(qb,'logit',qref[None,:]/2)}
 pt={k:high_threshold(v) for k,v in ps.items()};bt={k:high_threshold(v) for k,v in bps.items()}
 # Nonpositive high thresholds indicate that high/high lacks support at positive cost.
 valid=bt['utility']>0
 invalid_counts=np.sum(~valid,axis=0)
 differences=bt['accuracy']-bt['utility'];pointdiff=pt['accuracy']-pt['utility'];sd=differences.std(axis=0,ddof=1)
 critical=float(np.quantile(np.max(np.abs((differences-pointdiff)/sd),axis=1),.95));bands=np.column_stack([pointdiff-critical*sd,pointdiff+critical*sd])
 results=[]
 for m,name in enumerate(MODELS):
  rr=bt['accuracy'][valid[:,m],m]/bt['utility'][valid[:,m],m];lc=100*(bt['linear'][valid[:,m],m]/bt['utility'][valid[:,m],m]-1)
  results.append({'model':name,'label':LABELS[m],'utility_threshold':float(pt['utility'][m]),'utility_ci_low':ci(bt['utility'][:,m])[0],'utility_ci_high':ci(bt['utility'][:,m])[1],'accuracy_threshold':float(pt['accuracy'][m]),'accuracy_ratio':float(pt['accuracy'][m]/pt['utility'][m]),'conditional_positive_support_ratio_ci_low':ci(rr)[0],'conditional_positive_support_ratio_ci_high':ci(rr)[1],'nonpositive_utility_threshold_draws':int(invalid_counts[m]),'threshold_difference':float(pointdiff[m]),'difference_ci_low':ci(differences[:,m])[0],'difference_ci_high':ci(differences[:,m])[1],'simultaneous_difference_low':float(bands[m,0]),'simultaneous_difference_high':float(bands[m,1]),'linear_percent_change':float(100*(pt['linear'][m]/pt['utility'][m]-1)),'conditional_linear_change_ci_low':ci(lc)[0],'conditional_linear_change_ci_high':ci(lc)[1],'linear_threshold_difference_ci_low':ci(bt['linear'][:,m]-bt['utility'][:,m])[0],'linear_threshold_difference_ci_high':ci(bt['linear'][:,m]-bt['utility'][:,m])[1]})
 mapping=[]
 for kind in ps:
  for m,name in enumerate(MODELS):
   mapping.append({'model':name,'label':LABELS[m],'mapping':kind,'threshold':float(pt[kind][m]),'threshold_ci_low':ci(bt[kind][:,m])[0],'threshold_ci_high':ci(bt[kind][:,m])[1],'relative_to_utility':float(pt[kind][m]/pt['utility'][m]),'nonpositive_threshold_draws':int(np.sum(bt[kind][:,m]<=0))})
 ex_util=dominance_gap(bps['utility'][:,0],.00015,0)>1e-12;ex_acc=dominance_gap(bps['accuracy'][:,0],.00015,2)>1e-12
 joint=ex_util&ex_acc
 # Algebraic mapping invariants and explicit private-outcome checks.
 assert np.allclose(probability(2*q),ps['utility'])
 assert np.allclose(probability(q+.5,'logit',qref),ps['logit_matched'])
 assert not np.allclose(probability(q+.5),ps['utility'])
 # Profit quality predicts client dollar benefit under the constructed scoring environment.
 assert np.allclose(profit*oracle.mean(),point[:,:,2])
 old=list(csv.DictReader((root/'cpu_revision/point_comparison_input.csv').open()))
 for m,name in enumerate(MODELS):
  for kind,oldkind in [('utility','measured'),('accuracy','accuracy'),('linear','linear')]:
   expected=float(next(x for x in old if x['model']==name and x['mapping']==oldkind and float(x['gamma'])==1)['high_symmetric_NE_ceases_at_kappa'])
   assert np.isclose(pt[kind][m],expected,rtol=1e-10,atol=1e-15)
 dump_csv(out/'comparison_uncertainty.csv',results);dump_csv(out/'mapping_sensitivity.csv',mapping)
 np.savez_compressed(out/'bootstrap_thresholds.npz',**bt)
 manifest={'draws':draws,'seed':seed,'historical_stochastic_rows':19440,'scenario_count':216,'families':54,'products':3,'generation_seeds':SEEDS,'resampling':'Within each fixed product sample 18 value families with replacement, retain four variants, then sample one seed per selected scenario occurrence independently per model; use that seed jointly across budgets and metrics. Families sampled jointly across models.','simultaneous_panel':'Six accuracy-minus-utility private threshold differences; exploratory maximum-standardized-deviation 95% bands.','simultaneous_critical_value':critical,'nonpositive_utility_threshold_draws_by_model':dict(zip(MODELS,invalid_counts.tolist())),'ratio_interval_caveat':'Ratio intervals condition on positive utility thresholds. Primary difference intervals and bands retain every draw, including nonpositive thresholds. A nonpositive signed threshold means high/high is not supported at positive cost.','all_six_simultaneous_difference_lower_bounds_positive':bool(np.all(bands[:,0]>0)),'example':{'model':MODELS[0],'kappa':.00015,'utility_low_strict_dominance_frequency':float(ex_util.mean()),'accuracy_high_strict_dominance_frequency':float(ex_acc.mean()),'joint_frequency':float(joint.mean()),'joint_count':int(joint.sum()),'interpretation':'Bootstrap stability frequency, not a posterior or market probability.'},'quality_reference_by_model':dict(zip(MODELS,qref.tolist())),'mean_oracle_profit_usd':float(oracle.mean()),'input_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'new_inference':False,'limits':'Conditional on constructed tasks, three fixed products, original prompts and scoring CDF; excludes market shift, construction-split uncertainty, model/protocol misspecification and estimated client selection/value.'}
 (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'comparison_results':results,'example':manifest['example'],'simultaneous_positive':manifest['all_six_simultaneous_difference_lower_bounds_positive']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path);p.add_argument('--draws',type=int,default=10000);p.add_argument('--seed',type=int,default=20261008);args=p.parse_args();analyze(args.root,args.out or args.root/'cpu_revision/results',args.draws,args.seed)
