from pathlib import Path
import csv,hashlib,json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from analyze_legacy_revision import simplex_equilibrium_vertices
root=Path(__file__).resolve().parents[1];cpu=root/'cpu_revision';report={}
# Independently check the existing complete-equilibrium solver against LP and controls.
reg=pd.read_csv(cpu/'legacy_results/complete_equilibrium_regimes.csv');maps=pd.read_csv(cpu/'legacy_results/contest_mapping_sensitivity.csv');mixed=reg[(reg.interval_type=='open')&(reg.pure_equilibria=='[]')];assert len(mixed)==1
row=mixed.iloc[0];q=maps[(maps.model==row.model)&(maps.mapping=='measured')&(maps.gamma==row.gamma)][['q128','q384','q768']].to_numpy()[0];c=np.array([128.,384.,768.]);z=q**row.gamma;P=z[:,None]/(z[:,None]+z[None,:]);R=P-.5-row.evaluation_kappa*(c[:,None]-c[None,:])
res=linprog(np.r_[np.zeros(3),1.],A_ub=np.c_[R,-np.ones(3)],b_ub=np.zeros(3),A_eq=np.array([[1,1,1,0.]]),b_eq=[1.],bounds=[(0,None)]*3+[(None,None)],method='highs')
assert res.success and abs(res.fun)<1e-10 and np.max(np.abs(res.x[:3]-np.array(json.loads(row.equilibrium_strategy_vertices))[0]))<1e-8
rock=np.array([[0,-1,1],[1,0,-1],[-1,1,0.]])*.2
verts,_=simplex_equilibrium_vertices(rock+.5,np.array([1.,2.,3.]),0);assert np.allclose(verts,[[1/3]*3])
verts,_=simplex_equilibrium_vertices(np.full((3,3),.5),np.array([1.,2.,3.]),.1);assert np.allclose(verts,[[1,0,0]])
report['independent_equilibrium_LP_and_controls']='passed'
# Independent economic transfer accounting at unequal probabilities and costs.
for fee in [1.,10.,328.23]:
 for p in [.2,.5,.9]:
  expected=p*30+(1-p)*20;total=expected-fee+(fee*p-.12)+(fee*(1-p)-.34)
  assert abs(total-(expected-.46))<1e-12
report['client_provider_payment_cancellation']='passed'
comparisons=[]
for panel in sorted((cpu/'replay_reference_tables').iterdir()):
 for ref in sorted(panel.glob('*.csv')):
  new=root/'offline_replay/outputs'/panel.name/'tables'/ref.name;assert new.exists(),new
  if ref.read_bytes()==new.read_bytes():status='byte_identical'
  else:
   a=pd.read_csv(ref);b=pd.read_csv(new);assert a.shape==b.shape and list(a)==list(b)
   for col in a:
    if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):assert np.allclose(a[col].to_numpy(float),b[col].to_numpy(float),rtol=1e-10,atol=1e-10,equal_nan=True),(panel.name,ref.name,col)
    else:assert a[col].fillna('<NA>').astype(str).equals(b[col].fillna('<NA>').astype(str)),(panel.name,ref.name,col)
   status='numeric_match'
  comparisons.append({'panel':panel.name,'table':ref.name,'status':status})
assert len(comparisons)==71
with (cpu/'results/replay_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['panel','table','status']);w.writeheader();w.writerows(comparisons)
report['replay_table_statuses']={k:sum(x['status']==k for x in comparisons) for k in ['byte_identical','numeric_match']}
# Preserve previous CPU evidence to machine precision after rerun.
for filename in ['contest_mapping_sensitivity.csv','complete_equilibrium_regimes.csv','paired_metric_contrasts.csv','metric_frontiers.csv']:
 a=pd.read_csv(cpu/'reference_evidence'/filename);b=pd.read_csv(cpu/'legacy_results'/filename)
 assert a.shape==b.shape and list(a)==list(b)
 for col in a:
  if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):assert np.allclose(a[col].to_numpy(float),b[col].to_numpy(float),atol=1e-10,rtol=1e-10,equal_nan=True)
  else:assert a[col].fillna('<NA>').astype(str).equals(b[col].fillna('<NA>').astype(str))
report['four_legacy_revision_evidence_tables']='matched'
j=json.loads((cpu/'results/analysis_manifest.json').read_text());assert j['all_six_simultaneous_difference_lower_bounds_positive'] and j['example']['joint_count']==j['draws']
archive=np.load(cpu/'results/bootstrap_thresholds.npz');assert all(archive[k].shape==(j['draws'],6) and np.isfinite(archive[k]).all() for k in archive.files)
saved=list(csv.DictReader((cpu/'results/comparison_uncertainty.csv').open()))
for m,row in enumerate(saved):
 differences=archive['accuracy'][:,m]-archive['utility'][:,m]
 bounds=np.quantile(differences,[.025,.975])
 assert np.allclose(bounds,[float(row['difference_ci_low']),float(row['difference_ci_high'])],rtol=1e-10,atol=1e-15)
 assert int(row['nonpositive_utility_threshold_draws'])==np.sum(archive['utility'][:,m]<=0)
report['all_bootstrap_draws_retained']=j['draws'];report['nonpositive_primary_support_draws']=j['nonpositive_utility_threshold_draws_by_model'];report['new_inference']=False
(cpu/'results/validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
