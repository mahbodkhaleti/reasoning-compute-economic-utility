"""Analyze exported pilot responses; diagnostic sessions never count as observations."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
ROOT=Path(__file__).resolve().parent

def fit(x,y,fee=None):
 # Bounds prevent interpreting complete separation as a precise finite slope.
 X=np.column_stack([np.ones(len(y)),x]);bounds=[(-5,5),(0,100)]
 if fee is not None:
  X=np.column_stack([X,fee]);bounds.append((-100,0))
 fun=lambda b:float(np.logaddexp(0,X@b).sum()-y@(X@b))
 result=minimize(fun,np.zeros(X.shape[1]),jac=lambda b:X.T@(expit(X@b)-y),bounds=bounds,method='L-BFGS-B')
 assert result.success or np.isfinite(result.fun)
 return {'coefficients':result.x.tolist(),'negative_log_likelihood':float(result.fun),'bound_reached':bool(any(abs(b-lo)<1e-6 or abs(b-hi)<1e-6 for b,(lo,hi) in zip(result.x,bounds))),'converged':bool(result.success),'n':len(y)}

def read_sessions(directory,design):
 sessions=[];test_count=0;seen=set();profiles=design['profiles'];trials={x['trial_id']:x for x in design['trials']}
 for p in sorted(directory.glob('choice_*.json')):
  d=json.loads(p.read_text());pid=d['participant_id'];assert pid not in seen,'Duplicate participant code';seen.add(pid)
  if d.get('is_test') or pid.upper().startswith('TEST'):test_count+=1;continue
  assert d['study_version']==design['version'] and d.get('hypothetical_choices') is True
  assert len(d['responses'])==12 and {r['trial_id'] for r in d['responses']}==set(trials)
  for r in d['responses']:
   t=trials[r['trial_id']];assert r['block']==t['block'];assert set([r['left_profile'],r['right_profile']])==set([t['profile_1'],t['profile_2']]);assert r['choice'] in ['A','B']
   fees={t['profile_1']:t['fee_1'],t['profile_2']:t['fee_2']};assert r['left_fee']==fees[r['left_profile']] and r['right_fee']==fees[r['right_profile']]
   assert r['chosen_profile']==(r['left_profile'] if r['choice']=='A' else r['right_profile']) and r['elapsed_ms']>=0
  sessions.append(d)
 return sessions,test_count

def analyze(directory,out,draws):
 design=json.loads((ROOT/'trial_design.json').read_text());sessions,test_count=read_sessions(directory,design)
 if not sessions:
  out.write_text(json.dumps({'status':'awaiting_human_responses','observed_participants':0,'diagnostic_sessions_excluded':test_count,'scientific_gate_complete':False},indent=2)+'\n');print('No observed human responses. Study evidence remains pending.');return 3
 profiles=design['profiles'];blocks={}
 for block in ['equal_fee','fee_tradeoff']:
  rows=[]
  for si,d in enumerate(sessions):
   for r in d['responses']:
    if r['block']!=block:continue
    a,b=profiles[r['left_profile']],profiles[r['right_profile']]
    rows.append([si,float(r['choice']=='A'),np.log(a['display_q'])-np.log(b['display_q']),a['display_q']-b['display_q'],(a['display_profit']-b['display_profit'])/100,(r['left_fee']-r['right_fee'])/10])
  rows=np.array(rows);specs={'tullock_displayed_ratio':2,'logit_displayed_ratio':3,'logit_displayed_profit':4};fits={}
  for name,col in specs.items():
   point=fit(rows[:,col],rows[:,1],rows[:,5] if block=='fee_tradeoff' else None);rng=np.random.default_rng(20261008);bs=[]
   for _ in range(draws):
    indices=rng.integers(0,len(sessions),len(sessions));sample=np.concatenate([rows[rows[:,0]==i] for i in indices]);bs.append(fit(sample[:,col],sample[:,1],sample[:,5] if block=='fee_tradeoff' else None)['coefficients'])
   point['participant_bootstrap_95_percentile_intervals']=np.quantile(bs,[.025,.975],axis=0).tolist();fits[name]=point
  blocks[block]=fits
 out.write_text(json.dumps({'status':'pilot_exploratory_only','observed_participants':len(sessions),'diagnostic_sessions_excluded':test_count,'bootstrap_draws':draws,'blocks':blocks,'scientific_gate_complete':False,'limitations':'Hypothetical contrast-selected benchmark-profile choices; single-score competing models, participant-clustered uncertainty, bounded estimates under separation. No deployed-market choice technology, actual willingness to pay, or client welfare identified. Predictor coefficients use displayed rounded information. No confirmatory model selection or parameter transfer is performed.'},indent=2)+'\n')
 print('Pilot responses analyzed; confirmatory and market gates remain incomplete.');return 0
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--responses',type=Path,required=True);p.add_argument('--out',type=Path,default=ROOT/'human_analysis_status.json');p.add_argument('--draws',type=int,default=500);a=p.parse_args();raise SystemExit(analyze(a.responses,a.out,a.draws))
