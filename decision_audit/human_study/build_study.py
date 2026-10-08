from pathlib import Path
import json,itertools,hashlib,csv
import numpy as np
ROOT=Path(__file__).resolve().parent;archive=ROOT.parent.parent if (ROOT.parent.parent/'preserved_outputs').exists() else ROOT.parents[1]/'cpu_ready/package'
models=['deepseek_r1_llama_8b','phi4_mini_reasoning','qwen3_4b','qwen3_8b','qwen3_14b','granite_3_2_8b_instruct'];paths=list((archive/'robustness_extensions/server_outputs/raw').glob('*_reasoning_stochastic.jsonl'))
for m in ['qwen3_14b','granite_3_2_8b_instruct']:paths+=list((archive/'preserved_outputs'/m/'historical/raw').glob('*_reasoning_stochastic.jsonl'))
cells={};hashes={}
for path in sorted(paths):
 hashes[str(path.relative_to(archive))]=hashlib.sha256(path.read_bytes()).hexdigest()
 for line in path.open():
  x=json.loads(line);cells.setdefault((x['model'],x['budget']),[]).append([x['test_utility_ratio'],float(x['test_correct']),x['test_utility']])
profiles=[]
for model in models:
 for budget in [128,384,768]:
  a=np.array(cells[(model,budget)]);assert a.shape==(1080,3);q,acc,profit=a.mean(axis=0)
  profiles.append({'profile_id':f'S{len(profiles)+1:02d}','model':model,'budget':budget,'utility_ratio':float(q),'accuracy':float(acc),'mean_profit_usd':float(profit),'display_q':round(float(q),3),'display_accuracy':round(float(acc),3),'display_profit':round(float(profit),2),'n_scenarios':216})
j=json.loads((ROOT.parent/'results/audit_manifest.json').read_text());profiles.append({'profile_id':'S19','model':'calculator_train','budget':0,'utility_ratio':j['calculator_mean_ratio'],'accuracy':None,'mean_profit_usd':j['calculator_mean_profit_usd'],'display_q':round(j['calculator_mean_ratio'],3),'display_accuracy':None,'display_profit':round(j['calculator_mean_profit_usd'],2),'n_scenarios':216})
# The published exact-label calculator accuracy, rather than tie-admitting audit accuracy.
base=list(csv.DictReader((ROOT.parent/'data/archived_analytic_baselines.csv').open()));calc=[r for r in base if r['model']=='calculator_train'];acc=sum(r['test_correct']=='True' for r in calc)/len(calc);profiles[-1]['accuracy']=acc;profiles[-1]['display_accuracy']=round(acc,3)
pairs=list(itertools.combinations(profiles,2));conflict=[(a,b) for a,b in pairs if (a['display_q']-b['display_q'])*(a['display_profit']-b['display_profit'])<0];aligned=[(a,b) for a,b in pairs if (a['display_q']-b['display_q'])*(a['display_profit']-b['display_profit'])>0 and b['profile_id']!='S19'];aligned.sort(key=lambda p:abs(p[0]['display_q']-p[1]['display_q']))
chosen=conflict[:3]+[aligned[0],aligned[-1],(profiles[2],profiles[-1])];assert len(chosen)==6 and len({tuple(sorted([a['profile_id'],b['profile_id']])) for a,b in chosen})==6
trials=[]
for i,(a,b) in enumerate(chosen):trials.append({'trial_id':f'E{i+1:02d}','block':'equal_fee','profile_1':a['profile_id'],'profile_2':b['profile_id'],'fee_1':10.,'fee_2':10.,'contrast_selected':True})
for i,(a,b) in enumerate(chosen):
 high,low=sorted([a,b],key=lambda x:x['display_q'],reverse=True)
 trials.append({'trial_id':f'F{i+1:02d}','block':'fee_tradeoff','profile_1':high['profile_id'],'profile_2':low['profile_id'],'fee_1':[10,25,50,10,25,50][i],'fee_2':0.,'contrast_selected':True})
# Participant UI only receives anonymized profile identifiers and displayed statistics.
public={p['profile_id']:{k:p[k] for k in ['profile_id','display_q','display_accuracy','display_profit','n_scenarios']} for p in profiles}
(ROOT/'profiles_key.json').write_text(json.dumps(profiles,indent=2)+'\n');(ROOT/'trial_design.json').write_text(json.dumps({'version':'pilot_v1','profiles':public,'trials':trials},indent=2)+'\n');(ROOT/'build_provenance.json').write_text(json.dumps({'seed':'Participant-side seeded randomization from anonymous participant ID','source_sha256':hashes,'profile_protocol':'Three-budget stochastic frontier; 216 scenarios x five stored generation seeds, plus calculator.','design':'Six contrast-selected equal-fee pairs and six fee tradeoffs; not a representative sample of service pairs.','human_responses_collected':0},indent=2)+'\n')
html=(ROOT/'survey_template.html').read_text().replace('__STUDY_DATA__',json.dumps({'version':'pilot_v1','profiles':public,'trials':trials}))
(ROOT/'survey.html').write_text(html)
print('Study profiles',len(profiles),'trials',len(trials),'conflicting pair candidates',len(conflict))
