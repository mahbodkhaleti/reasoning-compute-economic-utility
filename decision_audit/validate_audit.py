from pathlib import Path
import json,importlib.util,tempfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
r=pd.read_csv(ROOT/'results/decision_error_audit.csv',dtype={'scenario_id':str});r['scenario_id']=r.scenario_id.str.zfill(12);assert len(r)==23328
assert set(r.error_class)=={'both_optimal','display_optimal_heldout_suboptimal','display_suboptimal_heldout_optimal','both_suboptimal'}
assert np.allclose(r.test_optimal_utility-r.test_utility,r.calculator_regret-r.profit_gap_vs_calculator,atol=1e-9)
assert np.all(r.displayed_objective_regret>=-1e-9)
s=pd.read_csv(ROOT/'results/calculator_comparison.csv');assert len(s)==36
for _,x in s.iterrows():assert abs(sum(x[c+'_rate'] for c in ['both_optimal','display_optimal_heldout_suboptimal','display_suboptimal_heldout_optimal','both_suboptimal'])-1)<1e-10
j=json.loads((ROOT/'results/audit_manifest.json').read_text());assert j['all_six_simultaneous_profit_gap_upper_bounds_negative'] and j['all_six_display_calculator_simultaneous_gap_upper_bounds_negative']
# An independent menu oracle agrees with the displayed objective classification.
sc=pd.read_csv(ROOT/'data/scenarios.csv',dtype={'scenario_id':str});lookup={x['scenario_id'].zfill(12):x for x in sc.to_dict('records')}
for x in r.query('budget==768').itertuples():
 menu=json.loads(lookup[x.scenario_id]['candidates_json']);profits=[(x.value-o['bid'])*float(format(o['train_p'],'.4f')) for o in menu];chosen=next(i for i,o in enumerate(menu) if o['bid']==x.chosen_bid)
 assert bool(x.displayed_optimal)==(max(profits)-profits[chosen]<1e-9)
# Missing/diagnostic human data must not become a completed evidence gate.
p=ROOT/'human_study/analyze_human_choices.py';spec=importlib.util.spec_from_file_location('human_checks',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with tempfile.TemporaryDirectory() as t:
 temp=Path(t);(temp/'choice_TEST_FLAG.json').write_text(json.dumps({'participant_id':'TEST_FLAG','is_test':True,'responses':[]}));code=module.analyze(temp,temp/'result.json',5);status=json.loads((temp/'result.json').read_text());assert code==3 and status['observed_participants']==0 and status['diagnostic_sessions_excluded']==1 and not status['scientific_gate_complete']
(ROOT/'results/validation.json').write_text(json.dumps({'record_and_design_counts':'passed','regret_identity':'passed','displayed_menu_oracle':'passed','two_calculator_baselines':'passed','error_categories_partition':'passed','human_diagnostic_exclusion':'passed','human_observed_responses':0,'human_evidence_gate':'pending','survey_browser_visual_check':'not performed: local-file URL policy blocked preview','survey_static_flow_check':'passed separately with mock document'},indent=2)+'\n')
print('Saved-decision and human-data safeguards passed.')
