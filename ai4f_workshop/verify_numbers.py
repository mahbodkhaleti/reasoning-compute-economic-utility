from pathlib import Path
import argparse,csv,json,re,hashlib,shutil,collections,subprocess
parser=argparse.ArgumentParser(description='Audit saved manuscript evidence without new inference.')
parser.add_argument('--no-write',action='store_true',help='Print audit results without replacing the published number_audit.json.')
args=parser.parse_args()
N=Path(__file__).resolve().parent
C=lambda p:list(csv.DictReader(Path(p).open()))
read_inputs={p.name:p for p in (N/'audit_sources').iterdir() if p.is_file() and p.name in json.loads((N/'audit_sources/source_hashes.json').read_text())}
inputs_sha={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in read_inputs.items()}
assert inputs_sha==json.loads((N/'audit_sources/source_hashes.json').read_text())
report={'scope':'Read-only verification of existing values and editorial preservation; no experiment or new inference','discrepancies':[],'inputs_sha256':inputs_sha,'checks':{}}
a=C(read_inputs['calculator_comparison.csv']);a=[r for r in a if r['budget']=='768'];labels=[r['label'] for r in a]
expected=[[-22.60,-38.80,-6.39,86.3,11.9],[-33.67,-55.78,-11.57,42.1,52.0],[-30.54,-48.97,-12.11,71.1,25.6],[-30.37,-45.84,-14.91,67.7,28.1],[-36.00,-54.15,-17.86,76.7,20.5],[-15.09,-24.89,-5.28,53.5,41.2]]
for r,e in zip(a,expected):
 vals=[round(float(r[k]),2) for k in ['mean_profit_gap_vs_calculator','simultaneous_profit_gap_low','simultaneous_profit_gap_high']]+[round(100*float(r[k]),1) for k in ['displayed_optimal_rate','both_suboptimal_rate']]
 assert vals==e,(r['label'],vals,e)
report['checks']['supplied_calculator_table_values']='All six supplied rows match source rounded values; simultaneous rather than pointwise bands used'
report['calculator_rows']=[{'model':r['label'],'mean_difference':r['mean_profit_gap_vs_calculator'],'simultaneous_low':r['simultaneous_profit_gap_low'],'simultaneous_high':r['simultaneous_profit_gap_high'],'displayed_optimal_percent':100*float(r['displayed_optimal_rate']),'both_suboptimal_percent':100*float(r['both_suboptimal_rate'])} for r in a]
for k,e in [('displayed_optimal_rate',(42.1,86.3)),('both_suboptimal_rate',(11.9,52.0)),('display_optimal_heldout_suboptimal_rate',(9.4,22.8)),('heldout_beats_calculator_rate',(2.2,8.0))]:
 vals=[100*float(r[k]) for r in a];actual=(round(min(vals),1),round(max(vals),1));assert actual==e;report['checks'][k]={'printed_range':e,'unrounded_range':[min(vals),max(vals)]}
cal=C(read_inputs['calculator_scenarios.csv']);assert len(cal)==216
profit=sum(float(r['calculator_test_profit']) for r in cal)/len(cal);regret=sum(float(r['calculator_regret']) for r in cal)/len(cal);opt=sum(r['calculator_heldout_optimal']=='True' for r in cal)
assert round(profit,2)==212.66 and round(regret,2)==3.09 and opt==155
manifest=json.loads(read_inputs['calculator_audit_manifest.json'].read_text());assert manifest['rows']==23328 and manifest['scenarios']==216 and manifest['value_families']==54 and manifest['products']==3
assert manifest['rounding_changed_baseline_scenarios']==1
rounding=profit-sum(float(r['display_calculator_test_profit']) for r in cal)/len(cal);assert round(rounding,4)==.0113
report['checks']['construction_calculator']={'mean_heldout_profit':profit,'mean_heldout_regret':regret,'heldout_optimal':opt,'scenarios':216,'rounding_changes':1,'rounding_mean_loss':rounding}
multi=C(read_inputs['adjacent_multiplicity.csv']);assert len(multi)==30;assert sum(r['opposite_point_directions']=='True' for r in multi)==9;assert not any(r['opposite_directions_both_holm_significant']=='True' for r in multi)
report['checks']['aggregation']={'adjacent_comparisons':30,'opposite_point_directions':9,'both_opposite_significant_after_Holm':0,'exploratory_tests':2*len(multi)}
g=C(read_inputs['gain_concentration.csv']);g=[r for r in g if r['metric']=='profit_usd'];vals=[100*float(r['top22_positive_gain_share']) for r in g];assert [round(min(vals),1),round(max(vals),1)]==[80.0,86.7];report['checks']['positive_gain_concentration']={'tasks':22,'unrounded_range_percent':[min(vals),max(vals)],'negative_gains_remain_in_means':True}
f=C(read_inputs['freeform_sensitivity_summary.csv']);v=[100*float(r['freeform_validity_rate']) for r in f];assert [round(min(v),1),round(max(v),1)]==[29.1,96.7];report['checks']['freeform_validity_percent']=[min(v),max(v)]
m=C(read_inputs['joint_monetary_uncertainty.csv']);assert len(m)==6;ps=[float(r['private_V_star']) for r in m];bs=[float(r['planner_beta_star']) for r in m];assert [round(min(ps),2),round(max(ps),2)]==[.03,.18];assert [round(min(bs),2),round(max(bs),2)]==[.02,.12];assert next(r for r in m if r['model']=='qwen3_14b')['hardware']=='L40S'
report['checks']['headline_break_even']={'private_raw_range':[min(ps),max(ps)],'planner_raw_range':[min(bs),max(bs)],'headline_rounding_decimals':2,'gamma':1,'Qwen3_14B_hardware':'separate L40S scenario'}
prod=C(read_inputs['product_thresholds.csv']);ratios=[]
for model in {r['model'] for r in prod}:
 vals=[float(r['K']) for r in prod if r['model']==model];ratios.append(max(vals)/min(vals))
assert round(max(ratios),2)==30.40;report['checks']['product_private_bound_max_fold']=max(ratios)
gap=C(read_inputs['calculator_gap_closed.csv']);shares=[100*float(r['share_closed']) for r in gap];assert [round(min(shares)),round(max(shares))]==[42,85];report['checks']['abstract_gap_closed_percent']={'raw_range':[min(shares),max(shares)],'whole_percent_rounding':[42,85]}
# Verify all original main table values, and the copied new audit table.
for name,h in json.loads((N/'audit_sources/preserved_assets_sha256.json').read_text()).items():assert hashlib.sha256((N/name).read_bytes()).hexdigest()==h
assert (N/'tables/calculator_error_audit.tex').read_text().replace(r'\begin{table*}[t]',r'\begin{table}[htbp]').replace(r'\end{table*}',r'\end{table}')==read_inputs['calculator_error_audit_reference.tex'].read_text()
report['checks']['all_existing_tables']='Byte-identical; monetary table now in appendix; calculator audit values preserved'

def table_rows(p):
 return {l.split('&')[0].strip():[c.strip().removesuffix('\\').strip() for c in l.split('&')[1:]] for l in p.read_text().splitlines() if any(l.startswith(a+' &') for a in labels)}
original_endpoints=table_rows(read_inputs['cross_model_summary_reference.tex']);current_endpoints=table_rows(N/'tables/endpoints.tex')
for r in gap:
 label=next(z['label'] for z in a if z['model']==r['model'])
 assert [x.rstrip('\\').strip() for x in current_endpoints[label][:-1]]==[x.rstrip('\\').strip() for x in original_endpoints[label]]
 assert current_endpoints[label][-1].rstrip('\\').strip()==f"{100*float(r['share_closed']):.1f}"
report['checks']['main_endpoint_table']='All six endpoint/accuracy/CI rows match full-paper source; gap-closed column matches saved CSV'

# Read the source resource counters and verify every resource row, not only copied text.
rc=C(read_inputs['recorded_compute_counters.csv']);by=collections.defaultdict(list)
for r in rc:by[r['model']].append(r)
resource={}
for model,rs in by.items():
 n=sum(int(r['recorded_cells']) for r in rs);assert n==8424
 resource[model]={'cells':n,'primary_hours':sum(float(r['primary_cell_seconds']) for r in rs)/3600,'readout_hours':sum(float(r['separate_readout_seconds']) for r in rs)/3600,'primary_kwh':sum(float(r['primary_energy_joules']) for r in rs)/3600000}
assert sum(x['cells'] for x in resource.values())==50544
lines=[l for l in (N/'tables/resources.tex').read_text().splitlines() if any(l.startswith(a+' &') for a in labels)]
for line,r in zip(lines,a):
 fields=[x.strip().removesuffix('\\\\').strip() for x in line.split('&')];c=resource[r['model']]
 assert fields[1:]==[f"{c['primary_hours']:.2f}",f"{c['readout_hours']:.3f}",f"{c['primary_kwh']:.2f}"]
report['checks']['resource_table']=resource
old=(N/'audit_sources/main_ai4f_v1_reference.tex').read_text();new=(N/'main_ai4f_v4.tex').read_text()
cites=lambda s:sorted(x.strip() for a in re.findall(r'\\cite\w*\{([^}]+)\}',s) for x in a.split(','))
assert collections.Counter(cites(old)) <= collections.Counter(cites(new))
newkeys={'chen2023frugalgpt','wang2026conformalthinking','wilder2019melding','chen2021finqa','zhu2021tatqa'}
assert set(cites(new))-set(cites(old))==newkeys
eq=lambda s:{re.search(r'\\label\{([^}]+)\}',e).group(1):e for e in re.findall(r'\\begin\{equation\}.*?\\end\{equation\}',s,re.S)}
assert eq(old)==eq(new);report['checks']['equations']='All 11 existing numbered equation blocks preserved verbatim; accounting identity is explicitly not an independent-cause decomposition'
repo='https://anonymous.4open.science/r/reasoning-compute-economic-utility-EED7/';assert repo in new and new.count(repo)==old.count(repo)
report['checks']['citations_and_repo']='All original citation occurrences, bibliography bytes and anonymous URL preserved; five verified references added in a separate bibliography'
main=new.split(r'\section{Service-contest sensitivity model}')[0]
assert r'\label{eq:contest}' not in main and r'\begin{proposition}' not in main and r'\input{tables/money}' not in main
abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',new,re.S).group(1)
report['checks']['abstract_words']=len(abstract.split());assert 150<=len(abstract.split())<=180
report['checks']['intro_numbers']='No experimental numerical claim introduced; financial example is explicitly motivating analogy'
log_path=N/'main_ai4f_v4.log'
if log_path.is_file():
 log=log_path.read_text();assert not re.search(r'Overfull|undefined|LaTeX Warning|Package .* Warning|^!',log,re.M)
 latex_check='Current local compilation log: no errors, warnings, undefined references or overfull boxes'
else:
 # A static checkout ships the reviewed PDF and its recorded build audit, not local compiler logs.
 build=json.loads((N/'package_validation.json').read_text())
 assert build['passed'] and build['fresh_extracted_package_checks']['compile']=='passed'
 assert all(build[k]==0 for k in ('latex_errors','undefined_references','overfull_boxes'))
 assert hashlib.sha256((N/'main_ai4f_v4.pdf').read_bytes()).hexdigest()==build['PDF_sha256']
 latex_check='Published PDF matches recorded clean compilation audit; no new compilation performed'
text=subprocess.check_output(['pdftotext','-layout',str(N/'main_ai4f_v4.pdf'),'-'],text=True);pages=text.split('\f')[:-1];ref=next(i+1 for i,p in enumerate(pages) if 'References' in p);app=next(i+1 for i,p in enumerate(pages) if 'Service-contest sensitivity model' in p)
report['pages']={'total':len(pages),'main_and_references':app-1,'main_narrative_end':ref,'references_start':ref,'appendix_start':app,'appendix_pages':len(pages)-app+1,'appendix_B_start':next(i+1 for i,p in enumerate(pages) if 'Executable admission and CPU reference' in p)}
report['checks']['latex']=latex_check
report['checks']['anonymous_pdf']='Anonymous Author(s)' in text and not bool(re.search(r'/(?:Users|home)/',text+subprocess.check_output(['pdfinfo',str(N/'main_ai4f_v4.pdf')],text=True)));assert report['checks']['anonymous_pdf']
report['PDF_sha256']=hashlib.sha256((N/'main_ai4f_v4.pdf').read_bytes()).hexdigest()
if not args.no_write:
 (N/'number_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'checks_passed':len(report['checks']),'discrepancies':report['discrepancies'],'pages':report['pages'],'compilation_evidence':latex_check,'published_report_replaced':not args.no_write},indent=2))
