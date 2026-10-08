from pathlib import Path
import csv,json
root=Path(__file__).resolve().parents[1];out=root/'cpu_revision/results';paper=root/'paper'
r=list(csv.DictReader((out/'comparison_uncertainty.csv').open()));maps=list(csv.DictReader((out/'mapping_sensitivity.csv').open()));j=json.loads((out/'analysis_manifest.json').read_text())
table=r'''\begin{table}[htbp]
\centering\small
\caption{Metric and shape sensitivity with paired uncertainty at $\gamma=1$. The threshold supports the 768/768 pure equilibrium. Accuracy/utility and linear change are point estimates. The accuracy-minus-utility threshold difference has exploratory simultaneous 95\% bands over six models, in units of $10^{-4}$; all bands exclude zero. The bootstrap uses 10,000 product-stratified family/seed draws, retaining all draws.}
\label{tab:metricshapecomparison}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}
\toprule
Model & Utility threshold & Accuracy/utility & Difference [simult. band] & Linear change (\%) \\
\midrule
'''
for x in r:
 t=float(x['utility_threshold']); exponent=-5
 table+=f"{x['label']} & ${t/1e-5:.2f}\\times10^{{-5}}$ & {float(x['accuracy_ratio']):.2f} & {float(x['threshold_difference'])*1e4:.2f} [{float(x['simultaneous_difference_low'])*1e4:.2f}, {float(x['simultaneous_difference_high'])*1e4:.2f}] & {float(x['linear_percent_change']):+.2f} \\\\\n"
table+=r'''\bottomrule
\end{tabular}
\end{table}
'''
(paper/'tables/metric_shape_comparison.tex').write_text(table)
table=r'''\begin{table}[htbp]
\centering\small
\caption{Alternative selection assumptions on the same stochastic menu. Profit quality is mean expected bidder profit divided by mean oracle profit. Its signed high/high indifference bound has pointwise 95\% family/seed intervals, in units of $10^{-4}$; a nonpositive bound gives no high/high equilibrium at positive cost. Unsupported-draw percentages retain every draw. The matched logit uses fixed noise scale $s=\overline q_m$, matching the Tullock $\gamma=1$ local slope at equal reference quality. Ratios are point estimates; these alternatives are not fitted client behavior.}
\label{tab:selectionmappings}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lrrrr}
\toprule
Model & Profit/utility & Profit bound [95\% CI] & Unsupported (\%) & Logit/utility \\
\midrule
'''
for x in r:
 p=next(a for a in maps if a['model']==x['model'] and a['mapping']=='profit');l=next(a for a in maps if a['model']==x['model'] and a['mapping']=='logit_matched')
 table+=f"{x['label']} & {float(p['relative_to_utility']):.2f} & {float(p['threshold'])*1e4:.2f} [{float(p['threshold_ci_low'])*1e4:.2f}, {float(p['threshold_ci_high'])*1e4:.2f}] & {100*int(p['nonpositive_threshold_draws'])/j['draws']:.2f} & {float(l['relative_to_utility']):.2f} \\\\\n"
table+=r'''\bottomrule
\end{tabular}
\end{table}
'''
(paper/'tables/selection_mappings.tex').write_text(table)
# Standalone readable evidence report, kept separate from manuscript claims.
report=['# Completed CPU extension results','', '8 October 2026. No new inference.','', '## Paired uncertainty','', '10,000 within-product value-family/seed draws; all four candidate variants and paired budget/metric vectors are retained. Generation seeds are sampled independently by model, jointly across budgets within each selected scenario occurrence. Products, prompts and scoring CDFs stay fixed.','', '| Model | Accuracy minus utility threshold | Simultaneous 95% band |','|---|---:|---|']
for x in r:report.append(f"| {x['label']} | {float(x['threshold_difference']):.6g} | [{float(x['simultaneous_difference_low']):.6g}, {float(x['simultaneous_difference_high']):.6g}] |")
report+=['','All six exploratory simultaneous bands exclude zero. The DeepSeek strict-dominance contrast at kappa=0.00015 survives all 10,000 draws. This is a bootstrap stability frequency, not a market probability or independent confirmation.','', 'Twelve Phi utility draws have a nonpositive signed high/high bound. Absolute difference intervals retain those draws. Ratio intervals in the supplementary CSV condition on positive utility bounds and are labelled accordingly. They are not used for the primary uncertainty claim.','', 'Only DeepSeek and Qwen3-14B have linear-minus-measured threshold intervals excluding zero. Four intervals include zero; Qwen3-4B and Granite show substantial upper-tail shape sensitivity. The small point changes do not justify a general claim that intermediate shape is unimportant.','', '## Alternative selection and client-value assumptions','', 'Profit-based selection changes point private thresholds to 0.24–1.85 times the utility thresholds. High/high support is absent in 24.71% of Phi draws, 36.68% of Qwen3-4B draws, 9.42% of Qwen3-8B draws and 0.06% of Granite draws. DeepSeek and Qwen3-14B retain support in every draw.','', 'A quality-difference logit at matched reference slope changes point thresholds by +1.40% to +4.28%. Holding its noise scale fixed makes selection invariant to common score shifts. Doubling or halving discrimination changes point thresholds to approximately 2.01–2.07 or 0.51–0.52 times the utility thresholds. The functional form and discrimination remain unestimated.','', 'See service_model_specification.md for explicit service accounting, client-value units and assumptions. Neither model identifies real client choice, auction total surplus, service payment, richer task competence or unconstrained budget agreement.','', '## Remaining work','', 'CPU uncertainty, explicit alternative mappings, manuscript integration and portable packaging are completed in this bundle. Venue-specific format, author/anonymity information and checklist requirements await the target venue. Richer tasks and matched above-ceiling evidence need new inference; market mappings need external evidence.']
(root/'cpu_revision/results_report.md').write_text('\n'.join(report)+'\n')
