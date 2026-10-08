from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;PAPER=ROOT/'paper' if (ROOT/'paper').exists() else ROOT.parent/'paper';df=pd.read_csv(ROOT/'results/calculator_comparison.csv');audit=pd.read_csv(ROOT/'results/decision_error_audit.csv',dtype={'scenario_id':str});j=json.loads((ROOT/'results/audit_manifest.json').read_text());high=df[df.budget==768]
table=r'''\begin{table}[htbp]
\centering\small
\caption{Calculator and observable-error audit of the greedy 768-token frontier. Profit differences are model minus construction-split calculator, with exploratory simultaneous 95\% family bands over six models. Displayed-optimal actions maximize profit under the four-decimal probabilities actually shown; held-out optimality admits payoff ties and differs from exact-label accuracy. All 216 scenarios and three option orders are retained.}
\label{tab:calculatorerroraudit}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrr}
\toprule
Model & Profit difference, USD [band] & Displayed-optimal (\%) & Both suboptimal (\%) \\
\midrule
'''
for r in high.itertuples():table+=f'{r.label} & {r.mean_profit_gap_vs_calculator:.2f} [{r.simultaneous_profit_gap_low:.2f}, {r.simultaneous_profit_gap_high:.2f}] & {100*r.displayed_optimal_rate:.1f} & {100*r.both_suboptimal_rate:.1f} \\\\\n'
table+=r'''\bottomrule
\end{tabular}
\end{table}
''';(PAPER/'tables/calculator_error_audit.tex').write_text(table)
sub=[]
for (model,cat,b),g in audit.groupby(['model','category','budget']):
 if b not in [0,768]:continue
 sub.append({'model':model,'category':cat,'budget':int(b),'mean_profit_usd':float(g.test_utility.mean()),'mean_gap_vs_calculator_usd':float(g.profit_gap_vs_calculator.mean()),'displayed_optimal_rate':float(g.displayed_optimal.mean()),'mean_displayed_objective_loss_usd':float(g.displayed_objective_regret.mean())})
pd.DataFrame(sub).to_csv(ROOT/'results/product_endpoint_audit.csv',index=False)
concentration=[]
for model,g in audit[audit.budget==768].groupby('model'):
 scenario=(g.test_optimal_utility-g.test_utility).groupby(g.scenario_id).mean().sort_values(ascending=False);concentration.append({'model':model,'top_22_of_216_scenarios_regret_share':float(scenario.iloc[:22].sum()/scenario.sum())})
pd.DataFrame(concentration).to_csv(ROOT/'results/regret_concentration.csv',index=False)
lines=['# Saved-decision and calculator audit','', '8 October 2026. No new inference.','', '## Main finding','', f"The construction-split calculator earns mean expected bidder profit of ${j['calculator_mean_profit_usd']:.2f}, with mean held-out regret ${j['calculator_mean_regret_usd']:.2f}. Every greedy 768-token model remains below this profit baseline under the six-model simultaneous exploratory bands. Additional tokens improve LLM performance relative to zero; they do not establish a practical advantage over the numerical policy.",'', '| Model | LLM minus calculator, USD | Simultaneous 95% band | Displayed-optimal actions |','|---|---:|---|---:|']
for r in high.itertuples():lines.append(f'| {r.label} | {r.mean_profit_gap_vs_calculator:.2f} | [{r.simultaneous_profit_gap_low:.2f}, {r.simultaneous_profit_gap_high:.2f}] | {100*r.displayed_optimal_rate:.1f}% |')
lines+=['', '## What errors can and cannot identify','', 'Actions are classified by whether they maximize expected profit under the four-decimal probabilities shown and under held-out scoring. The four classes partition every record. They measure observable objective agreement, not internal arithmetic competence, hallucination or a causal reasoning mechanism. Held-out utility optimality permits equally profitable bids, unlike the original exact-label accuracy statistic.','', f"The full-precision calculator is held-out utility-optimal in {j['calculator_heldout_optimal_scenarios']}/216 scenarios. Thus an action can be optimal for the supplied distribution yet lose held-out profit. At 768 tokens, displayed-optimal/held-out-suboptimal choices comprise 9.4–22.8% of model records; both-suboptimal choices comprise 11.9–52.0%. The displayed precision changes the full-precision calculator's tie-broken action in one scenario; the original baseline is preserved. The display-matched calculator loses only $0.0113 in mean held-out profit, and all six model deficits remain negative under separately recomputed simultaneous bands against it.",'', 'The exact decomposition is: model held-out regret = calculator held-out regret + (calculator held-out profit − model held-out profit). The second term is signed: a model can sometimes beat the calculator under the scoring distribution despite being worse under the displayed objective. Such events occur in 2.2–8.0% of 768-token records; they do not overturn the aggregate disadvantage.','', 'Product-specific endpoint means and regret concentration are supplementary descriptive diagnostics. They use the constructed tasks and do not identify real-market task frequencies. The largest mean dollar-profit model need not be the largest mean utility-ratio model because these objectives weight task scales differently.','', '## Uncertainty and limits','', '10,000 paired within-product value-family draws (seed 20261009) retain four candidate variants and all budgets, after averaging the three deterministic option orders within each scenario. The six 768-token dollar-profit gaps receive simultaneous maximum-standardized-deviation bands. This conditions on original prompts, products, construction and scoring CDFs. No independent confirmation, market generalization or stochastic-generation uncertainty is claimed for this greedy audit.','', 'Calculator elapsed-time/cost zeros in archived analytic rows are placeholders, not measurements. Therefore this is a gross expected-profit comparison, not a measured net-payoff or deployment cost-effectiveness ranking.','', '## Human study status','', 'The accompanying human_study directory contains proposed participant materials and an offline survey. Human responses have not been collected. Hypothetical benchmark-profile choice can test responses to the displayed quality and fee information; it cannot alone identify deployed-market selection, actual willingness to pay, or realized client surplus.']
(ROOT/'results_report.md').write_text('\n'.join(lines)+'\n')
