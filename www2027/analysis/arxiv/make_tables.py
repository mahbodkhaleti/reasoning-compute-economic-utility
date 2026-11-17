"""Write every table of the arXiv version as LaTeX from saved results files (no number typed by hand).
Output: paper_arxiv/tables/*.tex and results/arxiv/facts.json (key ranges quoted in the text)."""
import json, os, numpy as np
R = "/tmp/www2027/results"; T = "/tmp/www2027/results/arxiv/tables"; os.makedirs(T, exist_ok=True)
J = lambda p: json.load(open(f"{R}/{p}"))
M = ["deepseek_r1_llama_8b", "phi4_mini_reasoning", "qwen3_4b", "qwen3_8b", "qwen3_14b", "granite_3_2_8b_instruct"]
LAB = {"deepseek_r1_llama_8b": "DeepSeek-8B", "phi4_mini_reasoning": "Phi-mini", "qwen3_4b": "Qwen3-4B", "qwen3_8b": "Qwen3-8B", "qwen3_14b": "Qwen3-14B", "granite_3_2_8b_instruct": "Granite-8B"}
CATS = ["not_reached", "arith", "incomplete", "passed"]
P = lambda x: (lambda v: f"$-${-v}" if v < 0 else f"{v}")(int(round(100 * x)) + 0)
def CI(d):
    if d is None: return "--"
    if isinstance(d, list): d = d[0]
    pt = d["pt"]; lo, hi = d["ci"] if "ci" in d else (d["lo"], d["hi"])
    return f"{P(pt)} [{P(lo)}, {P(hi)}]"
def ORf(d): return f"{d['OR']:.2f} [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}]"
def table(name, cols, header, rows, caption, label, star=False, size="\\small", sep=None, pos="t"):
    env = "table*" if star else "table"
    s = [f"\\begin{{{env}}}[{pos}]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size, "\\centering",
         (f"\\setlength{{\\tabcolsep}}{{{sep}}}" if sep else "") , f"\\begin{{tabular}}{{{cols}}}", "\\toprule", header + " \\\\", "\\midrule"] + [r + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    open(f"{T}/{name}.tex", "w").write("\n".join(s) + "\n")
F = {}
EB = J("arxiv/ebay_budget.json"); R1S = J("R1/r1_summary.json"); RUN1 = J("followup/run1_summary.json"); PHI4 = J("followup/run1_phi4096_summary.json")
R3 = J("rev25/r3_stated_answer.json"); FP = J("rev25/r1_firstprice.json"); AN0 = J("AN/an_p0_summary.json"); ANO = J("AN/an_overnight_summary.json")
A9 = json.load(open("/tmp/www2027/analysis/revision/out/t9_judge6.json"))["A3"]
B8 = json.load(open("/tmp/www2027/analysis/revision/out/t10_judge7.json"))["B8"]
T10B = json.load(open("/tmp/www2027/analysis/revision/out/t10b_hybrid_split.json"))["forced768"]
fix = lambda d: dict(pt=d["share"], ci=d["ci"])
F1 = json.load(open("/tmp/www2027/analysis/revision/out/t8_fixes.json"))["F1"]
natgap = lambda m: dict(pt=F1[m]["natural_all"]["gap_readout"], ci=F1[m]["natural_all"]["gap_readout_ci"])
RUN1.update(PHI4)

# ---- T: utility ratio by budget (eBay)
B = ["0", "64", "128", "256", "512", "768", "-1"]
rows = []
for m in M:
    e = EB[m]["by_budget"]; nat = RUN1.get(m, {}).get("q_natural")
    rows.append(LAB[m] + " & " + " & ".join(f"{e[b]['q']:.3f}" for b in B) + " & " + (f"{nat['pt']:.3f}" if nat else "--"))
table("tab_q_budget", "lcccccccc", "Model & $q(0)$ & $q(64)$ & $q(128)$ & $q(256)$ & $q(512)$ & $q(768)$ & 1,024 run & 4,096 run", rows,
      "Held-out utility ratio $q$ on eBay by forced reasoning budget, and without forcing under a 1,024-token and a 4,096-token cap. Calculator: 0.981; random choice: 0.711; 648 decisions per cell.",
      "tab:qbudget", star=True)

# ---- T: gap recovered, main
rows = []
for m in M:
    e = EB[m]["by_budget"]; st = R3.get(f"{m}|768", {}).get("gap_recovered_stated")
    fp = FP["models"][m]["gap768"]
    an = AN0["auctionnet"][m]["gap768"]; anc = ANO[m]["gap_closed"]; an4 = ANO[m].get("gap_unf4096")
    rows.append(" & ".join([LAB[m], CI(e["256"]["gap"]), CI(e["512"]["gap"]), CI(B8[m]["answer"]), CI(st) if st else "--",
                            CI(natgap(m)), CI(RUN1[m]["gap_recovered_natural"]) if m in RUN1 else "--", CI(fp)]))
table("tab_gap_ebay", "lccccccc", "Model & 256 & 512 & 768 & 768, stated & 1,024 run & 4,096 run & 768, FP eq.", rows,
      "Gap to the calculator recovered on eBay, \\% [95\\% family-bootstrap interval]. Columns 256--768: forced budgets with the paper's letter readout. Stated: the answer the trace states, where it states one, and the letter otherwise. 1,024 and 4,096 run: no forcing, letter read after the model stops. FP eq.: 768 tokens scored against the first-price equilibrium win curve.",
      "tab:gapebay", star=True, size="\\footnotesize", sep="4pt")

# ---- T: readout ablation (eBay + AuctionNet)
DO = J("R1/downstream_open.json"); DC = J("R1/downstream_closed.json")
import importlib.util
rows = []
for m in M:
    r = R1S[m]; a = ANO[m]
    rows.append(" & ".join([LAB[m], P(r["reproduces_saved"]), P(r["agree_stated_orig"]), P(r["agree_stated_alt"]),
                            CI(r["gap_orig"]), CI(r["gap_alt"]) if "gap_alt" in r else "--", CI(r["gap_diff"]) if "gap_diff" in r else "--",
                            CI(a["gap_open"]), CI(a["gap_closed"]), CI(a["gap_diff"])]))
table("tab_readout", "lccccccccc", "& \\multicolumn{6}{c}{eBay} & \\multicolumn{3}{c}{AuctionNet} \\\\ \\cmidrule(lr){2-7}\\cmidrule(lr){8-10} Model & Repro. & Agree, open & Agree, closed & Open & Closed & Change & Open & Closed & Change", rows,
      "Closing the reasoning block before reading the answer, at 768 forced tokens. Repro.: share of the saved letters that the rerun reproduces before the change (\\%). Agree: share of decisions where the letter equals the answer the trace states (\\%). Open, Closed, Change: gap recovered with the paper's readout (rerun letters), with the closed-block readout, and their difference, \\% [95\\% interval].",
      "tab:readout", star=True, size="\\scriptsize", sep="3pt")

# ---- T: where the gap goes (eBay: open, closed, unforced 1,024; reader-only upper bound on arithmetic)
R2B = J("rev25/r2b_reader_only_split.json"); C1 = J("extra/C1_forcing_confound.json")
cell = lambda d: CI(d)
rows = []
for m in M:
    a = A9[m]
    rows.append(" & ".join([LAB[m], P(EB[m]["by_budget"]["768"]["finished_matcher"]), f"{a['K5']['median']}/{a['K7']['median']}"] + [cell(fix(T10B[m][k])) for k in CATS] + [P(R2B[m]["reader_only"]["arith"]["pt"] if isinstance(R2B[m]["reader_only"]["arith"], dict) else R2B[m]["reader_only"]["arith"])]))
table("tab_split_ebay", "lcccccc c", "Model & Finished & Median & Not reached & Arithmetic & Cut off & Passed over & Arith., upper", rows,
      "Where the eBay 768-token gap goes, \\% of the gap [95\\% interval], letters read with the block open; rows sum to 100 up to rounding, and intervals can dip below zero because a wrong answer can beat the calculator on held-out data. Finished: forced 768-token traces in which the matcher finds every option's value. Median: tokens by which half of unforced traces finish ($K=5$/$K=7$). Arithmetic uses the lower bound (matcher value where found); Arith., upper counts every misread value as miscomputed.",
      "tab:splitebay", star=True, size="\\footnotesize")
rows = []
for m in M:
    rows.append(" & ".join([LAB[m]] + [cell(DC["models"][m]["split"][k]) for k in CATS] + [P(C1[m]["unforced_split"][k]["pt"] if isinstance(C1[m]["unforced_split"][k], dict) else C1[m]["unforced_split"][k]) for k in CATS]))
table("tab_split_ebay2", "lcccc cccc", "& \\multicolumn{4}{c}{768 tokens, closed block} & \\multicolumn{4}{c}{No forcing, 1,024 cap} \\\\ \\cmidrule(lr){2-5}\\cmidrule(lr){6-9} Model & NR & Ar & Cut & Pass & NR & Ar & Cut & Pass", rows,
      "The eBay gap split under the closed-block readout (\\% [95\\% interval]) and for the unforced 1,024-token run (\\%, point estimates). NR: not reached; Ar: arithmetic (lower bound); Cut: cut off; Pass: passed over.",
      "tab:splitebay2", star=True, size="\\footnotesize")

# ---- AuctionNet main table
SPO = J("AN/an_split.json"); SPC = J("AN/an_split_closed.json"); RA = J("AN/an_regression_audit.json")
def cis(d, j): return f"{P(d['pt'][j])} [{P(d['lo'][j])}, {P(d['hi'][j])}]"
rows = []
for m in M:
    a = AN0["auctionnet"][m]; o = ANO[m]
    rows.append(" & ".join([LAB[m], f"{a['q']['0']:.3f}", f"{a['q']['768']:.3f}", CI(a["gap768"]), CI(o["gap_closed"]), CI(a["gap_unf1024"]), CI(o["gap_unf4096"]) if "gap_unf4096" in o else "--", P(a["gap768_dollar"])]))
table("tab_an_main", "lccccccc", "Model & $q(0)$ & $q(768)$ & 768 & 768, closed & 1,024 run & 4,096 run & 768, dollars", rows,
      f"AuctionNet: utility ratio and gap recovered, \\% [95\\% interval; families resampled within category]. Calculator $q$ = {AN0['auctionnet']['qwen3_4b']['q_calc']:.3f}; random choice {AN0['auctionnet']['qwen3_4b']['q_rand']:.3f}. Dollars: pooled over decisions instead of averaged per decision.",
      "tab:anmain", star=True, size="\\footnotesize")
rows = []
for m in M:
    so = SPO[f"{m}|768"]["split"]; sc = SPC[f"{m}|768"]["split"]
    rows.append(" & ".join([LAB[m], P(SPO[f"{m}|768"]["finished"])] + [cis(so, j) for j in range(4)] + [P(sc["pt"][j]) for j in range(4)]))
table("tab_an_split", "lc cccc cccc", "& & \\multicolumn{4}{c}{Open block} & \\multicolumn{4}{c}{Closed block} \\\\ \\cmidrule(lr){3-6}\\cmidrule(lr){7-10} Model & Fin. & NR & Ar & Cut & Pass & NR & Ar & Cut & Pass", rows,
      "Where the AuctionNet 768-token gap goes, \\% of the gap. Open block: [95\\% interval]; closed block: point estimates. Fin.: forced traces that evaluate every option. Abbreviations as in Table~\\ref{tab:splitebay2}.",
      "tab:ansplit", star=True, size="\\footnotesize")

# ---- wrong bids: highest-bid share, sampling, surplus
EC = J("followup/econ_surplus_pct.json"); SA = J("followup/sampled_highest.json")
rows = []
for m in M:
    o = DO["models"][m]; c = DC["models"][m]; e = EC[m]
    rows.append(" & ".join([LAB[m], str(o["n_wrong"]), P(o["highest_share_of_wrong"]), P(o["highest_chance"]), P(c["highest_share_of_wrong"]), P(SA[f"stochastic|{m}"]["highest_share"]),
                            P(AN0["auctionnet"][m]["highest_share_wrong768"]), P(e["loss_pct"]), f"{e['loss_per_1000']/1000:.1f}", P(e["share_loss_from_highest"])]))
table("tab_wrong", "lccccccccc", "Model & Wrong & Highest & Chance & Closed & Sampled & AuctionNet & Loss & \\$k/1{,}000 & From highest", rows,
      "Wrong answers at 768 tokens and their cost on eBay. Wrong: number of the 648 answers that differ from the calculator. Highest: share of wrong answers that are the menu's highest bid (\\%), against the share expected if a wrong answer were a uniformly random non-calculator option (Chance). Closed: same share with the closed-block readout. Sampled: temperature 0.6, top-$p$ 0.95, five seeds. AuctionNet: same share on the second market. Loss: pay-your-bid surplus lost relative to the calculator's bids, as \\% of the calculator's surplus and in thousands of dollars per 1,000 decisions. From highest: share of that loss due to answers that are the highest bid (\\%).",
      "tab:wrong", star=True, size="\\footnotesize")

# ---- regression tables
R5 = J("rev25/r5_passover_regression.json"); RAR = RA["regression"]
NAMES = [("best_is_high", "Best option is the highest bid"), ("finished", "List finished"), ("gap12", "Relative profit gap, best vs runner-up"),
         ("pos", "Position of best option (0 first, 1 last)"), ("K7", "$K=7$"), ("logv", "log value relative to product median")]
rows = [f"{n} & {ORf(R5['pooled'][k])} & {ORf(RAR['pooled'][k])}" for k, n in NAMES]
table("tab_reg", "lcc", "Covariate & eBay & AuctionNet", rows,
      f"Passing over the best option: logistic regression, odds ratios [95\\% interval], errors clustered by scenario. Sample: 768-token traces that write the best option's value and rank it first (eBay $n={R5['n']}$, {R5['events']} passed over; AuctionNet $n={RAR['n']}$, {RAR['events']} passed over). Product (category) and model fixed effects are included; the reference model is Qwen3-8B.",
      "tab:reg")
rows = []
for m in M:
    k1 = f"C(mod)[{LAB[m]}]:pos"; k2 = f"C(mod)[{LAB[m]}]:best_is_high"
    rows.append(f"{LAB[m]} & {ORf(R5['by_model'][k2])} & {ORf(R5['by_model'][k1])}")
table("tab_reg_model", "lcc", "Model & Best is highest & Position", rows,
      "Per-model slopes from the eBay regression with model interactions, odds ratios [95\\% interval].", "tab:regmodel", pos="h!")
PS = J("position_size/position_vs_size.json")
rows = []
for key, lab in (("ebay_deepseek_qwen", "eBay, DeepSeek-8B and Qwen"), ("ebay_all", "eBay, all models"), ("auctionnet_deepseek_qwen", "AuctionNet, DeepSeek-8B and Qwen"), ("auctionnet_all", "AuctionNet, all models")):
    d = PS[key]; rows.append(f"{lab} & {d['n_wrong']} & {ORf(d['high'])} & {ORf(d['first'])} & {ORf(d['last'])}")
table("tab_possize", "lcccc", "Sample & Wrong answers & Highest bid & Listed first & Listed last", rows,
      "Size or position? Conditional logit over the non-calculator options of each wrong 768-token answer, odds ratios [95\\% scenario-bootstrap interval]. Option order is randomised, so position is independent of size.",
      "tab:possize", size="\\footnotesize")

# ---- audit
R4 = J("rev25/r4_audit_oos.json")
rows = []
for m in M:
    rows.append(" & ".join([LAB[m], P(R4["syn_to_ebay"][m]["gap_answer"]), P(R4["syn_to_ebay"][m]["gap_audit"]), P(R4["nested_leave_product_out"][m]["gap_audit"]), P(R4["nested_family_6fold"][m]["gap_audit"]),
                            P(R4["ebay_to_syn"][m]["gap_audit"]), P(R4["nested_leave_product_out"][m]["calls"]),
                            P(RA["audit_ebay_setting"]["by_model"][LAB[m]]["gap_audit"]), P(RA["audit_nested_loco"]["by_model"][LAB[m]]["gap_audit"]), P(RA["audit_ebay_setting"]["by_model"][LAB[m]]["calls"])]))
table("tab_audit", "lccccccccc", "& \\multicolumn{6}{c}{eBay} & \\multicolumn{3}{c}{AuctionNet} \\\\ \\cmidrule(lr){2-7}\\cmidrule(lr){8-10} Model & Answer & Syn.$\\to$eBay & LPO & Fam. & eBay$\\to$Syn. & Flags & eBay set. & LOCO & Flags", rows,
      "The audit: gap recovered (\\%) when bids on which the text reader and the answer disagree are flagged and overridden with the calculator's bid. Answer: no audit. Syn.$\\to$eBay: reader setting chosen on synthetic traces. LPO: nested, setting chosen on two products and tested on the third. Fam.: nested, family 6-fold. eBay$\\to$Syn.: chosen on eBay, tested on synthetic traces. Flags: share of decisions flagged (\\%). AuctionNet: setting chosen on eBay alone, and nested leave-one-category-out.",
      "tab:audit", star=True, size="\\footnotesize")
CA = R4["curve_at"]
rows = [f"{n} & {P(CA[k]['0.1'])} & {P(CA[k]['0.2'])} & {P(CA[k]['0.3'])} & {P(RA['curve_at20'][k])}" for k, n in (("audit", "Audit (reader disagrees)"), ("low_letter_prob", "Lowest letter probability"), ("unfinished_list", "Unfinished list"), ("random", "Random"))]
table("tab_flagrate", "lcccc", "Flag & 10\\% & 20\\% & 30\\% & AuctionNet, 20\\%", rows,
      "Gap recovered (\\%, mean over models) when a fixed share of bids is flagged and overridden, by flagging rule.", "tab:flagrate")

# ---- robustness
R6 = J("rev25/r6_r8_robustness.json"); R9 = J("rev25/r9_fp_robustness.json")
rows = []
for m in M:
    h = R6["hier"][m]["gap768_hier"]
    rows.append(" & ".join([LAB[m], CI(B8[m]["answer"]), f"{100*h['pt']:.0f} [{100*h['ci'][0]:.0f}, {100*h['ci'][1]:.0f}]", f"{100*R6['landscape_both']['models'][m]['share_positive' if False else 'calc_lead_768']['share_positive']:.1f}"]
                + [P(R9[v][m]["gap768"]) for v in ("lognormal", "gamma", "weibull", "increment", "n_minus_1", "n_plus_1")]))
table("tab_robust", "lccc cccccc", "& \\multicolumn{3}{c}{Empirical curve} & \\multicolumn{6}{c}{First-price equilibrium curve, variants} \\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-10} Model & Family boot. & Product boot. & Calc. ahead & Logn. & Gamma & Weib. & Incr. & $n-1$ & $n+1$", rows,
      "Robustness of gap recovered at 768 tokens (\\%). Family boot.: the main interval. Product boot.: resampling the three products, then families within them. Calc. ahead: share of landscape-bootstrap draws (construction and held-out auctions both resampled) in which the calculator beats the model (\\%). First-price variants: value distribution (lognormal, gamma, Weibull), a coarser bid increment, and one bidder fewer or more.",
      "tab:robust", star=True, size="\\footnotesize")

# ---- cost
BE = R6["breakeven"]
rows = [f"{LAB[m]} & {100*BE[m]['cost768_usd']:.2f} & {100*BE[m]['breakeven_usd']:.2f} & {BE[m]['ustar_over_be_min']:.0f} & {BE[m]['ustar_over_be_median']:,.0f} & {BE[m]['gain_over_cost']:,.0f}" for m in M]
table("tab_cost", "lccccc", "Model & Cost (\\textcent) & Break-even (\\textcent) & Min.\\ / B-e & Median / B-e & Gain / cost", rows,
      "Cost of 768 reasoning tokens per decision at measured unbatched speed and \\$0.74--0.79 per GPU-hour, and the break-even best expected profit $c_{768}/(q(768)-q(0))$. Min.\\ and Median: the smallest and median best expected profit across scenarios, as multiples of the break-even. Gain / cost: mean dollar gain from the 768 tokens over their cost.",
      "tab:cost")

# ---- prompting (Run 4)
R4P = J("followup/run4_summary.json")
rows = []
for m in ["phi4_mini_reasoning", "granite_3_2_8b_instruct", "deepseek_r1_llama_8b", "qwen3_8b"]:
    for rt, lab in (("forced768", "768"), ("unforced1024", "1,024 run")):
        d = R4P.get(f"{m}|{rt}")
        if d: rows.append(f"{LAB[m]} & {lab} & {CI(d['gap_paper'])} & {CI(d['gap_new'])} & {CI(d['gap_diff'])}")
table("tab_prompt", "llccc", "Model & Setting & Original prompt & With list instruction & Change", rows,
      "Adding one instruction (``End by listing the expected profit of each option, then answer with the option that has the largest.''): gap recovered, \\% [95\\% interval].", "tab:prompt", size="\\footnotesize")

# ---- facts for the text
F["qcalc"] = 0.981
json.dump(F, open(f"{R}/arxiv/facts.json", "w"), indent=1)
print("tables:", sorted(os.listdir(T)))

# ================= Appendix B tables =================
# B1 full regression coefficients
TERM = [("Intercept", "Intercept"), ("gap12", "Relative profit gap"), ("pos", "Position of best option"), ("K7", "$K=7$"), ("logv", "log value / product median"),
        ("best_is_high", "Best option is the highest bid"), ("finished", "List finished")]
MODT = [(f"C(mod, Treatment('Qwen3-8B'))[T.{l}]", f"Model: {l}") for l in ["DeepSeek-8B", "Phi-mini", "Qwen3-4B", "Qwen3-14B", "Granite-8B"]]
rows = [f"{n} & {ORf(R5['pooled'][k])} & {ORf(RAR['pooled'][k])}" for k, n in TERM + MODT]
rows += [f"eBay product: Palm Pilot & {ORf(R5['pooled']['C(product)[T.Palm]'])} & --",
         f"eBay product: Xbox & {ORf(R5['pooled']['C(product)[T.Xbox]'])} & --",
         f"AuctionNet product: low-intent & -- & {ORf(RAR['pooled']['C(product)[T.low-intent]'])}",
         f"AuctionNet product: mid-intent & -- & {ORf(RAR['pooled']['C(product)[T.mid-intent]'])}"]
table("tab_reg_full", "lcc", "Term & eBay & AuctionNet", rows,
      "All coefficients of the passing-over regression (odds ratios [95\\% interval], errors clustered by scenario; reference model Qwen3-8B). On eBay the reference product is Cartier watches; on AuctionNet it is high-intent impressions.",
      "tab:regfull", size="\\footnotesize", pos="h!")
# B2 unforced runs
rows = []
for m in M:
    f1 = F1[m]["natural_all"]; a = AN0["auctionnet"][m]; e4 = RUN1.get(m); o = ANO[m]
    rows.append(" & ".join([LAB[m], f"{f1['mean_tokens']:.0f}", P(f1["at_cap"]), f"{e4['tokens_median']:.0f}" if e4 else "--", f"{100*e4['at_cap']:.1f}" if e4 else "--",
                            f"{a['unf_mean_tokens']:.0f}", P(a["unf_hit_cap"]), f"{o['unf4096_tokens_median']:.0f}" if o.get("unf4096_tokens_median") else "--", f"{100*o['unf4096_hit_cap']:.1f}" if o.get("unf4096_hit_cap") is not None else "--"]))
table("tab_unforced", "lcccccccc", "& \\multicolumn{4}{c}{eBay} & \\multicolumn{4}{c}{AuctionNet} \\\\ \\cmidrule(lr){2-5}\\cmidrule(lr){6-9} & \\multicolumn{2}{c}{1,024 cap} & \\multicolumn{2}{c}{4,096 cap} & \\multicolumn{2}{c}{1,024 cap} & \\multicolumn{2}{c}{4,096 cap} \\\\ Model & Mean tok. & At cap & Median tok. & At cap & Mean tok. & At cap & Median tok. & At cap", rows,
      "Token use without forcing: mean or median reasoning tokens and the share of decisions that reach the cap (\\%).", "tab:unforced", size="\\footnotesize", pos="h!")
# B3 stated-answer parser
rows = []
for m in M:
    d = R3[f"{m}|768"]; u = R3[f"{m}|-1"]
    rows.append(" & ".join([LAB[m], P(d["coverage"]), P(d["agree_with_letter"]), P(d["stated_right_where_parsed"]), P(d["letter_right_where_parsed"]), P(u["coverage"]), P(u["agree_with_letter"])]))
table("tab_stated", "lcccccc", "& \\multicolumn{4}{c}{768 forced tokens} & \\multicolumn{2}{c}{1,024 run} \\\\ \\cmidrule(lr){2-5}\\cmidrule(lr){6-7} Model & Coverage & Agrees & Stated right & Letter right & Coverage & Agrees", rows,
      "The stated-answer parser on eBay (\\%). Coverage: traces in which the parser finds a stated choice. Agrees: that choice equals the letter. Stated right and Letter right: share of parsed traces in which the stated choice, or the letter, is the calculator's bid.",
      "tab:stated", size="\\footnotesize", pos="h!")
# B4 code check of arithmetic
R2 = J("rev25/r2_code_check.json")
rows = []
for m in M:
    d = R2[f"{m}|768"]
    rows.append(" & ".join([LAB[m], f"{d['options_written']:,}", f"{d['options_found_by_both']:,}", P(d["reader_matches_truth_when_matcher_found"]), P(d["traces_with_a_miscomputed_value"])]))
table("tab_codecheck", "lcccc", "Model & Options written & Found by both & Reader = truth & Traces with a miscomputed value", rows,
      "Agreement between the two pieces of trace-reading code at 768 forced tokens on eBay. Options written: options for which the trace states a value. Found by both: the matcher and the reader both find it. Reader = truth: the reader's number equals the true expected profit (\\%). Last column: traces in which at least one written value is miscomputed (\\%).",
      "tab:codecheck", size="\\footnotesize", pos="h!")
# B5 AuctionNet data and per-category recovery
INFO = json.load(open(f"{R}/AN/ov/results/auctionnet/scenarios_info.json"))["info"]
CATN = [("high-intent impressions", "High-intent"), ("mid-intent impressions", "Mid-intent"), ("low-intent impressions", "Low-intent")]
rows = [f"{lab} & {INFO[c]['n_train']:,} & {INFO[c]['n_test']:,} & {min(INFO[c]['values'])}--{max(INFO[c]['values'])} & {INFO[c]['step']} & {INFO[c]['grid_points']}" for c, lab in CATN]
table("tab_an_data", "lccccc", "Product & Construction opp. & Held-out opp. & Values & Bid step & Grid points", rows,
      "AuctionNet products. Opportunities are ad impressions from periods 7--9 (construction) and 10 (held-out); values and bids are per 1{,}000 impressions. Grid points: candidate bids between the 2nd and 98th percentile of competing prices.",
      "tab:andata", size="\\footnotesize", pos="h!")
rows = [" & ".join([LAB[m]] + [P(AN0["auctionnet"][m]["gap768_by_cat"][c]) for c, _ in CATN]) for m in M]
table("tab_an_cat", "lccc", "Model & High-intent & Mid-intent & Low-intent", rows,
      "AuctionNet gap recovered at 768 forced tokens by product (\\%, point estimates).", "tab:ancat", size="\\footnotesize", pos="h!")
# B6 distribution of the best expected profit (eBay)
UQ = R6["ustar_quantiles"]
rows = [" & ".join([f"\\${UQ[k]:,.2f}" for k in ("0", "5", "25", "50", "75", "95" if "95" in UQ else "90", "100")])]
table("tab_ustar", "ccccccc", "Min & 5th & 25th & Median & 75th & 90th & Max", rows,
      "Best expected profit per eBay scenario, $\\max_b U(b)$, by percentile. It sets the stakes against which the cost of reasoning tokens is compared (Table~\\ref{tab:cost}).",
      "tab:ustar", size="\\footnotesize", pos="h!")
print("appendix tables written")
