"""Rebuild Tables 1-3 from the re-run outputs in results/cpu_revisions/repro/ (the paper's scripts re-executed with
outputs redirected) and compare every cell with the numbers printed in paper/main.tex.
Writes results/cpu_revisions/tables_check.json and prints any mismatch."""
import json, re, os
RP = "/tmp/www2027/results/cpu_revisions/repro"; TEX = "/tmp/www2027/paper/main.tex"
J = lambda n: json.load(open(f"{RP}/{n}.json"))
t7, t8, t9, t10, t10b, t12, t13 = J("t7_costs"), J("t8_fixes"), J("t9_judge6"), J("t10_judge7"), J("t10b_hybrid_split"), J("t12_judge9"), J("t13_judge10")
MODELS = ["deepseek_r1_llama_8b", "phi4_mini_reasoning", "qwen3_4b", "qwen3_8b", "qwen3_14b", "granite_3_2_8b_instruct"]
LAB = dict(zip(MODELS, ["DeepSeek-8B", "Phi-mini", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Granite-8B"]))
p = lambda x: round(100 * x)
ci = lambda c: f"[{p(c[0])},{p(c[1])}]"
rows = {}
for m in MODELS:
    b8 = t10["B8"][m]; f1 = t8["F1"][m]["natural_all"]; c = t7[LAB[m]]
    q768 = t9["A1"][m]["by_budget"][5]["q_read"]
    rows[("T1", m)] = [f"{q768:.3f}".lstrip("0"), f"{p(b8['answer']['pt'])} {ci(b8['answer']['ci'])}", f"{p(b8['ge2']['pt'])} {ci(b8['ge2']['ci'])}",
                       f"{p(b8['oracle']['pt'])} {ci(b8['oracle']['ci'])}", f"{p(f1['gap_readout'])} {ci(f1['gap_readout_ci'])}",
                       f"{p(b8['dollar']['pt'])} {ci(b8['dollar']['ci'])}", f"{100 * c['breakeven_usd']:.1f}"]
    a3 = t9["A3"][m]; sp = t10b["forced768"][m]
    rows[("T2", m)] = [f"{p(a3['forced768_complete'])}", f"{a3['K5']['median']}/{a3['K7']['median']}"] + \
                      [f"{p(sp[k]['share'])} [{p(sp[k]['ci'][0])}, {p(sp[k]['ci'][1])}]" for k in ("not_reached", "arith", "incomplete", "passed")]
    e6 = t13["E6"][m]; d1 = t12["D1"][m]; a5 = t9["A5"][m]
    rows[("T3", m)] = [f"{p(e6['right768_if_complete'])}", f"{p(d1['passed_rate_nat_complete'])}", (a5["or_best"], a5["or_best_ci"])]
tex = open(TEX).read()
def tex_row(table_label, name):
    i = tex.index(f"\\label{{{table_label}}}"); j = tex.index("\\bottomrule", i)
    line = next(l for l in tex[i:j].splitlines() if l.strip().startswith(name + " "))
    cells = [re.sub(r"\\phantom\{0\}|\\textcent|\\%|\$|\\\\", "", x).strip() for x in line.split("&")[1:]]
    return cells
out = {"mismatches": [], "rows": {}}
for (tab, m), mine in rows.items():
    lbl = {"T1": "tab:main", "T2": "tab:traces", "T3": "tab:checks"}[tab]
    paper = tex_row(lbl, LAB[m])
    norm = lambda s: re.sub(r"\s+", "", str(s)).replace("−", "-").replace("$-$", "-")
    if tab == "T3":
        orv, orc = mine[2]
        pv = paper[2]; num = float(pv.split("[")[0]); lo, hi = [float(x) for x in pv.split("[")[1].rstrip("]").split(",")]
        def same(printed, val):
            s_ = str(printed).strip(); dec = len(s_.split(".")[1]) if "." in s_ else 0
            return round(val, dec) == float(s_)
        ps = pv.split("[")[0].strip(); pl, ph = [x.strip() for x in pv.split("[")[1].rstrip("]").split(",")]
        okor = same(ps, orv) and same(pl, orc[0]) and same(ph, orc[1])
        mine_s = mine[:2] + [f"{orv:.2f} [{orc[0]:.2f}, {orc[1]:.2f}]"]
        bad = [k for k in range(2) if norm(mine[k]) != norm(paper[k])] + ([] if okor else [2])
    else:
        mine_s = mine
        bad = [k for k in range(len(mine)) if norm(mine[k]) != norm(paper[k])]
    out["rows"][f"{tab}|{LAB[m]}"] = dict(recomputed=mine_s, paper=paper, mismatched_cells=bad)
    if bad: out["mismatches"].append(dict(table=tab, model=LAB[m], cells=[(mine_s[k], paper[k]) for k in bad]))
json.dump(out, open("/tmp/www2027/results/cpu_revisions/tables_check.json", "w"), indent=1)
for k, v in out["rows"].items(): print(k, "OK" if not v["mismatched_cells"] else f"MISMATCH {[(v['recomputed'][c], v['paper'][c]) for c in v['mismatched_cells']]}")
print("total mismatches:", len(out["mismatches"]))
