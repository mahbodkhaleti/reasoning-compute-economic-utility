"""A8: gap recovered at 768 under each scoring curve (model choices fixed; models saw the eBay empirical curve).
Reads the outputs of analysis/revision/t11_judge8.py (C5: lognormal, shaded), t13_judge10.py (E1: order statistic),
t14_judge11.py (F1: first-price equilibrium) and t10_judge7.py (B8: main empirical curve). Writes A8_curves.json."""
import json
O = "/tmp/www2027/analysis/revision/out"
t10, t11, t13, t14 = [json.load(open(f"{O}/{n}.json")) for n in ("t10_judge7", "t11_judge8", "t13_judge10", "t14_judge11")]
MODELS = ["deepseek_r1_llama_8b", "phi4_mini_reasoning", "qwen3_4b", "qwen3_8b", "qwen3_14b", "granite_3_2_8b_instruct"]
R = {m: dict(empirical=t10["B8"][m]["answer"]["pt"], lognormal=t11["C5"]["lognormal"][m]["gap_recovered"], shaded_085=t11["C5"]["shaded085"][m]["gap_recovered"],
             order_statistic=t13["E1"][m]["gap_recovered"], first_price_equilibrium=t14["F1"][m]["gap_recovered"]) for m in MODELS}
json.dump(R, open("/tmp/www2027/results/cpu_revisions/A8_curves.json", "w"), indent=1)
for m, v in R.items(): print(m, {k: round(100 * x) for k, x in v.items()})
