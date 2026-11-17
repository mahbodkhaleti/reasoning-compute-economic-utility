"""Run the paper's trace extractors unchanged (analysis/extract5.py = 2% parser v2; extract8.py = text-reader
segments) on follow-up traces. Usage: python3 prep_traces.py <run_dir> <tag>
Writes ../followup_<tag>_first.csv and ../followup_<tag>_segments.csv.gz."""
import json, os, sys, subprocess, tempfile, csv, gzip, shutil, glob
run_dir, tag = os.path.abspath(sys.argv[1]), sys.argv[2]
A = "/tmp/www2027/analysis"
tmp = tempfile.mkdtemp(); raw = f"{tmp}/robustness_extensions/server_outputs/raw"; os.makedirs(raw)
for f in glob.glob(f"{run_dir}/*.jsonl"):
    m = os.path.basename(f)[:-6]
    with open(f) as src, open(f"{raw}/fixed_{m}_reasoning_greedy.jsonl", "w") as dst:
        for line in src:
            r = json.loads(line)
            r["budget"] = f'{r["setting"]}'   # keep settings apart (e.g. unforced4096, forced768)
            dst.write(json.dumps(r) + "\n")
disp = {}
for r in csv.DictReader(open(f"{A}/scenarios.csv")):
    disp[str(int(r["scenario_id"]))] = {str(float(c["bid"])): round(float(c["train_p"]), 4) for c in json.loads(r["candidates_json"])}
json.dump(disp, open(f"{tmp}/disp.json", "w"))
subprocess.run([sys.executable, f"{A}/extract5.py", tmp, f"{tmp}/disp.json", f"{A}/followup_{tag}_first.csv"], check=True)
seg = f"{A}/followup_{tag}_segments.csv"
subprocess.run([sys.executable, "/tmp/www2027/analysis/revision/extract8.py", tmp, seg, "historical"], check=True)
with open(seg, "rb") as fi, gzip.open(seg + ".gz", "wb") as fo: shutil.copyfileobj(fi, fo)
os.remove(seg); shutil.rmtree(tmp)
