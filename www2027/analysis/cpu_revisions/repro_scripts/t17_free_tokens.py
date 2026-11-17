"""Token use of the unforced (cap 1,024) greedy run: share of traces using more than 768 tokens, median, share at cap."""
import json, numpy as np
from common_rev import *
u = load_parse(0.02); u = u[u.budget == -1]
R = {m: dict(over768=float((u[u.model == m].reasoning_tokens > 768).mean()), median=float(u[u.model == m].reasoning_tokens.median()),
             at_cap=float((u[u.model == m].reasoning_tokens >= 1024).mean())) for m in MODELS}
json.dump(R, open(f"{OUTDIR}/t17_free_tokens.json", "w"), indent=1); print(json.dumps(R, indent=1))
