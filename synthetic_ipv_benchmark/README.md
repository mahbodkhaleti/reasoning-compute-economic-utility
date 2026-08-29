# Canonical synthetic IPV first-price robustness benchmark

This directory defines the single synthetic benchmark used by both additional
server packages. It is a controlled decision-quality benchmark inspired by the
IPV auction environment in Zhu et al. (2024); it is **not** an exact replication
of their multi-agent experiment. The source describes three bidders, independent
private values uniform on `[0,99]`, one-dollar bid increments, and random tie
resolution. Its agents interact in repeated auction simulations. Here those
features motivate a fixed candidate-choice task, so the object being measured is
decision quality under controlled reasoning compute rather than emergent
multi-agent auction behavior.

Source checked: the NeurIPS 2024 workshop paper on
[OpenReview](https://openreview.net/pdf?id=FB9mTtJpJI), especially its IPV
first-price/second-price benchmark description.

Each of 216 tasks gives one focal bidder an integer private value, five or seven
candidate bids, and the exact win probability for each bid. Two opponent values
are independently uniform on the integers `{0,...,99}`. The risk-neutral
symmetric continuous-IPV benchmark strategy `b(v)=2v/3` is assigned to the two
opponents. Win probabilities are
computed by exact enumeration over all 10,000 opponent-value pairs, with ties
split uniformly among the highest bidders. Utility is `(value-bid)*p_win`.

Candidate sets are generated before and without consulting the optimal bid.
The design has 54 scenarios in each `K x style` cell for `K in {5,7}` and
`style in {structured,unstructured}`. The optimum is computed only after the
candidate set exists. A deterministic retry rule applies if all utilities are
non-positive.

Regenerate and validate with:

```bash
python generate_scenarios.py
python validate_scenarios.py
```

Both server packages pin `synthetic_ipv_manifest.jsonl` by the SHA256 recorded
in `manifest.sha256` and abort before inference if it differs.

After both packages finish, merge only hash- and coverage-identical results with:

```bash
python merge_six_model_results.py \
  --package-a ../synthetic_existing4_server_package \
  --package-b ../new_models_full_server_package
```

The merge checks all 4,536 keys per model, benchmark and scoring versions, the
manifest hash, and the exact scenario-ID set before producing combined tables,
figures, strategic summaries, and a descriptive Qwen-family synthetic table.
