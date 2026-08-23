# Qwen3-14B paper-addition package

This package collects every **new model inference** and **cheap downstream
analysis** required to evaluate Qwen3-14B under the protocol already used in
the EconML paper. It does not edit the manuscript and does not fabricate table
entries. Run it first; merge the validated outputs into the paper only after
the results have been inspected.

## Pinned model

- Model and tokenizer: `Qwen/Qwen3-14B`
- Revision: `40c069824f4251a91eefaf281ebe4c544efd3e18`
- Dtype: BF16
- License: Apache-2.0
- Official model page: <https://huggingface.co/Qwen/Qwen3-14B>

The requested revision is written to a run manifest before model loading. The
runner then resolves the downloaded model and tokenizer revisions and aborts
before the first evaluated cell if either differs. Every JSONL row also stores
both requested and resolved revisions.

## Protocol reproduced

The runner uses the existing 216 auction-grounded scenarios and prompt
templates without modification.

| Study | Budgets | Seeds | Option order | Rows |
|---|---:|---:|---|---:|
| Deterministic reasoning | `0,64,128,256,512,768,unconstrained` | `11,23,37` | randomized by order seed | 4,536 |
| Deterministic no-think | `0` | `11,23,37` | randomized by order seed | 648 |
| Stochastic reasoning | `128,384,768` | `1,7,21,42,99` | fixed seed `314159` | 3,240 |

Total: 8,424 scenario/budget/seed rows. The primary answer is the same
constrained option-letter next-token readout used in the paper. The runner also
stores the free-form answer, answer-token confidence, and (for the stochastic
study) a separate verbal-confidence readout.

Greedy decoding disables sampling. Stochastic decoding uses temperature `0.6`,
top-p `0.95`, and the pinned Qwen generation configuration's top-k `20`. The
resolved generation configuration is recorded before evaluated inference and
checked by the final validator.

For a finite budget, the reasoning stage is forced to exactly that many newly
generated tokens by setting equal minimum and maximum generation lengths and
blocking early EOS. The answer is then elicited separately and the primary
choice is obtained from constrained logits over the displayed option letters.
At budget zero no reasoning-stage token is generated; the constrained answer
is read directly from the prompt plus answer suffix. This matches the existing
runner rather than introducing a Qwen3-14B-specific protocol.

## A100 40 GB setup

Use the exact environment recorded for the earlier runs:

```bash
cd /path/to/reasoning-compute-economic-utility/qwen3_14b_addition
bash setup_environment.sh
export QWEN14_PYTHON_BIN="$PWD/.venv/bin/python"
```

If the existing server environment already has `torch==2.13.0`,
`transformers==5.15.0`, and `accelerate==1.14.0`, it can be reused:

```bash
export QWEN14_PYTHON_BIN=/path/to/existing/environment/bin/python
```

The preflight check requires CUDA, BF16 support, and at least 39 GiB visible on
GPU 0. Qwen3-14B BF16 should fit on a 40 GB A100, but run the isolated smoke
test before committing to the full job:

```bash
bash run_smoke_test.sh
```

The smoke outputs are stored separately and are never included in paper
results.

## Required data inputs

`scenarios.csv` and `auction_splits.csv` are not redistributed because the
license of the Online Auctions mirror is unspecified. By default, the wrapper
copies the protocol-matched versions from:

```text
../robustness_extensions/server_outputs/data/
```

If they are elsewhere:

```bash
export QWEN14_SOURCE_DATA_DIR=/absolute/path/to/server_outputs/data
```

If they must be reconstructed from a lawfully obtained `auction.csv`:

```bash
export QWEN14_AUCTION_CSV=/absolute/path/to/online-auctions-dataset/auction.csv
```

The existing deterministic preparation code and seed will rebuild them.

## Full run with checkpoint/resume

For a durable server session:

```bash
cd /path/to/reasoning-compute-economic-utility/qwen3_14b_addition
export QWEN14_PYTHON_BIN=/path/to/existing/environment/bin/python
export QWEN14_RESULTS_DIR=/path/to/qwen3_14b_results
nohup bash run_qwen3_14b.sh all > qwen3_14b_master.log 2>&1 &
echo $! > qwen3_14b_master.pid
tail -f qwen3_14b_master.log
```

Each JSONL row is flushed and `fsync`ed. Rerunning the same command skips every
completed `(scenario_id, budget, seed)` key and continues missing cells. A
fully complete file is detected before model loading, so completed phases do
not reload the checkpoint.

The phases can also be run separately:

```bash
bash run_qwen3_14b.sh deterministic
bash run_qwen3_14b.sh no_think
bash run_qwen3_14b.sh stochastic
bash run_qwen3_14b.sh analyze
bash run_qwen3_14b.sh validate
```

Optional measured monetary cost can be recorded without changing any choice
or strategic normalization:

```bash
export ROBUSTNESS_GPU_HOUR_USD=YOUR_A100_RENTAL_PRICE
```

Do not reuse the earlier RTX 4090 price unless it is actually the A100 rental
price for this run.

## Runtime

Wall time depends strongly on the A100 variant, clocks, and serving stack.
After a pilot or interrupted phase, obtain an empirical remaining-time estimate
from the actual recorded cell times:

```bash
bash run_qwen3_14b.sh estimate
```

The estimator is more reliable than a hardware-generic promise and warns that
early low-budget cells can underestimate later high-budget cells.

## Analyses and outputs

The offline stage generates model-only versions of all analyses required to
add the checkpoint to the paper:

- deterministic compute--utility frontier and paired endpoint bootstrap;
- product, candidate-set, and randomized-order breakdowns;
- stochastic frontier and hierarchical strategic-game bootstrap;
- constrained-versus-free-form validity/agreement sensitivity;
- answer-token and verbal-confidence analyses;
- beta and gamma sensitivity;
- deterministic six-budget compute-menu robustness;
- auction-CDF bootstrap using fixed stored choices;
- normalized `(kappa,rho,gamma)` phase maps and exact transitions;
- hybrid zero-compute sensitivity and bootstrap.

The package does **not** run direct online games. Those runs are not needed for
the paper's retained conditional game results, which enumerate the calibrated
frontier offline. Adding direct games would be a separate experiment rather
than a requirement for a protocol-matched fifth checkpoint.

The final archive is written to:

```text
scripts/qwen3_14b_paper_results.zip
```

It contains raw model outputs, tables, figures, code, configuration, validation
reports, and SHA-256 hashes. Auction-derived data and machine-specific logs are
hashed in the manifest but intentionally excluded from the archive.
