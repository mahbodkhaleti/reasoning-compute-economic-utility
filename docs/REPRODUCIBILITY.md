# Reproducibility record

This document records the software, hardware, and Hugging Face snapshots
available for the experiments. The canonical structured record is
`REPRODUCIBILITY_MANIFEST.json`; this narrative explains the retrospective
evidence for the original four checkpoints.

## Scope and provenance

The environment was inspected on 2026-08-23 at 00:33 server local time on
`5glab`. The recorded virtual environment was:

```text
/path/to/original/server/venv
```

The version information below is direct output from that environment. It is
evidence for the completed runs if the virtual environment was not modified
between inference and inspection. The historical result metadata independently
record PyTorch `2.13.0+cu130`, Python `3.10.12`, and an NVIDIA GeForce RTX 4090,
but the older metadata did not record exact Transformers, Accelerate, CUDA
driver, cuDNN, model revision, or tokenizer revision fields.

## Software and accelerator environment

| Component | Recorded value |
|---|---|
| Python | 3.10.12 (from the historical run metadata) |
| Transformers | 5.15.0 |
| Accelerate | 1.14.0 |
| PyTorch | 2.13.0 (`2.13.0+cu130` in the historical run metadata) |
| PyTorch CUDA build | 13.0 |
| NVIDIA driver | 580.173.02 |
| NVIDIA-SMI CUDA compatibility | 13.0 |
| cuDNN | 92000 (cuDNN 9.2.0) |
| GPU | NVIDIA GeForce RTX 4090 |
| GPU memory | 24,564 MiB reported by NVIDIA-SMI |

The inspection command was:

```bash
/path/to/original/server/venv/bin/python - <<'PY'
import importlib.metadata as md
import torch

for package in ["transformers", "accelerate", "torch"]:
    print(package, md.version(package))

print("torch CUDA build:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("GPU:", torch.cuda.get_device_name(0))
PY

nvidia-smi
```

The inspection emitted a deprecation warning stating that the `pynvml`
package is deprecated in favor of `nvidia-ml-py`. This warning does not change
the numerical experiment protocol or results. It should nevertheless be
removed from a future locked environment by replacing a direct `pynvml`
dependency where compatible with the measurement code.

Transient inspection values such as GPU temperature, power draw, utilization,
process ID, and memory occupied by an active process are intentionally not
treated as reproducibility requirements. They describe the server at the time
of inspection rather than the software or experimental configuration.

## Model and tokenizer snapshots

The following were the only snapshot directories found for the four model
repositories in the configured Hugging Face cache during inspection:

| Checkpoint repository | Cached snapshot commit |
|---|---|
| `microsoft/Phi-4-mini-reasoning` | `0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a` |
| `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `6a6f4aa4197940add57724a7707d069478df56b1` |
| `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |

The runner loaded each tokenizer and model from the same Hugging Face repository
and did not pass a distinct tokenizer `revision`. Subject to the cache-integrity
caveat below, the tokenizer assets therefore resolve to the same repository
snapshot shown in the table.

### Historical-identification caveat

The completed-run JSON metadata did not save the resolved commit hashes. The
hashes above are the sole snapshots present in the server cache at inspection,
not commit fields recovered from the historical JSONL records. They identify
the snapshots used by the completed experiments only under the assumption that
the cache was neither replaced nor supplemented between the runs and this
inspection. They must not be described as independently logged historical
revisions unless that assumption can be verified from server backups or logs.

The current runner records `model_revision` and `tokenizer_revision`
prospectively so future runs do not require this inference.

The machine-readable audit is archived in `docs/HF_CACHE_AUDIT.json`, and the
reproduction config pins these recovered snapshots explicitly. This improves
replicability but does not convert post-hoc cache evidence into a revision field
logged contemporaneously with the completed inference.

The cache inspection can be repeated with:

```bash
for repo in \
  models--microsoft--Phi-4-mini-reasoning \
  models--Qwen--Qwen3-4B \
  models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B \
  models--Qwen--Qwen3-8B
do
  echo "=== $repo ==="
  find "/path/to/huggingface/cache/$repo/snapshots" \
    -mindepth 1 -maxdepth 1 -type d -printf '%f\n'
done
```

## Recommended archival action

Before releasing the anonymous reproducibility package, preserve:

1. this file;
2. a fully pinned package lock or `pip freeze` captured from the server;
3. the four snapshot hashes above;
4. `scripts/config.json`, prompt/runner code, and all per-run metadata;
5. the existing SHA-256 output manifest.

For a fresh reproduction, pass the recorded commit explicitly as the model and
tokenizer `revision` rather than relying on the mutable `main` branch. No model
weights should be copied into the anonymous repository; they should be obtained
from their original Hugging Face repositories subject to their licenses.

## Qwen3-14B, Granite-8B, and offline replay

Qwen3-14B used `Qwen/Qwen3-14B` at model and tokenizer commit
`40c069824f4251a91eefaf281ebe4c544efd3e18`. Unlike the original-four panel,
the requested and resolved commits were written to each run manifest before
and during inference. Those manifests also record Python 3.12.3, PyTorch
2.9.1+cu128, Transformers 5.1.0, Accelerate 1.12.0, CUDA 12.8, driver
580.159.04, bfloat16 loading, and one NVIDIA L40S with 47,665,709,056 bytes of
memory.

Granite-8B used `ibm-granite/granite-3.2-8b-instruct` at model and tokenizer
commit `610d8c6ee9c84ce51f6dfd7bc5c0215d95d49695`. Its manifests recorded the
resolved revisions contemporaneously together with Python 3.10.12, PyTorch
2.13.0+cu130, Transformers 5.15.0, Accelerate 1.14.0, CUDA 13.0, driver
580.173.02, bfloat16 loading, and one NVIDIA RTX 4090. The tokenizer requires
`thinking=True`; this protocol is recorded as not semantically identical to
Qwen native thinking even though external token budgeting, readout, and scoring
are shared.

All six historical and synthetic panels are preserved under
`robustness_extensions/server_outputs/raw/` and `preserved_outputs/`. To verify
coverage and regenerate the reported analyses without model inference, run
from the repository root:

```bash
bash reproduce_all_from_preserved.sh
```

The original `analyze_fixed.py` used Python's randomized `hash()` for
descriptive cell-level frontier intervals. The canonical replay uses a stable
SHA-256-derived seed for those cells. The paired endpoint, hierarchical,
free-form, CDF, and strategic analyses already use explicit seeds; their
estimands are unchanged.
