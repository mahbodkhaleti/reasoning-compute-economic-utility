#!/usr/bin/env bash
# One-time setup on the RunPod pod (RTX 6000 Ada, network volume at /workspace).
# Builds two environments with the library versions recorded in the paper's run metadata:
#   env_a: python 3.10, torch 2.13.0+cu130, transformers 5.15.0, accelerate 1.14.0
#          (Phi-mini, Qwen3-4B, Qwen3-8B, DeepSeek-8B, Granite-8B)
#   env_b: python 3.12, torch 2.9.1+cu128, transformers 5.1.0, accelerate 1.12.0 (Qwen3-14B)
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${HF_TOKEN:-}" ]]; then echo "Note: HF_TOKEN not set. The models are public, so this is fine, but downloads may be slower."; fi
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)   driver: $driver"
if [[ ${driver%%.*} -lt 580 ]]; then
  echo "Driver $driver is too old for torch 2.13.0+cu130 (needs 580 or newer)."
  echo "Redeploy the pod with the CUDA 13.0 filter selected, then rerun this script."
  exit 1
fi

command -v uv >/dev/null || { pip install -q uv || curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.10 env_a
uv pip install --python env_a/bin/python --index-url https://download.pytorch.org/whl/cu130 "torch==2.13.0"
uv pip install --python env_a/bin/python "transformers==5.15.0" "accelerate==1.14.0" pandas numpy nvidia-ml-py

uv venv --python 3.12 env_b
uv pip install --python env_b/bin/python --index-url https://download.pytorch.org/whl/cu128 "torch==2.9.1"
uv pip install --python env_b/bin/python "transformers==5.1.0" "accelerate==1.12.0" pandas numpy nvidia-ml-py

for e in env_a env_b; do
  $e/bin/python -c "import torch,transformers;print('$e', torch.__version__, transformers.__version__, 'cuda ok:', torch.cuda.is_available())"
done
echo "Setup done."
