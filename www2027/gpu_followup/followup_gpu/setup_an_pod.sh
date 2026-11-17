#!/usr/bin/env bash
# Setup for the AuctionNet pod: env_a / env_b (paper's HF versions, for prompt ids and letter readout) + env_v (vLLM, generation only).
set -euo pipefail
cd "$(dirname "$0")"
bash setup_pod.sh
export PATH="$HOME/.local/bin:$PATH"
uv pip install --python env_a/bin/python pyyaml pyarrow pytest
uv pip install --python env_b/bin/python pyyaml pyarrow
uv venv --python 3.12 env_v
uv pip install --python env_v/bin/python vllm pyyaml ninja flashinfer-cubin
env_v/bin/python -c "import vllm, torch; print('env_v vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.is_available())"
rm -rf ~/.cache/uv /workspace/.cache/uv
echo "Setup done."
