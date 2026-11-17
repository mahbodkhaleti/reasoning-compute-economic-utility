# source this in every new terminal:  source env.sh
export HF_HOME=/workspace/hf
export ROBUSTNESS_HF_CACHE=/workspace/hf/hub
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
pyfor() { if [[ "$1" == "qwen3_14b" ]]; then echo env_b/bin/python; else echo env_a/bin/python; fi; }
export VLLM_USE_FLASHINFER_SAMPLER=0
