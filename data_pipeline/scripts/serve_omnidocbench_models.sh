#!/usr/bin/env bash
set -euo pipefail

MODEL_KIND=${1:?usage: serve_omnidocbench_models.sh base|qa}
VLLM_BIN=${VLLM_BIN:-.venv-vllm/bin/vllm}
QA_MAX_MODEL_LEN=${QA_MAX_MODEL_LEN:-8192}
QA_GPU_MEMORY_UTILIZATION=${QA_GPU_MEMORY_UTILIZATION:-0.85}

case "$MODEL_KIND" in
  base)
    BASE_MODEL=${BASE_MODEL:?Set BASE_MODEL to the base model checkpoint path}
    "$VLLM_BIN" serve "$BASE_MODEL" \
      --served-model-name Qwen/Qwen3-VL-2B-Instruct \
      --chat-template training/qwen3_nonthinking.jinja \
      --host 0.0.0.0 \
      --port 8000
    ;;
  qa)
    QA_MODEL=${QA_MODEL:?Set QA_MODEL to the QA model checkpoint path}
    "$VLLM_BIN" serve "$QA_MODEL" \
      --served-model-name Qwen/Qwen3.5-9B \
      --chat-template training/qwen3_nonthinking.jinja \
      --host 0.0.0.0 \
      --port 10000 \
      --max-model-len "$QA_MAX_MODEL_LEN" \
      --gpu-memory-utilization "$QA_GPU_MEMORY_UTILIZATION"
    ;;
  *)
    echo "usage: serve_omnidocbench_models.sh base|qa" >&2
    exit 2
    ;;
esac
