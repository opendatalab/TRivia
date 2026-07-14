#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-data_pipeline/configs/omnidocbench_test_config.py}
PYTHON_BIN=${PYTHON_BIN:-.venv-vllm/bin/python}

"$PYTHON_BIN" -m data_pipeline.cli sample --config "$CONFIG"
"$PYTHON_BIN" -m data_pipeline.cli filter --config "$CONFIG"
"$PYTHON_BIN" -m data_pipeline.cli generate-qa --config "$CONFIG"
"$PYTHON_BIN" data_pipeline/scripts/build_training_jsonl.py \
  --input-jsonl data_pipeline/outputs/omnidocbench_work/s3_qa_generation/s3.jsonl \
  --output-jsonl data_pipeline/outputs/omnidocbench_train.jsonl
