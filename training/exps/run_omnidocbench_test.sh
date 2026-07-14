#!/usr/bin/env bash
set -euo pipefail

export no_proxy="${no_proxy:-http://localhost:10000}"
export llm_serve_urls="${llm_serve_urls:-[\"http://localhost:10000\"]}"

export WANDB_PROJECT="${WANDB_PROJECT:-trivia_test}"

BASE_MODEL=${BASE_MODEL:?Set BASE_MODEL to the base model checkpoint path}
SWIFT_BIN="${SWIFT_BIN:-.venv-train/bin/swift}"

MAX_PIXELS=$((512 * 28 * 28)) \
MIN_PIXELS=$((256 * 28 * 28)) \
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" \
CUDA_VISIBLE_DEVICES=2 \
"$SWIFT_BIN" rlhf \
    --rlhf_type grpo \
    --model "$BASE_MODEL" \
    --seed 42 \
    --use_hf true \
    --dataset data_pipeline/outputs/omnidocbench_train.jsonl \
    --external_plugins training/exps/trivia_reward_plugin.py \
    --reward_funcs QA_F1_score TEDS \
    --reward_weights 1 0 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 8000 \
    --sleep_level 0 \
    --num_generations 6 \
    --temperature 1.2 \
    --loss_type bnpo \
    --num_train_epochs 3 \
    --data_seed 42 \
    --dataloader_num_workers 1 \
    --dataset_num_proc 1 \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --padding_side left \
    --padding_free false \
    --max_length 8192 \
    --per_device_train_batch_size 2 \
    --optimizer multimodal \
    --learning_rate 1e-6 \
    --aligner_lr 1e-6 \
    --vit_lr 2e-7 \
    --lr_scheduler_type constant \
    --freeze_aligner false \
    --freeze_vit false \
    --gradient_accumulation_steps 24 \
    --eval_strategy no \
    --save_steps 1 \
    --save_total_limit 1 \
    --save_only_model true \
    --max_completion_length 2048 \
    --max_new_tokens 2048 \
    --logging_steps 1 \
    --output_dir data_pipeline/outputs/omnidocbench_train_run \
    --report_to wandb \
    --warmup_ratio 0.0 \
    --log_completions false
