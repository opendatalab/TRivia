# #!/bin/bash

export no_proxy="http://xxxx:10001" # url for qa llm service

MAX_PIXELS=$((1280 * 28 * 28)) \
MIN_PIXELS=$((256 * 28 * 28)) \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model /path/to/base_model \
    --seed 42 \
    --use_hf true \
    --train_type full \
    --dataset /path/to/dataset \
    --val_dataset /path/to/val_dataset \
    --reward_funcs QA_F1_score TEDS \
    --reward_weights 1 0 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.2 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 8000 \
    --sleep_level 0 \
    --num_generations 8 \
    --rejection_sample false \
    --rejection_strategy tag_tail \
    --temperature 1.2 \
    --loss_type bnpo \
    --data_seed 42 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --train_type full \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --padding_side left \
    --padding_free false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --optimizer multimodal \
    --learning_rate 1e-6 \
    --aligner_lr 1e-6 \
    --vit_lr 2e-7 \
    --lr_scheduler_type constant \
    --freeze_aligner false \
    --freeze_vit false \
    --gradient_accumulation_steps 8 \
    --eval_steps 5 \
    --save_steps 5 \
    --save_total_limit 2 \
    --save_only_model true \
    --max_completion_length 4500 \
    --max_new_tokens 4500 \
    --logging_steps 1 \
    --output_dir /path/to/output \
    --report_to wandb \
    --warmup_ratio 0.05 \
    --deepspeed zero3_offload \
    --log_completions false