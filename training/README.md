# TRivia training
We use [ms-swift](https://github.com/modelscope/ms-swift) to build the training framework, and implement the filtering-based stabilization in the [`ms-swift/swift/trainers/rlhf_trainer/grpo_trainer.py`](./ms-swift/swift/trainers/rlhf_trainer/grpo_trainer.py) file, and implement the QA-based reward in the [`ms-swift/swift/plugin/orm.py`](./ms-swift/swift/plugin/orm.py) file.



# Installation

```bash
pip install vllm==0.8.5 --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate
pip install qwen-vl-utils[decord]
pip install bs4
pip install Levenshtein
pip install apt
# Optioanl
pip install -U flash-attn --no-build-isolation
```

ms-swift
```bash
cd ms-swift
pip install -e .
```

# Usage

## Data preparation

We follow the [ms-swift data format](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html#id6) and add an extra column for `qa_paris` to indicate the QA for this image. Here is an example of a single data entry.

```json
{
    "messages": [ // Prompt and label for table recognition. Should be replaced by your owns.
        {"role": "user", "content": "You are an AI specialized in recognizing and extracting table from the image. Your mission is to analyze the table in the image and generate the result in OTSL format using specified tags.\n<image>"}, 
        {"role": "assistant", "content": "<table></table>"}
    ],
    "qa_pairs": [ // A list of qa pairs generated for this image.
        {"question": "What is the unit listed for all values under Isc current?", "answer": "mA"}, 
        ...
    ],
    "images": ["/path/to/img"]
}
```

### Validation dataset
You can also prepare a validation dataset with HTML annotations for calculating TEDS during evaluation. The data format should be as follows:

```json
{
    "messages": [ // Prompt and label for table recognition. Should be replaced by your owns.
        {"role": "user", "content": "You are an AI specialized in recognizing and extracting table from the image. Your mission is to analyze the table in the image and generate the result in OTSL format using specified tags.\n<image>"}, 
        {"role": "assistant", "content": "..."} // grount truth html annotation for this table image
    ],
    "images": ["/path/to/img"]
}
```


## Run experiment

We recommend using at least two servers: one to deploy the language model for QA and the other to run GRPO training.

First, launch the language model service on other deployments for QA. Here is an example of deploying Qwen3-8B.
```bash
vllm serve Qwen/Qwen3-8B --served-model-name Qwen/Qwen3-8B --chat-template ./qwen3_nonthinking.jinja --host 0.0.0.0 --port 10000
```

Then, configure the IP address and port of the QA service on the training server and run training scripts.
```bash
# Configure the IP address and port of the QA service.
export llm_serve_urls="http://xxxx:10000"

# Configure your dataset in training scripts and run. 
bash exp/run.sh

# exp/run.sh
export no_proxy="http://xxxx:10000" # url for qa llm service
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
    --rejection_sample false \ # Filter-based stablization
    --rejection_strategy tag_tail \ # We determine whether the output is valid by checking if it ends with a specific character.
    --rejection_tag_tail "<nl>\n" \ # For OTSL tags, a valid ending should be `<nl>`, and for HTML tables, it should be `</table>` or `</html>`.
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
```