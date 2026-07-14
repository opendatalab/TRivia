# TRivia Data Pipeline

This directory contains the standalone data-generation pipeline used by TRivia. It starts from table-recognition samples, uses a base vision-language model to produce diverse table candidates, filters samples by structural consistency, generates QA supervision, converts the result into an ms-swift training file, and evaluates the trained model with TEDS.

All commands below assume they are run from the repository root:

```bash
cd TRivia
```

## Directory Layout

```text
data_pipeline/
|-- cli.py                         # Command-line entry point for pipeline stages.
|-- config.py                      # Default pipeline configuration schema.
|-- configs/
|   `-- omnidocbench_test_config.py # Demo config for OmniDocBench.
|-- prompts/
|   |-- qa_answer.txt              # Prompt used for QA-based evaluation/filtering.
|   `-- qa_generation.txt          # Prompt used to generate QA pairs.
|-- scripts/
|   |-- build_training_jsonl.py    # Converts pipeline output to ms-swift training JSONL.
|   |-- evaluate_table_recognition.py
|   |-- evaluate_table_recognition_vllm.py
|   |-- prepare_omnidocbench_tables.py
|   |-- run_omnidocbench_pipeline.sh
|   `-- serve_omnidocbench_models.sh
|-- stages/
|   |-- s1_base_sampling.py        # Generate multiple table candidates per input image.
|   |-- s2_teds_filter.py          # Filter by pairwise TEDS consistency.
|   |-- s3_qa_generation.py        # Generate QA pairs for selected samples.
|   |-- s5_qa_evaluation.py        # Optional QA answer evaluation.
|   `-- s6_qa_filter.py            # Optional filtering by answer F1.
`-- utils/
    |-- io.py
    |-- openai_client.py
    |-- teds.py
    `-- vllm_backend.py
```

Generated data is written under `data_pipeline/data/` and `data_pipeline/outputs/`. Those directories are intentionally treated as runtime artifacts.

## Environment Setup

The pipeline requires Python 3.10+ and a CUDA-capable GPU for vLLM-based generation and evaluation. The examples below use `uv`, but a regular `python -m venv` environment also works.

Install `uv` if needed:

```bash
python -m pip install uv
```

Install the data pipeline and inference dependencies. In fact, all we need to do is install vLLM:

```bash
uv venv .venv-vllm --python 3.12
source .venv-vllm/bin/activate
uv pip install vllm --torch-backend=cu129
uv pip install requests pillow openai tqdm transformers qwen-vl-utils peft
```

If your CUDA or PyTorch setup requires a specific wheel index, install `torch` and `vllm` according to your platform first, then install the remaining packages. Model checkpoints are not bundled with this repository; download or place them locally and pass their paths through the config files or environment variables shown below.

## Pipeline Stages

The main entry point is:

```bash
python -m data_pipeline.cli <command> --config <path/to/config.py>
```

Supported commands:

| Command | Stage | What it does | Main output |
| --- | --- | --- | --- |
| `sample` | S1 | Runs the base table-recognition model and samples multiple HTML candidates for each table image. | `s1_base_sampling/s1.jsonl` |
| `filter` | S2 | Computes pairwise TEDS among candidates and keeps samples in the configured consistency range. | `s2_teds_filter/s2.jsonl` |
| `generate-qa` | S3 | Uses a QA model to generate question-answer pairs for each kept table. | `s3_qa_generation/s3.jsonl` |
| `evaluate-qa` | S5 | Optionally asks a QA model to answer generated questions from table predictions. | `s5_qa_evaluation/s5.jsonl` |
| `filter-qa` | S6 | Optionally filters rows by answer F1. | `s6_qa_filter/s6.jsonl` |
| `run-all` | S1-S6 | Runs the configured stages in order. | Configured final output |

The default config schema lives in `data_pipeline/config.py`. To customize a run, create a Python config with the same uppercase variable names, then pass it with `--config`.

Example:

```bash
python -m data_pipeline.cli sample --config data_pipeline/configs/omnidocbench_test_config.py
python -m data_pipeline.cli filter --config data_pipeline/configs/omnidocbench_test_config.py
python -m data_pipeline.cli generate-qa --config data_pipeline/configs/omnidocbench_test_config.py
```

After S3, convert the generated QA data into the ms-swift training format:

```bash
python data_pipeline/scripts/build_training_jsonl.py \
  --input-jsonl data_pipeline/outputs/omnidocbench_work/s3_qa_generation/s3.jsonl \
  --output-jsonl data_pipeline/outputs/omnidocbench_train.jsonl
```

## OmniDocBench Demo

This repository includes an end-to-end demo that builds a TRivia training set from the OmniDocBench table-recognition benchmark.

### 1. Prepare Table Crops

`scripts/prepare_omnidocbench_tables.py` downloads `OmniDocBench.json` and page images from `opendatalab/OmniDocBench` on Hugging Face, crops table regions, and writes the input JSONL expected by the pipeline.

```bash
python data_pipeline/scripts/prepare_omnidocbench_tables.py \
  --output-jsonl data_pipeline/data/omnidocbench_tables.jsonl \
  --image-dir data_pipeline/data/images \
  --raw-dir data_pipeline/data/raw/omnidocbench
```

For a quick smoke test:

```bash
python data_pipeline/scripts/prepare_omnidocbench_tables.py \
  --output-jsonl data_pipeline/data/omnidocbench_tables.jsonl \
  --image-dir data_pipeline/data/images \
  --raw-dir data_pipeline/data/raw/omnidocbench \
  --max-tables 20
```

Each output row contains the cropped table image, the reference HTML annotation, and the table-recognition prompt.

### 2. Start Model Services

The OmniDocBench demo config uses two OpenAI-compatible vLLM services:

| Service | Default URL | Role |
| --- | --- | --- |
| Base VLM | `http://localhost:8000/v1` | Generates candidate table HTML. |
| QA LLM | `http://localhost:10000/v1` | Generates QA pairs and provides reward-time QA answers. |

Start the base model service in one terminal:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
BASE_MODEL=/path/to/Qwen3-VL-2B-Instruct \
bash data_pipeline/scripts/serve_omnidocbench_models.sh base
```

Start the QA model service in another terminal:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
QA_MODEL=/path/to/Qwen3.5-9B \
bash data_pipeline/scripts/serve_omnidocbench_models.sh qa
```

The served model names are configured in `data_pipeline/scripts/serve_omnidocbench_models.sh` and must match the names in `data_pipeline/configs/omnidocbench_test_config.py`.

### 3. Run the Demo Pipeline

Run the full demo pipeline:

```bash
bash data_pipeline/scripts/run_omnidocbench_pipeline.sh
```

This script runs `sample`, `filter`, `generate-qa`, and `build_training_jsonl.py`. The final training file is:

```text
data_pipeline/outputs/omnidocbench_train.jsonl
```

You can also override the demo config:

```bash
CONFIG=/path/to/custom_omnidocbench_config.py \
bash data_pipeline/scripts/run_omnidocbench_pipeline.sh
```

## Training

Install the training stack:

```bash
uv venv .venv-train --python 3.12
source .venv-train/bin/activate
uv pip install vllm --torch-backend=cu129
uv pip install -e training/ms-swift
uv pip install wandb qwen-vl-utils
```

The OmniDocBench training experiment is defined in:

```text
training/exps/run_omnidocbench_test.sh
```

It trains with GRPO through ms-swift, reads `data_pipeline/outputs/omnidocbench_train.jsonl`, and loads rewards from `training/exps/trivia_reward_plugin.py`.

The reward plugin contains:

| Reward | Purpose |
| --- | --- |
| `QA_F1_score` | Ask a QA model to answer generated questions from the predicted table and compare against reference answers. |
| `TEDS` | Measure structural similarity between predicted and reference table HTML. |

Before training, make sure the QA reward service is running, authenticate with Weights & Biases using `wandb login` (or set `WANDB_API_KEY` in your environment), and provide the base-model checkpoint path:

```bash
export llm_serve_urls='["http://localhost:10000"]'
export BASE_MODEL=/path/to/Qwen3-VL-2B-Instruct
bash training/exps/run_omnidocbench_test.sh
```

Training outputs are written to:

```text
data_pipeline/outputs/omnidocbench_train_run/
```

## Evaluation

Use `data_pipeline/scripts/evaluate_table_recognition_vllm.py` to generate HTML on the evaluation JSONL and compute TEDS against `solution`.

Evaluate the base model:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -m data_pipeline.scripts.evaluate_table_recognition_vllm \
  --dataset data_pipeline/outputs/omnidocbench_train.jsonl \
  --model /path/to/Qwen3-VL-2B-Instruct \
  --output your-path-to.jsonl \
  --batch-size 16 \
  --max-new-tokens 2048 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.75
```

Evaluate a trained LoRA adapter:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -m data_pipeline.scripts.evaluate_table_recognition_vllm \
  --dataset data_pipeline/outputs/omnidocbench_train.jsonl \
  --model /path/to/Qwen3-VL-2B-Instruct \
  --adapter your-path-to \
  --output your-path-to.jsonl \
  --batch-size 16 \
  --max-new-tokens 2048 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.75
```

The evaluator writes one JSONL file with per-sample predictions and a sibling `.summary.json` file. If the JSONL already exists, evaluation resumes from the number of completed rows; use a new `--output` path for a clean rerun.

## Demo Results

The table below reports mean TEDS on the 187-sample OmniDocBench demo set generated by this pipeline.

| Model | mean TEDS |
| --- | ---: |
| Before training: Qwen3-VL-2B-Instruct | 0.501662 |
| After training: v5 checkpoint-69 LoRA | 0.616795 (+22.950%) |

This improvement shows that the pipeline can turn table-recognition data into QA-guided RL training data and improve table-structure generation quality under the same TEDS evaluation protocol.
