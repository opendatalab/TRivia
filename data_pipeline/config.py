"""Configuration for the standalone TRivia data generation pipeline.

Edit the module-level values in this file or pass another Python config file
with the same variable names to ``python -m data_pipeline.cli --config``.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, fields
from pathlib import Path


INPUT_JSONL = "input.jsonl"
OUTPUT_JSONL = "output.jsonl"
WORK_DIR = "outputs/trivia_pipeline"

BASE_BACKEND = "vllm_offline"
BASE_MODEL = "/path/to/base_model"
BASE_OPENAI_URL = "http://localhost:8000/v1"
BASE_OPENAI_MODEL = "base-model"
BASE_OPENAI_API_KEY = "EMPTY"
NUM_SAMPLES = 4
TEMPERATURE = 1.0
MAX_TOKENS = 8192
TOP_P = 1.0

VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.8
VLLM_TRUST_REMOTE_CODE = True
VLLM_USE_PROCESSOR_CHAT_TEMPLATE = False

TEDS_MIN = 0.5
TEDS_MAX = 0.9

QA_GEN_URL = "http://localhost:10000/v1"
QA_GEN_MODEL = "Qwen/Qwen3-8B"
QA_GEN_API_KEY = "EMPTY"
QA_GEN_PROMPT_FILE = "prompts/qa_generation.txt"
QA_GEN_MAX_TOKENS = 2048
QA_GEN_TEMPERATURE = 0.0

ENABLE_QA_EVAL = False
QA_ANSWER_URL = "http://localhost:10001/v1"
QA_ANSWER_MODEL = "Qwen/Qwen3-8B"
QA_ANSWER_API_KEY = "EMPTY"
QA_ANSWER_PROMPT_FILE = "prompts/qa_answer.txt"
QA_ANSWER_MAX_TOKENS = 512
QA_ANSWER_TEMPERATURE = 0.0

MIN_QA_COUNT = 3
ANSWER_F1_THRESHOLD = 0.8

S1_DIR = "s1_base_sampling"
S2_DIR = "s2_teds_filter"
S3_DIR = "s3_qa_generation"
S5_DIR = "s5_qa_evaluation"
S6_DIR = "s6_qa_filter"


@dataclass(frozen=True)
class PipelineConfig:
    input_jsonl: str = INPUT_JSONL
    output_jsonl: str = OUTPUT_JSONL
    work_dir: str = WORK_DIR

    base_backend: str = BASE_BACKEND
    base_model: str = BASE_MODEL
    base_openai_url: str = BASE_OPENAI_URL
    base_openai_model: str = BASE_OPENAI_MODEL
    base_openai_api_key: str = BASE_OPENAI_API_KEY
    num_samples: int = NUM_SAMPLES
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS
    top_p: float = TOP_P

    vllm_tensor_parallel_size: int = VLLM_TENSOR_PARALLEL_SIZE
    vllm_gpu_memory_utilization: float = VLLM_GPU_MEMORY_UTILIZATION
    vllm_trust_remote_code: bool = VLLM_TRUST_REMOTE_CODE
    vllm_use_processor_chat_template: bool = VLLM_USE_PROCESSOR_CHAT_TEMPLATE

    teds_min: float = TEDS_MIN
    teds_max: float = TEDS_MAX

    qa_gen_url: str = QA_GEN_URL
    qa_gen_model: str = QA_GEN_MODEL
    qa_gen_api_key: str = QA_GEN_API_KEY
    qa_gen_prompt_file: str = QA_GEN_PROMPT_FILE
    qa_gen_max_tokens: int = QA_GEN_MAX_TOKENS
    qa_gen_temperature: float = QA_GEN_TEMPERATURE

    enable_qa_eval: bool = ENABLE_QA_EVAL
    qa_answer_url: str = QA_ANSWER_URL
    qa_answer_model: str = QA_ANSWER_MODEL
    qa_answer_api_key: str = QA_ANSWER_API_KEY
    qa_answer_prompt_file: str = QA_ANSWER_PROMPT_FILE
    qa_answer_max_tokens: int = QA_ANSWER_MAX_TOKENS
    qa_answer_temperature: float = QA_ANSWER_TEMPERATURE

    min_qa_count: int = MIN_QA_COUNT
    answer_f1_threshold: float = ANSWER_F1_THRESHOLD

    s1_dir: str = S1_DIR
    s2_dir: str = S2_DIR
    s3_dir: str = S3_DIR
    s5_dir: str = S5_DIR
    s6_dir: str = S6_DIR


def _load_module(path: str):
    config_path = Path(path)
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(path: str | None = None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()

    module = _load_module(path)
    values = {}
    for field in fields(PipelineConfig):
        key = field.name.upper()
        if hasattr(module, key):
            values[field.name] = getattr(module, key)
    return PipelineConfig(**values)

