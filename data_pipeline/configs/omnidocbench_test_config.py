INPUT_JSONL = "data_pipeline/data/omnidocbench_tables.jsonl"
OUTPUT_JSONL = "data_pipeline/outputs/omnidocbench_pipeline_output.jsonl"
WORK_DIR = "data_pipeline/outputs/omnidocbench_work"

BASE_BACKEND = "openai_compatible"
BASE_OPENAI_URL = "http://localhost:8000/v1"
BASE_OPENAI_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
BASE_OPENAI_API_KEY = "EMPTY"

NUM_SAMPLES = 4
TEMPERATURE = 1.0
MAX_TOKENS = 4096
TOP_P = 0.95
OPENAI_CONCURRENCY = 32

TEDS_MIN = 0.5
TEDS_MAX = 0.9
TEDS_WORKERS = 8

QA_GEN_URL = "http://localhost:10000/v1"
QA_GEN_MODEL = "Qwen/Qwen3.5-9B"
QA_GEN_API_KEY = "EMPTY"
QA_GEN_PROMPT_FILE = "data_pipeline/prompts/qa_generation.txt"
QA_GEN_MAX_TOKENS = 2048
QA_GEN_TEMPERATURE = 0.0

ENABLE_QA_EVAL = False
QA_ANSWER_URL = "http://localhost:10000/v1"
QA_ANSWER_MODEL = "Qwen/Qwen3.5-9B"
QA_ANSWER_API_KEY = "EMPTY"
QA_ANSWER_PROMPT_FILE = "data_pipeline/prompts/qa_answer.txt"
QA_ANSWER_MAX_TOKENS = 512
QA_ANSWER_TEMPERATURE = 0.0

MIN_QA_COUNT = 3
ANSWER_F1_THRESHOLD = 0.8

GENERATION_CONFIG = {
    "stage_common": {
        "models": [
            {
                "name": BASE_OPENAI_MODEL,
                "request": {
                    "max_tokens": 4096,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "presence_penalty": 1.5,
                    "extra_body": {
                        "top_k": 20,
                        "min_p": 0.0,
                        "repetition_penalty": 1.0,
                        "enable_thinking": False,
                    },
                },
                "samples": 4,
            }
        ],
    },
    "qa_generation": {
        "models": [
            {
                "name": QA_GEN_MODEL,
                "request": {
                    "max_tokens": 2048,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "extra_body": {
                        "enable_thinking": False,
                    },
                },
                "samples": 1,
            }
        ],
    },
    "qa_evaluation": {
        "models": [
            {
                "name": QA_ANSWER_MODEL,
                "request": {
                    "max_tokens": 512,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "extra_body": {
                        "enable_thinking": False,
                    },
                },
                "samples": 1,
            }
        ],
    },
}
