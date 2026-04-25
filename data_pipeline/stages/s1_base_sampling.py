from __future__ import annotations

from typing import Any

from ..config import PipelineConfig
from ..utils.io import read_jsonl, stage_file, write_jsonl
from ..utils.openai_client import chat_completion, make_client, row_to_openai_messages
from ..utils.vllm_backend import generate_vllm_offline


def _generate_openai(rows: list[dict[str, Any]], config: PipelineConfig) -> list[list[str]]:
    client = make_client(config.base_openai_url, config.base_openai_api_key)
    generations = []
    for row in rows:
        generations.append(
            chat_completion(
                client=client,
                model=config.base_openai_model,
                messages=row_to_openai_messages(row),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                n=config.num_samples,
            )
        )
    return generations


def run_base_sampling(config: PipelineConfig) -> str:
    rows = read_jsonl(config.input_jsonl)
    if config.base_backend == "vllm_offline":
        generations = generate_vllm_offline(rows, config)
    elif config.base_backend == "openai_compatible":
        generations = _generate_openai(rows, config)
    else:
        raise ValueError(f"Unknown base backend: {config.base_backend}")

    output_rows = []
    for row, row_generations in zip(rows, generations):
        output_row = dict(row)
        output_row["samples"] = [
            {"sample_id": index, "text": text}
            for index, text in enumerate(row_generations)
        ]
        output_rows.append(output_row)

    output_path = stage_file(config.work_dir, config.s1_dir, "s1.jsonl")
    write_jsonl(output_path, output_rows)
    return output_path

