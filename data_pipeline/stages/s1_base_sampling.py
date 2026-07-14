from __future__ import annotations

import asyncio
import json
from typing import Any

from ..config import PipelineConfig
from ..utils.io import read_jsonl, stage_file, write_jsonl
from ..utils.openai_client import (
    async_chat_completion,
    make_async_client,
    row_to_openai_messages,
)
from ..utils.vllm_backend import generate_vllm_offline


async def _generate_openai_row(
    row: dict[str, Any],
    model_config: dict[str, Any],
    client: Any,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        row_generations = await async_chat_completion(
            client=client,
            model=model_config["name"],
            messages=row_to_openai_messages(row),
            request=model_config["request"],
            samples=model_config["samples"],
        )
    return _make_output_row(row, row_generations)


def _make_output_row(row: dict[str, Any], row_generations: list[str]) -> dict[str, Any]:
    output_row = dict(row)
    output_row["samples"] = [
        {"sample_id": index, "text": text}
        for index, text in enumerate(row_generations)
    ]
    return output_row


async def _generate_openai_async(
    rows: list[dict[str, Any]], config: PipelineConfig, output_path: str
) -> None:
    model_config = config.generation_config["stage_common"]["models"][0]
    semaphore = asyncio.Semaphore(config.openai_concurrency)
    with open(output_path, "w", encoding="utf-8") as f:
        async with make_async_client(config.base_openai_url, config.base_openai_api_key) as client:
            tasks = [
                asyncio.create_task(_generate_openai_row(row, model_config, client, semaphore))
                for row in rows
            ]
            for task in asyncio.as_completed(tasks):
                output_row = await task
                f.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                f.flush()


def _generate_openai(rows: list[dict[str, Any]], config: PipelineConfig, output_path: str) -> None:
    asyncio.run(_generate_openai_async(rows, config, output_path))


def run_base_sampling(config: PipelineConfig) -> str:
    rows = read_jsonl(config.input_jsonl)
    output_path = stage_file(config.work_dir, config.s1_dir, "s1.jsonl")
    if config.base_backend == "vllm_offline":
        generations = generate_vllm_offline(rows, config)
        output_rows = [
            _make_output_row(row, row_generations)
            for row, row_generations in zip(rows, generations)
        ]
        write_jsonl(output_path, output_rows)
    elif config.base_backend == "openai_compatible":
        _generate_openai(rows, config, output_path)
    else:
        raise ValueError(f"Unknown base backend: {config.base_backend}")

    return output_path

