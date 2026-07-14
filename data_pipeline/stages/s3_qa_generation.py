from __future__ import annotations

import asyncio
import ast
import json
import re
from typing import Any

from ..config import PipelineConfig
from ..utils.io import first_image, read_jsonl, read_text, stage_file, write_jsonl
from ..utils.openai_client import async_chat_completion, make_async_client, vision_user_message


def _render_prompt(template: str, row: dict) -> str:
    return template.format(
        image_path=first_image(row),
        messages=row["messages"],
        samples=row.get("samples", []),
        teds_average=row.get("teds_average"),
    )


def _parse_qa_response(text: str) -> list[dict]:
    stripped = text.strip()
    if stripped == "None":
        return []
    if "[" not in stripped:
        return []
    start = stripped.index("[")
    end = stripped.rindex("]") + 1
    array_text = stripped[start:end].translate(
        str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
    )
    try:
        return json.loads(array_text)
    except json.JSONDecodeError:
        pairs = re.findall(
            r'\{\s*"question"\s*:\s*"(.*?)"\s*,\s*"answer"\s*:\s*"(.*?)"\s*\}',
            array_text,
            flags=re.DOTALL,
        )
        if pairs:
            return [{"question": question, "answer": answer} for question, answer in pairs]
        return ast.literal_eval(array_text)


async def _generate_row_qa(
    row: dict,
    prompt_template: str,
    model_config: dict[str, Any],
    client: Any,
    semaphore: asyncio.Semaphore,
) -> dict:
    prompt = _render_prompt(prompt_template, row)
    async with semaphore:
        responses = await async_chat_completion(
            client=client,
            model=model_config["name"],
            messages=[vision_user_message(prompt, first_image(row))],
            request=model_config["request"],
            samples=model_config["samples"],
        )
    output_row = dict(row)
    output_row["qa_generation_raw"] = responses[0]
    output_row["QAs"] = _parse_qa_response(responses[0])
    return output_row


async def _run_qa_generation_async(
    config: PipelineConfig, input_path: str | None = None
) -> str:
    source_path = input_path or stage_file(config.work_dir, config.s2_dir, "s2.jsonl")
    rows = read_jsonl(source_path)
    prompt_template = read_text(config.qa_gen_prompt_file)
    model_config = config.generation_config["qa_generation"]["models"][0]
    semaphore = asyncio.Semaphore(config.openai_concurrency)

    async with make_async_client(config.qa_gen_url, config.qa_gen_api_key) as client:
        output_rows = await asyncio.gather(
            *(
                _generate_row_qa(row, prompt_template, model_config, client, semaphore)
                for row in rows
            )
        )

    output_path = stage_file(config.work_dir, config.s3_dir, "s3.jsonl")
    write_jsonl(output_path, output_rows)
    return output_path


def run_qa_generation(config: PipelineConfig, input_path: str | None = None) -> str:
    return asyncio.run(_run_qa_generation_async(config, input_path))

