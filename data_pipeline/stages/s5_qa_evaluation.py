from __future__ import annotations

import asyncio
from typing import Any

from ..config import PipelineConfig
from ..utils.io import first_image, read_jsonl, stage_file, write_jsonl
from ..utils.openai_client import (
    async_chat_completion,
    make_async_client,
    text_user_message,
    vision_user_message,
)


EN_VISION_QA_PROMPT = """Given a table image and a corresponding question, your task is to respond appropriately based on table image. If the table do not contain the answer of question, output "Not answerable".
Your answer should be a short phrase of only few words. Output the answer within <answer> </answer>.
Question: {question}"""

ZH_VISION_QA_PROMPT = """给定一个表格图像以及一个相应的问题，你的任务是根据表格图片回答该问题。如果该表格不包含该问题的答案，请输出"无法回答"。你的答案必须简短、仅有一两个词语。输出答案时用<answer></answer>包裹。
问题: {question}"""

EN_TEXT_QA_PROMPT = """Answer the following question. Your answer should be a short phrase of only few words. If you cannot answer this question, output "Not answerable".
Your answer should be a short phrase of only few words. Output the answer within <answer> </answer>.
Question: {question}"""

ZH_TEXT_QA_PROMPT = """回答下面的问题。你的答案必须简短、仅有一两个词语。如果你无法回答该问题，请输出"无法回答"。你的答案必须简短、仅有一两个词语。输出答案时用<answer></answer>包裹。
问题: {question}"""


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _render_vision_prompt(qa: dict) -> str:
    template = ZH_VISION_QA_PROMPT if _contains_chinese(qa["question"]) else EN_VISION_QA_PROMPT
    return template.format(question=qa["question"])


def _render_text_prompt(qa: dict) -> str:
    template = ZH_TEXT_QA_PROMPT if _contains_chinese(qa["question"]) else EN_TEXT_QA_PROMPT
    return template.format(question=qa["question"])


async def _answer_qa(
    messages: list[dict],
    model_config: dict[str, Any],
    client: Any,
    semaphore: asyncio.Semaphore,
) -> str:
    async with semaphore:
        return (
            await async_chat_completion(
                client=client,
                model=model_config["name"],
                messages=messages,
                request=model_config["request"],
                samples=model_config["samples"],
            )
        )[0]


async def _evaluate_qa(
    row: dict,
    qa: dict,
    model_config: dict[str, Any],
    client: Any,
    semaphore: asyncio.Semaphore,
) -> dict:
    vision_prompt = _render_vision_prompt(qa)
    text_prompt = _render_text_prompt(qa)
    answer_with_image, answer_without_image = await asyncio.gather(
        _answer_qa(
            [vision_user_message(vision_prompt, first_image(row))],
            model_config,
            client,
            semaphore,
        ),
        _answer_qa([text_user_message(text_prompt)], model_config, client, semaphore),
    )
    evaluated_qa = dict(qa)
    evaluated_qa["answer_with_image"] = answer_with_image
    evaluated_qa["answer_without_image"] = answer_without_image
    return evaluated_qa


async def _evaluate_row(
    row: dict,
    model_config: dict[str, Any],
    client: Any,
    semaphore: asyncio.Semaphore,
) -> dict:
    evaluated_qas = await asyncio.gather(
        *(
            _evaluate_qa(row, qa, model_config, client, semaphore)
            for qa in row["QAs"]
        )
    )

    output_row = dict(row)
    output_row["QAs"] = evaluated_qas
    return output_row


async def _run_qa_evaluation_async(
    config: PipelineConfig, input_path: str | None = None
) -> str:
    source_path = input_path or stage_file(config.work_dir, config.s3_dir, "s3.jsonl")
    rows = read_jsonl(source_path)
    model_config = config.generation_config["qa_evaluation"]["models"][0]
    semaphore = asyncio.Semaphore(config.openai_concurrency)

    async with make_async_client(config.qa_answer_url, config.qa_answer_api_key) as client:
        output_rows = await asyncio.gather(
            *(
                _evaluate_row(row, model_config, client, semaphore)
                for row in rows
            )
        )

    output_path = stage_file(config.work_dir, config.s5_dir, "s5.jsonl")
    write_jsonl(output_path, output_rows)
    return output_path


def run_qa_evaluation(config: PipelineConfig, input_path: str | None = None) -> str:
    return asyncio.run(_run_qa_evaluation_async(config, input_path))

