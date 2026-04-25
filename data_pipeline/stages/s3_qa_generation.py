from __future__ import annotations

import json

from ..config import PipelineConfig
from ..utils.io import first_image, read_jsonl, read_text, stage_file, write_jsonl
from ..utils.openai_client import chat_completion, make_client, vision_user_message


def _render_prompt(template: str, row: dict) -> str:
    return template.format(
        image_path=first_image(row),
        messages=row["messages"],
        samples=row.get("samples", []),
        teds_average=row.get("teds_average"),
    )


def run_qa_generation(config: PipelineConfig, input_path: str | None = None) -> str:
    source_path = input_path or stage_file(config.work_dir, config.s2_dir, "s2.jsonl")
    rows = read_jsonl(source_path)
    prompt_template = read_text(config.qa_gen_prompt_file)
    client = make_client(config.qa_gen_url, config.qa_gen_api_key)

    output_rows = []
    for row in rows:
        prompt = _render_prompt(prompt_template, row)
        responses = chat_completion(
            client=client,
            model=config.qa_gen_model,
            messages=[vision_user_message(prompt, first_image(row))],
            temperature=config.qa_gen_temperature,
            max_tokens=config.qa_gen_max_tokens,
        )
        output_row = dict(row)
        output_row["qa_generation_raw"] = responses[0]
        output_row["QAs"] = json.loads(responses[0])
        output_rows.append(output_row)

    output_path = stage_file(config.work_dir, config.s3_dir, "s3.jsonl")
    write_jsonl(output_path, output_rows)
    return output_path

