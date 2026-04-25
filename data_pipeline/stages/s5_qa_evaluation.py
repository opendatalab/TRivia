from __future__ import annotations

from ..config import PipelineConfig
from ..utils.io import first_image, read_jsonl, read_text, stage_file, write_jsonl
from ..utils.openai_client import chat_completion, make_client, text_user_message, vision_user_message


def _render_prompt(template: str, row: dict, qa: dict) -> str:
    return template.format(
        image_path=first_image(row),
        question=qa["question"],
        answer=qa["answer"],
        qa=qa,
    )


def run_qa_evaluation(config: PipelineConfig, input_path: str | None = None) -> str:
    source_path = input_path or stage_file(config.work_dir, config.s3_dir, "s3.jsonl")
    rows = read_jsonl(source_path)
    prompt_template = read_text(config.qa_answer_prompt_file)
    client = make_client(config.qa_answer_url, config.qa_answer_api_key)

    output_rows = []
    for row in rows:
        evaluated_qas = []
        for qa in row["QAs"]:
            prompt = _render_prompt(prompt_template, row, qa)
            answer_with_image = chat_completion(
                client=client,
                model=config.qa_answer_model,
                messages=[vision_user_message(prompt, first_image(row))],
                temperature=config.qa_answer_temperature,
                max_tokens=config.qa_answer_max_tokens,
            )[0]
            answer_without_image = chat_completion(
                client=client,
                model=config.qa_answer_model,
                messages=[text_user_message(prompt)],
                temperature=config.qa_answer_temperature,
                max_tokens=config.qa_answer_max_tokens,
            )[0]
            evaluated_qa = dict(qa)
            evaluated_qa["answer_with_image"] = answer_with_image
            evaluated_qa["answer_without_image"] = answer_without_image
            evaluated_qas.append(evaluated_qa)

        output_row = dict(row)
        output_row["QAs"] = evaluated_qas
        output_rows.append(output_row)

    output_path = stage_file(config.work_dir, config.s5_dir, "s5.jsonl")
    write_jsonl(output_path, output_rows)
    return output_path

