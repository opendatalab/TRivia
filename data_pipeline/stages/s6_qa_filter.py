from __future__ import annotations

import re
from collections import Counter

from ..config import PipelineConfig
from ..utils.io import read_jsonl, stage_file, write_jsonl


def normalize_answer(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction)
    ref_tokens = normalize_answer(reference)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_is_correct(prediction: str, reference: str, threshold: float) -> bool:
    return token_f1(prediction, reference) >= threshold


def keep_qa(qa: dict, config: PipelineConfig) -> bool:
    with_image_correct = answer_is_correct(
        qa["answer_with_image"],
        qa["answer"],
        config.answer_f1_threshold,
    )
    without_image_correct = answer_is_correct(
        qa["answer_without_image"],
        qa["answer"],
        config.answer_f1_threshold,
    )
    return with_image_correct and not without_image_correct


def run_qa_filter(config: PipelineConfig, input_path: str | None = None) -> str:
    source_path = input_path or stage_file(config.work_dir, config.s5_dir, "s5.jsonl")
    rows = read_jsonl(source_path)
    output_rows = []

    for row in rows:
        kept_qas = [qa for qa in row["QAs"] if keep_qa(qa, config)]
        if len(kept_qas) >= config.min_qa_count:
            output_row = dict(row)
            output_row["QAs"] = kept_qas
            output_rows.append(output_row)

    stage_output_path = stage_file(config.work_dir, config.s6_dir, "s6.jsonl")
    write_jsonl(stage_output_path, output_rows)
    write_jsonl(config.output_jsonl, output_rows)
    return stage_output_path

