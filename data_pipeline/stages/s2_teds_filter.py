from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

from ..config import PipelineConfig
from ..utils.io import read_jsonl, stage_file, write_jsonl
from ..utils.teds import normalize_table_text, teds_score


TedsTask = tuple[int, str, str, str, str]
TedsResult = tuple[int, dict[str, float | str]]


def _score_teds_pair(task: TedsTask) -> TedsResult:
    row_index, left_sample_id, right_sample_id, left_text, right_text = task
    score = teds_score(normalize_table_text(left_text), normalize_table_text(right_text))
    return (
        row_index,
        {
            "left_sample_id": left_sample_id,
            "right_sample_id": right_sample_id,
            "teds": score,
        },
    )


def _collect_teds_tasks(rows: list[dict]) -> list[TedsTask]:
    tasks = []
    for row_index, row in enumerate(rows):
        for left, right in combinations(row["samples"], 2):
            tasks.append(
                (
                    row_index,
                    left["sample_id"],
                    right["sample_id"],
                    left["text"],
                    right["text"],
                )
            )
    return tasks


def run_teds_filter(config: PipelineConfig, input_path: str | None = None) -> str:
    source_path = input_path or stage_file(config.work_dir, config.s1_dir, "s1.jsonl")
    rows = read_jsonl(source_path)
    pair_scores_by_row = [[] for _ in rows]
    tasks = _collect_teds_tasks(rows)

    with ProcessPoolExecutor(max_workers=config.teds_workers) as executor:
        for row_index, pair_score in executor.map(_score_teds_pair, tasks):
            pair_scores_by_row[row_index].append(pair_score)

    kept_rows = []
    scored_rows = []
    for row, pair_scores in zip(rows, pair_scores_by_row, strict=True):
        average = sum(pair_score["teds"] for pair_score in pair_scores) / len(pair_scores)
        scored_row = dict(row)
        scored_row["teds_average"] = average
        scored_row["teds_pairs"] = pair_scores
        scored_row["teds_kept"] = config.teds_min <= average <= config.teds_max
        scored_rows.append(scored_row)
        if scored_row["teds_kept"]:
            kept_rows.append(scored_row)

    scores_path = stage_file(config.work_dir, config.s2_dir, "s2_scores.jsonl")
    kept_path = stage_file(config.work_dir, config.s2_dir, "s2.jsonl")
    write_jsonl(scores_path, scored_rows)
    write_jsonl(kept_path, kept_rows)
    return kept_path

