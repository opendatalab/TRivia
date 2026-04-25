from __future__ import annotations

from itertools import combinations

from ..config import PipelineConfig
from ..utils.io import read_jsonl, stage_file, write_jsonl
from ..utils.teds import teds_score


def pairwise_teds_average(samples: list[dict[str, str]]) -> tuple[float, list[dict[str, float | int]]]:
    scores = []
    pair_scores = []
    for left, right in combinations(samples, 2):
        score = teds_score(left["text"], right["text"])
        scores.append(score)
        pair_scores.append(
            {
                "left_sample_id": left["sample_id"],
                "right_sample_id": right["sample_id"],
                "teds": score,
            }
        )
    return sum(scores) / len(scores), pair_scores


def run_teds_filter(config: PipelineConfig, input_path: str | None = None) -> str:
    source_path = input_path or stage_file(config.work_dir, config.s1_dir, "s1.jsonl")
    rows = read_jsonl(source_path)
    kept_rows = []
    scored_rows = []

    for row in rows:
        average, pair_scores = pairwise_teds_average(row["samples"])
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

