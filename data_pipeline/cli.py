from __future__ import annotations

import argparse

from .config import load_config
from .stages.s1_base_sampling import run_base_sampling
from .stages.s2_teds_filter import run_teds_filter
from .stages.s3_qa_generation import run_qa_generation
from .stages.s5_qa_evaluation import run_qa_evaluation
from .stages.s6_qa_filter import run_qa_filter
from .utils.io import read_jsonl, write_jsonl


def run_all(config_path: str | None) -> None:
    config = load_config(config_path)
    s1_path = run_base_sampling(config)
    s2_path = run_teds_filter(config, s1_path)
    s3_path = run_qa_generation(config, s2_path)
    if config.enable_qa_eval:
        s5_path = run_qa_evaluation(config, s3_path)
        run_qa_filter(config, s5_path)
    else:
        write_jsonl(config.output_jsonl, read_jsonl(s3_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone TRivia data generation pipeline")
    parser.add_argument(
        "command",
        choices=["run-all", "sample", "filter", "generate-qa", "evaluate-qa", "filter-qa"],
    )
    parser.add_argument("--config", default=None, help="Python config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.command == "run-all":
        run_all(args.config)
    elif args.command == "sample":
        run_base_sampling(config)
    elif args.command == "filter":
        run_teds_filter(config)
    elif args.command == "generate-qa":
        run_qa_generation(config)
    elif args.command == "evaluate-qa":
        run_qa_evaluation(config)
    elif args.command == "filter-qa":
        run_qa_filter(config)


if __name__ == "__main__":
    main()

