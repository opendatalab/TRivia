from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data_pipeline.config import load_config
from data_pipeline.stages.s2_teds_filter import run_teds_filter
from data_pipeline.stages.s6_qa_filter import run_qa_filter
from data_pipeline.utils.io import read_jsonl, write_jsonl
from data_pipeline.utils.teds import teds_score


class PipelineTest(unittest.TestCase):
    def test_load_config_from_python_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.py"
            config_path.write_text(
                "\n".join(
                    [
                        'INPUT_JSONL = "input.jsonl"',
                        'OUTPUT_JSONL = "output.jsonl"',
                        'WORK_DIR = "work"',
                        'BASE_BACKEND = "openai_compatible"',
                        "NUM_SAMPLES = 4",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(str(config_path))

        self.assertEqual(config.input_jsonl, "input.jsonl")
        self.assertEqual(config.output_jsonl, "output.jsonl")
        self.assertEqual(config.work_dir, "work")
        self.assertEqual(config.base_backend, "openai_compatible")
        self.assertEqual(config.num_samples, 4)

    def test_teds_identical_tables_score_one(self) -> None:
        html = "<table><tr><td>A</td><td>B</td></tr></table>"

        self.assertEqual(teds_score(html, html), 1.0)

    def test_teds_filter_keeps_score_in_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            s1_path = tmp_path / "s1.jsonl"
            config_path = tmp_path / "config.py"
            work_dir = tmp_path / "work"
            table = "<table><tr><td>A</td></tr></table>"
            write_jsonl(
                str(s1_path),
                [
                    {
                        "images": ["image.png"],
                        "messages": "prompt",
                        "samples": [
                            {"sample_id": 0, "text": table},
                            {"sample_id": 1, "text": table},
                            {"sample_id": 2, "text": table},
                            {"sample_id": 3, "text": table},
                        ],
                    }
                ],
            )
            config_path.write_text(
                f'WORK_DIR = "{work_dir}"\nTEDS_MIN = 1.0\nTEDS_MAX = 1.0\n',
                encoding="utf-8",
            )

            output_path = run_teds_filter(load_config(str(config_path)), str(s1_path))

            rows = read_jsonl(output_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["teds_average"], 1.0)

    def test_qa_filter_keeps_visual_only_qas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            s5_path = tmp_path / "s5.jsonl"
            config_path = tmp_path / "config.py"
            work_dir = tmp_path / "work"
            output_path = tmp_path / "output.jsonl"
            qas = [
                {
                    "question": f"question {index}",
                    "answer": "42",
                    "answer_with_image": "42",
                    "answer_without_image": "unknown",
                }
                for index in range(3)
            ]
            write_jsonl(str(s5_path), [{"images": ["image.png"], "messages": "prompt", "QAs": qas}])
            config_path.write_text(
                "\n".join(
                    [
                        f'WORK_DIR = "{work_dir}"',
                        f'OUTPUT_JSONL = "{output_path}"',
                        "MIN_QA_COUNT = 3",
                        "ANSWER_F1_THRESHOLD = 0.8",
                    ]
                ),
                encoding="utf-8",
            )

            run_qa_filter(load_config(str(config_path)), str(s5_path))

            rows = read_jsonl(str(output_path))
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["QAs"]), 3)


if __name__ == "__main__":
    unittest.main()

