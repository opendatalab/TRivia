from __future__ import annotations

import argparse
import json
from pathlib import Path


ASSISTANT_PLACEHOLDER = "<table></table>"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_training_rows(rows: list[dict]) -> list[dict]:
    output_rows = []
    for row in rows:
        output_rows.append(
            {
                "messages": row["messages"] + [{"role": "assistant", "content": ASSISTANT_PLACEHOLDER}],
                "solution": row["html"],
                "qa_pairs": row["QAs"],
                "images": row["images"],
            }
        )
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TRivia data pipeline output to ms-swift JSONL.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    rows = build_training_rows(read_jsonl(Path(args.input_jsonl)))
    write_jsonl(Path(args.output_jsonl), rows)
    print(f"Wrote {len(rows)} training rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
