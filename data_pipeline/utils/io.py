from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def stage_dir(work_dir: str, name: str) -> str:
    path = os.path.join(work_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def stage_file(work_dir: str, name: str, filename: str) -> str:
    return os.path.join(stage_dir(work_dir, name), filename)


def first_image(row: dict[str, Any]) -> str:
    return row["images"][0]


def message_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    parts: list[str] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            parts.append(content)
        else:
            for item in content:
                if item.get("type") == "text":
                    parts.append(item["text"])
    return "\n".join(parts)


def with_suffix(path: str, suffix: str) -> str:
    source = Path(path)
    return str(source.with_name(source.stem + suffix + source.suffix))

