from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.utils.teds import normalize_table_text, teds_score


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(row: dict) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row["images"][0]},
                {"type": "text", "text": row["messages"][0]["content"]},
            ],
        }
    ]


def generate_table(model, processor, row: dict, max_new_tokens: int) -> str:
    messages = build_prompt(row)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def load_model(model_path: str, adapter_path: str | None):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def evaluate(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.dataset))
    if args.limit:
        rows = rows[: args.limit]

    model = load_model(args.model, args.adapter)
    processor = AutoProcessor.from_pretrained(args.model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_rows = read_jsonl(output_path) if output_path.exists() else []
    scores = [row["teds"] for row in completed_rows]
    start_index = len(completed_rows)
    with output_path.open("a", encoding="utf-8") as f:
        for index, row in enumerate(tqdm(rows[start_index:], desc="Evaluating", initial=start_index, total=len(rows)), start=start_index):
            prediction = generate_table(model, processor, row, args.max_new_tokens)
            reference = row.get("solution") or row["html"]
            error = None
            try:
                score = teds_score(
                    normalize_table_text(prediction),
                    normalize_table_text(reference),
                )
            except (IndexError, AttributeError, ValueError) as exc:
                score = 0.0
                error = f"{type(exc).__name__}: {exc}"
            scores.append(score)
            f.write(
                json.dumps(
                    {
                        "index": index,
                        "image": row["images"][0],
                        "prediction": prediction,
                        "reference": reference,
                        "teds": score,
                        "error": error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "adapter": args.adapter,
        "count": len(scores),
        "mean_teds": sum(scores) / len(scores),
        "min_teds": min(scores),
        "max_teds": max(scores),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate table recognition outputs with TEDS.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
