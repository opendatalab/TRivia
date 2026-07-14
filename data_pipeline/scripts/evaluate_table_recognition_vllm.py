from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.utils.teds import normalize_table_text, teds_score


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_messages(row: dict) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": row["images"][0]},
                {"type": "text", "text": row["messages"][0]["content"]},
            ],
        }
    ]


def build_request(processor, row: dict) -> dict:
    messages = build_messages(row)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": image_inputs[0]},
    }


def score_prediction(prediction: str, reference: str) -> tuple[float, str | None]:
    try:
        return (
            teds_score(
                normalize_table_text(prediction),
                normalize_table_text(reference),
            ),
            None,
        )
    except (IndexError, AttributeError, ValueError) as exc:
        return 0.0, f"{type(exc).__name__}: {exc}"


def evaluate(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.dataset))
    if args.limit:
        rows = rows[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_rows = read_jsonl(output_path)
    scores = [row["teds"] for row in completed_rows]
    start_index = len(completed_rows)

    processor = AutoProcessor.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=bool(args.adapter),
        max_loras=1,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )
    lora_request = LoRARequest("trivia_adapter", 1, args.adapter) if args.adapter else None

    with output_path.open("a", encoding="utf-8") as f:
        pending_rows = rows[start_index:]
        progress = tqdm(total=len(rows), initial=start_index, desc="Evaluating")
        for batch_start in range(0, len(pending_rows), args.batch_size):
            batch = pending_rows[batch_start: batch_start + args.batch_size]
            requests = [build_request(processor, row) for row in batch]
            outputs = llm.generate(requests, sampling_params, lora_request=lora_request)
            for row, output in zip(batch, outputs):
                index = len(scores)
                prediction = output.outputs[0].text
                reference = row.get("solution") or row["html"]
                score, error = score_prediction(prediction, reference)
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
                progress.update(1)
        progress.close()

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "adapter": args.adapter,
        "count": len(scores),
        "mean_teds": sum(scores) / len(scores),
        "min_teds": min(scores),
        "max_teds": max(scores),
        "backend": "vllm",
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate table recognition outputs with vLLM and TEDS.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
