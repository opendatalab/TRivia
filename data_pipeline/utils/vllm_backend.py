from __future__ import annotations

from typing import Any

from .io import first_image, message_text


def _raw_vllm_input(row: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    return {
        "prompt": message_text(row["messages"]),
        "multi_modal_data": {"image": Image.open(first_image(row)).convert("RGB")},
    }


def _processor_vllm_input(row: dict[str, Any], processor: Any) -> dict[str, Any]:
    from PIL import Image

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": message_text(row["messages"])},
                {"type": "image", "image": Image.open(first_image(row)).convert("RGB")},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": messages[0]["content"][1]["image"]},
    }


def generate_vllm_offline(rows: list[dict[str, Any]], config: Any) -> list[list[str]]:
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = None
    if config.vllm_use_processor_chat_template:
        processor = AutoProcessor.from_pretrained(config.base_model)

    inputs = [
        _processor_vllm_input(row, processor) if processor is not None else _raw_vllm_input(row)
        for row in rows
    ]
    llm = LLM(
        model=config.base_model,
        tensor_parallel_size=config.vllm_tensor_parallel_size,
        trust_remote_code=config.vllm_trust_remote_code,
        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=config.num_samples,
    )
    outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=True)
    return [[completion.text for completion in output.outputs] for output in outputs]

