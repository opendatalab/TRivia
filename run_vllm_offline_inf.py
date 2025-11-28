import json
import os
import re
import torch

from glob import glob
from PIL import Image
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoProcessor

from otsl_utils import otsl_to_html

SYSTEM_PROMPT_OTSL = (
    "You are an AI specialized in recognizing and extracting table from images. Your mission is to analyze the table image and generate the result in OTSL format using specified tags. Output only the results without any other words and explanation."
)

Image.MAX_IMAGE_PIXELS = None

def _process_single_path(p, model_type, tokenizer, processor, placeholder):
    image_path = p
    from qwen_vl_utils import process_vision_info
    messages = []
    if isinstance(SYSTEM_PROMPT_OTSL, str):
        messages = [
            {
                "role": "user", "content": [
                    {"type": "text", "text": SYSTEM_PROMPT_OTSL},
                    {
                        "type": "image", "image": Image.open(image_path).convert('RGB'),
                        "min_pixels": 256 * 28 * 28, "max_pixels": 3280 * 28 * 28,
                    }
                ]
            }
        ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    img_data, _ = process_vision_info(messages)
    return {
        "prompt": prompt,
        "multi_modal_data": {
            "image": img_data
        },
    }

def prepare_data(paths, ckpt_root):
    _inputs = []
    placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    processor = AutoProcessor.from_pretrained(ckpt_root)
    stop_token_ids = None

    from tqdm import tqdm
    import multiprocessing
    import concurrent.futures
    from functools import partial

    # 使用多进程处理每个path项，并使用 tqdm 显示进度
    with concurrent.futures.ThreadPoolExecutor(max_workers=96) as executor:
        func = partial(_process_single_path,
                    model_type=model_type,
                    tokenizer=tokenizer,
                    processor=processor,
                    placeholder=placeholder)
        
        # 使用 executor.map 来并行执行任务，并使用 tqdm 显示进度
        _inputs = list(tqdm(executor.map(func, paths), total=len(paths)))

    return _inputs, stop_token_ids

def run_ckpt(item):
    ckpt_root = item["ckpt_root"]
    image_root = item["image_root"]
    output_path = item["output_path"]
    if os.path.exists(output_path):
        print("Path Exisiting: " + output_path)

    from vllm import LLM, SamplingParams
    print("=" * 15 + "Preparing Data" + "=" * 15)
    if os.path.isfile(image_root):
        # 如果是文件，则读取文件内容，每行一个path
        paths = []
        with open(image_root, "r", encoding="utf-8") as f:
            paths = [line.strip() for line in f.readlines() if line.strip()]
    else:
        # 否则按目录读取图片
        paths = glob(image_root + "/*.jpg") + glob(image_root + "/*.png")
    _inputs, stop_token_ids = prepare_data_aug(paths, ckpt_root)
    print(_inputs[0])

    print("=" * 15 + "Preparing  LLM" + "=" * 15)
    llm = LLM(
        model=ckpt_root,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        # enforce_eager=True,
        gpu_memory_utilization=0.8,
        # max_num_seqs=4,
        limit_mm_per_prompt={"image": 1},
        seed=42
    )

    sampling_params = SamplingParams(temperature=0,
                                    max_tokens=8192,
                                    repetition_penalty=1.05,
                                    stop_token_ids=[])
    outputs = llm.generate(_inputs, sampling_params=sampling_params, use_tqdm=True)

    results = []
    for path, o in zip(paths, outputs):
        otsl_text = o.outputs[0].text
        html_text = otsl_to_html(otsl_text) if otsl_text else ""
        results.append({
            "path": path,
            "otsl": otsl_text,
            "html": html_text
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate model checkpoints')
    parser.add_argument('--ckpt_root', type=str, required=True,
                       help='Path to the model checkpoint directory')
    parser.add_argument('--image_root', type=str, required=True,
                       help='Path to the image folder or image list file')
    parser.add_argument('--output_path', type=str, default='./vllm_offline_output.json',
                       help='Path to the output prediction file')
    args = parser.parse_args()

    run_ckpt({
        "ckpt_root": args.ckpt_root,
        "image_root": args.image_root,
        "output_path": args.output_path
    })
