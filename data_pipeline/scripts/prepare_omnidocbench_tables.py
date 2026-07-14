from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image


DATASET_REPO = "opendatalab/OmniDocBench"
DATASET_BASE_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main"
ANNOTATION_FILE = "OmniDocBench.json"
TABLE_PROMPT = (
    "You are an AI specialized in recognizing and extracting table from images. "
    "Your mission is to analyze the table image and generate the result in HTML format "
    "using specified tags. Output only the results without any other words and explanation."
)


def download_file(url: str, output_path: Path) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def hf_file_url(path: str) -> str:
    return f"{DATASET_BASE_URL}/{quote(path)}"


def poly_to_bbox(poly: list[float], width: int, height: int, padding: int) -> tuple[int, int, int, int]:
    xs = poly[0::2]
    ys = poly[1::2]
    left = max(0, math.floor(min(xs)) - padding)
    top = max(0, math.floor(min(ys)) - padding)
    right = min(width, math.ceil(max(xs)) + padding)
    bottom = min(height, math.ceil(max(ys)) + padding)
    return left, top, right, bottom


def page_image_path(sample: dict) -> str:
    image_path = sample["page_info"]["image_path"]
    if image_path.startswith("images/"):
        return image_path
    return f"images/{Path(image_path).name}"


def build_row(crop_path: Path, html: str, latex: str | None, metadata: dict) -> dict:
    return {
        "messages": [{"role": "user", "content": TABLE_PROMPT}],
        "images": [str(crop_path.resolve())],
        "html": html,
        "latex": latex,
        "source": metadata,
    }


def prepare_tables(
    output_jsonl: Path,
    image_dir: Path,
    raw_dir: Path,
    max_tables: int | None,
    padding: int,
) -> int:
    annotation_path = raw_dir / ANNOTATION_FILE
    download_file(hf_file_url(ANNOTATION_FILE), annotation_path)
    samples = json.loads(annotation_path.read_text(encoding="utf-8"))

    raw_image_dir = raw_dir / "images"
    rows = []
    for sample_index, sample in enumerate(samples):
        page_path = page_image_path(sample)
        local_page_path = raw_image_dir / Path(page_path).name
        download_file(hf_file_url(page_path), local_page_path)

        with Image.open(local_page_path) as page_image:
            page_image = page_image.convert("RGB")
            for anno_index, annotation in enumerate(sample["layout_dets"]):
                if annotation["category_type"] != "table":
                    continue

                crop_name = f"{Path(page_path).stem}_table_{anno_index:04d}.png"
                crop_path = image_dir / crop_name
                bbox = poly_to_bbox(annotation["poly"], page_image.width, page_image.height, padding)
                page_image.crop(bbox).save(crop_path)

                metadata = {
                    "dataset": DATASET_REPO,
                    "sample_index": sample_index,
                    "anno_index": anno_index,
                    "page_image": page_path,
                    "bbox": list(bbox),
                    "attribute": annotation.get("attribute", {}),
                }
                rows.append(build_row(crop_path, annotation["html"], annotation.get("latex"), metadata))
                if max_tables is not None and len(rows) >= max_tables:
                    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
                    write_jsonl(output_jsonl, rows)
                    return len(rows)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, rows)
    return len(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OmniDocBench table crops for TRivia.")
    parser.add_argument("--output-jsonl", default="data_pipeline/data/omnidocbench_tables.jsonl")
    parser.add_argument("--image-dir", default="data_pipeline/data/images")
    parser.add_argument("--raw-dir", default="data_pipeline/data/raw/omnidocbench")
    parser.add_argument("--max-tables", type=int, default=None)
    parser.add_argument("--padding", type=int, default=2)
    args = parser.parse_args()

    count = prepare_tables(
        output_jsonl=Path(args.output_jsonl),
        image_dir=Path(args.image_dir),
        raw_dir=Path(args.raw_dir),
        max_tables=args.max_tables,
        padding=args.padding,
    )
    print(f"Prepared {count} table crops at {args.output_jsonl}")


if __name__ == "__main__":
    main()
