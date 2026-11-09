"""
Download and preprocess Vision-OPD-6K training data from HuggingFace.

Usage:
    python scripts/prepare_data.py --data-dir ./data

This script:
1. Downloads train.jsonl from yuanqianhao/Vision-OPD-6K
2. Downloads and extracts the original (box-free) images
3. Converts train.jsonl to the parquet format expected by the training pipeline
   (box-free: original images, red-box hint stripped from the question)
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Any

import datasets
from huggingface_hub import snapshot_download


REMOVE_HINT = (
    "Only focus on the objects inside the red bounding box in the image "
    "to answer this question."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Vision-OPD-6K training data.")
    parser.add_argument("--data-dir", default="./data", help="Output directory for processed data")
    parser.add_argument("--hf-repo", default="yuanqianhao/Vision-OPD-6K", help="HuggingFace dataset repo")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading, only preprocess")
    return parser.parse_args()


def download_dataset(repo_id: str, data_dir: str) -> None:
    print(f"Downloading dataset from {repo_id} ...")
    # Box-free training only needs the original images; the red-box student and teacher image
    # sets in the same repo add ~31 GB that would never be read.
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=data_dir,
        allow_patterns=["train.jsonl", "original_images/*"],
    )

    # We train box-free: both views come from the ORIGINAL image (no bounding box drawn).
    originals_dir = os.path.join(data_dir, "original_images")
    originals_tar = os.path.join(originals_dir, "original_images.tar.gz")
    if os.path.exists(originals_tar):
        print("Extracting original images ...")
        subprocess.run(["tar", "-xzf", "original_images.tar.gz", "-C", "."], cwd=originals_dir, check=True)
        os.remove(originals_tar)

    print("Image extraction complete.")


def clean_question(problem: str) -> str:
    text = (problem or "").replace("<image>", "").strip()
    text = text.replace(f"\n\n{REMOVE_HINT}", "")
    text = text.replace(REMOVE_HINT, "")
    return text.strip()


def build_record(item: dict[str, Any], data_dir: str) -> dict[str, Any]:
    # Box-free setting: the image is the original one (no bounding box drawn) and the
    # red-box hint is stripped from the question. The dataset wrapper augments this image
    # for the student and keeps a clean copy for the teacher (`weak_images`).
    image_path = os.path.join(data_dir, item["original_images"][0])
    question = clean_question(item.get("problem", ""))

    return {
        "data_source": "vision_opd_6k",
        "prompt": [{"role": "user", "content": f"<image> {question}"}],
        "images": [{"path": image_path}],
        "ability": "visual_question_answering",
        "reward_model": {
            "style": "none",
            "ground_truth": item.get("answer", ""),
        },
        "extra_info": {
            "answer": item.get("answer", ""),
            "question": question,
            "source_extra_info": item.get("extra_info", {}),
        },
    }


def convert_to_parquet(data_dir: str) -> None:
    jsonl_path = os.path.join(data_dir, "train.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found", file=sys.stderr)
        sys.exit(1)

    print("Converting train.jsonl to train.parquet ...")
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            records.append(build_record(item, data_dir))

    dataset = datasets.Dataset.from_list(records)
    output_path = os.path.join(data_dir, "train.parquet")
    dataset.to_parquet(output_path)
    print(f"Saved {len(records)} records to {output_path}")


def main() -> None:
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    if not args.skip_download:
        download_dataset(args.hf_repo, data_dir)

    convert_to_parquet(data_dir)
    print(f"\nData preparation complete. Training data at: {data_dir}/train.parquet")


if __name__ == "__main__":
    main()
