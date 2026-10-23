"""
scripts/download_data.py

Automated dataset acquisition script for the Jigsaw Toxic Comment Classification Challenge.
Downloads verified dataset splits from Hugging Face mirror:
  https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments

Dataset schema:
  - id: Unique comment identifier
  - comment_text: The raw text of the comment
  - toxic: 0 or 1
  - severe_toxic: 0 or 1
  - obscene: 0 or 1
  - threat: 0 or 1
  - insult: 0 or 1
  - identity_hate: 0 or 1
"""

import os
import sys
import gzip
import shutil
import argparse
import urllib.request
import pandas as pd

HF_BASE_URL = "https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments/resolve/main"
FILES_TO_DOWNLOAD = {
    "train.csv.gz": "train.csv",
    "test.csv.gz": "test.csv",
    "test_labels.csv.gz": "test_labels.csv",
}

EXPECTED_TRAIN_COLUMNS = [
    "id", "comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
]


def download_file(url: str, dest_path: str, chunk_size: int = 1024 * 1024):
    """Download a remote file with progress reporting."""
    print(f"Connecting to {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "ToxicCommentProject/1.0"})
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f_out:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\r  Downloaded: {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent:.1f}%)", end="", flush=True)
                else:
                    print(f"\r  Downloaded: {downloaded / (1024 * 1024):.1f} MB", end="", flush=True)
    print()


def decompress_gzip(gz_path: str, out_path: str):
    """Decompress a .gz file to the destination path."""
    print(f"Decompressing {gz_path} -> {out_path} ...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Successfully extracted {os.path.basename(out_path)} ({os.path.getsize(out_path) / (1024 * 1024):.1f} MB)")


def validate_dataset(csv_path: str, expected_cols: list):
    """Validate file existence, non-emptiness, and required columns."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected file {csv_path} does not exist.")

    # Read header and first 5 rows
    df_sample = pd.read_csv(csv_path, nrows=5)
    missing_cols = [c for c in expected_cols if c not in df_sample.columns]
    if missing_cols:
        raise ValueError(f"File {csv_path} is missing expected columns: {missing_cols}")

    # Check total rows
    total_rows = sum(1 for _ in open(csv_path, "r", encoding="utf-8", errors="ignore")) - 1
    print(f"Validation PASSED for {os.path.basename(csv_path)}: {total_rows:,} records with all expected columns {expected_cols}.")
    return total_rows


def main():
    parser = argparse.ArgumentParser(description="Download and validate Jigsaw Toxic Comment dataset.")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Directory to save raw datasets")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files already exist")
    parser.add_argument("--only_train", action="store_true", help="Download only train.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = {"train.csv.gz": "train.csv"} if args.only_train else FILES_TO_DOWNLOAD

    print("=" * 60)
    print("Jigsaw Toxic Comment Dataset Acquisition Pipeline")
    print(f"Target directory: {os.path.abspath(args.output_dir)}")
    print("=" * 60)

    for gz_name, csv_name in files.items():
        gz_path = os.path.join(args.output_dir, gz_name)
        csv_path = os.path.join(args.output_dir, csv_name)

        if os.path.exists(csv_path) and not args.force:
            print(f"Found existing {csv_name}. Skipping download (use --force to overwrite).")
        else:
            url = f"{HF_BASE_URL}/{gz_name}"
            download_file(url, gz_path)
            decompress_gzip(gz_path, csv_path)

        if csv_name == "train.csv":
            validate_dataset(csv_path, EXPECTED_TRAIN_COLUMNS)

    print("=" * 60)
    print("Dataset acquisition and validation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
