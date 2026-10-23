"""
scripts/prepare_data.py

Data preparation and preprocessing pipeline for the Jigsaw Toxic Comment dataset.
Performs:
  1. Loading raw train and test datasets.
  2. Cleaning text (stripping URLs, IP addresses, HTML, non-standard symbols).
  3. Validating multi-label targets across all 6 toxicity classes.
  4. Generating train/validation split (85% / 15%) with fixed random seed.
  5. Assembling the official held-out test set (filtering out -1 un-evaluated rows).
  6. Saving processed CSVs and label distribution metadata to data/processed/.
"""

import os
import sys
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import clean_text, compute_label_distribution, DEFAULT_LABELS


def prepare_data(raw_dir: str = "data/raw", processed_dir: str = "data/processed", random_state: int = 42):
    os.makedirs(processed_dir, exist_ok=True)

    train_path = os.path.join(raw_dir, "train.csv")
    test_path = os.path.join(raw_dir, "test.csv")
    test_labels_path = os.path.join(raw_dir, "test_labels.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing raw train file: {train_path}. Run scripts/download_data.py first.")

    print(f"Loading raw train dataset from {train_path} ...")
    df_train = pd.read_csv(train_path)
    print(f"Loaded {len(df_train):,} raw training records.")

    # Fill NaNs and clean text
    df_train["comment_text"] = df_train["comment_text"].fillna("").astype(str)
    print("Cleaning text for training records ...")
    df_train["clean_text"] = df_train["comment_text"].apply(clean_text)

    # Drop any that ended up completely empty
    df_train = df_train[df_train["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)
    print(f"Retained {len(df_train):,} clean non-empty comments.")

    # Compute label distributions
    dist = compute_label_distribution(df_train, DEFAULT_LABELS)
    dist_file = os.path.join(processed_dir, "label_distribution.json")
    with open(dist_file, "w", encoding="utf-8") as f:
        json.dump(dist, f, indent=2)
    print(f"Saved label distribution metadata to {dist_file}")

    # Create Train / Validation split
    print(f"Splitting into train (85%) and validation (15%) with random_state={random_state} ...")
    # For multi-label, we can stratify by the binary 'any_toxic' flag to maintain positive ratio
    df_train["any_toxic"] = (df_train[DEFAULT_LABELS].sum(axis=1) > 0).astype(int)

    train_split, val_split = train_test_split(
        df_train,
        test_size=0.15,
        random_state=random_state,
        stratify=df_train["any_toxic"]
    )

    train_split_path = os.path.join(processed_dir, "train_clean.csv")
    val_split_path = os.path.join(processed_dir, "val_clean.csv")

    train_split.to_csv(train_split_path, index=False)
    val_split.to_csv(val_split_path, index=False)
    print(f"Saved {len(train_split):,} train records to {train_split_path}")
    print(f"Saved {len(val_split):,} validation records to {val_split_path}")

    # Process official held-out test set if present
    if os.path.exists(test_path) and os.path.exists(test_labels_path):
        print("Processing official held-out test set ...")
        df_test_raw = pd.read_csv(test_path)
        df_test_labels = pd.read_csv(test_labels_path)

        merged_test = pd.merge(df_test_raw, df_test_labels, on="id")
        # In Kaggle Jigsaw, rows with -1 are un-evaluated rows
        evaluated_mask = merged_test["toxic"] != -1
        merged_test = merged_test[evaluated_mask].reset_index(drop=True)

        merged_test["comment_text"] = merged_test["comment_text"].fillna("").astype(str)
        merged_test["clean_text"] = merged_test["comment_text"].apply(clean_text)
        merged_test = merged_test[merged_test["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)

        test_clean_path = os.path.join(processed_dir, "test_clean.csv")
        merged_test.to_csv(test_clean_path, index=False)
        print(f"Saved {len(merged_test):,} evaluated held-out test records to {test_clean_path}")

    print("=" * 60)
    print("Data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and prepare Jigsaw Toxic Comment datasets.")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    prepare_data(raw_dir=args.raw_dir, processed_dir=args.processed_dir, random_state=args.random_state)
