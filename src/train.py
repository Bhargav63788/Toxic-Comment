"""
src/train.py

Context-Aware Training Pipeline for Multi-Label Toxic Comment Classification.
  - Preprocesses complete comments with negation scoping and contextual idioms.
  - Fits TF-IDF vectorizer strictly on training data (no data leakage).
  - Uses identity-subgroup sample reweighting to prevent lone-word identity bias.
  - Trains independent binary classifiers (One-vs-Rest) for all 6 Jigsaw labels:
      toxic, severe_toxic, obscene, threat, insult, identity_hate.
  - Evaluates on validation set and calibrates robust per-label decision thresholds.
  - Saves vectorizer, classifiers, and metadata into models/ directory.
"""

import os
import sys
import json
import time
import joblib
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import DEFAULT_LABELS
from src.context_analyzer import mark_negation_and_context, IDENTITY_WORDS


def compute_identity_sample_weights(texts: list, any_toxic_series: pd.Series) -> np.ndarray:
    """
    Upweight non-toxic comments containing identity/sensitive keywords so the
    model is heavily penalized for wrongly attributing toxicity to identity terms alone.
    """
    weights = np.ones(len(texts), dtype=float)
    is_nontoxic = (any_toxic_series == 0).values
    for idx, t in enumerate(texts):
        if is_nontoxic[idx]:
            t_low = t.lower()
            if any(id_word in t_low for id_word in IDENTITY_WORDS):
                weights[idx] = 3.5
    return weights


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_thresh: float = 0.25,
    max_thresh: float = 0.85,
    step: float = 0.02
) -> float:
    """Find probability threshold that maximizes F1 score on validation set with safety floor."""
    best_thresh = 0.5
    best_f1 = -1.0
    thresholds = np.arange(min_thresh, max_thresh, step)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(round(t, 2))
    return best_thresh


def train(
    train_path: str = "data/processed/train_clean.csv",
    val_path: str = "data/processed/val_clean.csv",
    out_dir: str = "models",
    max_features: int = 45000,
    c_reg: float = 2.0,
    random_state: int = 42,
    subsample: int = None
):
    os.makedirs(out_dir, exist_ok=True)
    start_time = time.time()

    print("=" * 65)
    print("Multi-Label Toxic Comment Detection - Contextual Training Pipeline")
    print(f"Train dataset: {train_path}")
    print(f"Val dataset:   {val_path}")
    print(f"Output dir:    {out_dir}")
    print("=" * 65)

    print("\n[1/5] Loading datasets ...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    if subsample and subsample < len(df_train):
        print(f"Subsampling training set to {subsample:,} records for fast mode ...")
        df_train = df_train.sample(n=subsample, random_state=random_state).reset_index(drop=True)

    print(f"Train samples: {len(df_train):,}")
    print(f"Val samples:   {len(df_val):,}")

    # Extract original text and apply contextual negation & idiom marking
    train_raw = df_train["comment_text"].fillna(df_train["clean_text"]).astype(str).values
    val_raw = df_val["comment_text"].fillna(df_val["clean_text"]).astype(str).values

    print("\n[2/5] Contextual text normalization & negation scoping ...")
    t0 = time.time()
    X_train_text = [mark_negation_and_context(t) for t in train_raw]
    X_val_text = [mark_negation_and_context(t) for t in val_raw]
    print(f"Contextual processing completed in {time.time() - t0:.2f}s")

    # Compute debiasing sample weights
    train_any_toxic = df_train[DEFAULT_LABELS].sum(axis=1)
    sample_weights = compute_identity_sample_weights(X_train_text, train_any_toxic)
    boosted_count = int(np.sum(sample_weights > 1.0))
    print(f"Debiasing: Boosted sample weight for {boosted_count:,} non-toxic identity comments.")

    print("\n[3/5] Fitting TF-IDF Vectorizer on contextual training data ...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        strip_accents="unicode"
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_val_vec = vectorizer.transform(X_val_text)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,} features")

    classifiers = {}
    thresholds = {}
    val_metrics = {}

    # Label-specific threshold floors to avoid noise triggers
    label_floors = {
        "toxic": 0.35,
        "severe_toxic": 0.20,
        "obscene": 0.28,
        "threat": 0.20,
        "insult": 0.30,
        "identity_hate": 0.25
    }

    print("\n[4/5] Training per-label classifiers & tuning thresholds on validation set ...")
    for label in DEFAULT_LABELS:
        t0 = time.time()
        y_train = df_train[label].values
        y_val = df_val[label].values

        clf = LogisticRegression(
            C=c_reg,
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
            n_jobs=1
        )
        clf.fit(X_train_vec, y_train, sample_weight=sample_weights)
        classifiers[label] = clf

        val_probs = clf.predict_proba(X_val_vec)[:, 1]
        floor = label_floors.get(label, 0.25)
        best_thresh = find_best_threshold(y_val, val_probs, min_thresh=floor)
        thresholds[label] = best_thresh

        val_preds = (val_probs >= best_thresh).astype(int)
        precision = float(precision_score(y_val, val_preds, zero_division=0))
        recall = float(recall_score(y_val, val_preds, zero_division=0))
        f1 = float(f1_score(y_val, val_preds, zero_division=0))
        auc = float(roc_auc_score(y_val, val_probs)) if len(np.unique(y_val)) > 1 else 0.5

        val_metrics[label] = {
            "threshold": best_thresh,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4),
            "positive_count_val": int(np.sum(y_val)),
            "train_duration_sec": round(time.time() - t0, 2)
        }

        print(f"  [{label:<14}] F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | ROC-AUC: {auc:.4f} | Opt-Thresh: {best_thresh:.2f} ({val_metrics[label]['train_duration_sec']}s)")

    # Aggregate validation scores
    macro_f1 = float(np.mean([m["f1"] for m in val_metrics.values()]))
    macro_auc = float(np.mean([m["roc_auc"] for m in val_metrics.values()]))
    print(f"\nValidation Macro F1: {macro_f1:.4f} | Validation Macro ROC-AUC: {macro_auc:.4f}")

    print("\n[5/5] Saving model artifacts ...")
    vec_path = os.path.join(out_dir, "tfidf_vectorizer.pkl")
    model_path = os.path.join(out_dir, "multilabel_toxicity_model.pkl")
    meta_path = os.path.join(out_dir, "model_metadata.json")

    joblib.dump(vectorizer, vec_path, compress=3)
    joblib.dump(classifiers, model_path, compress=3)

    metadata = {
        "model_name": "Contextual-TfIdf-LogisticRegression-MultiLabel",
        "labels": DEFAULT_LABELS,
        "optimal_thresholds": thresholds,
        "default_threshold": 0.5,
        "vectorizer_config": {
            "max_features": max_features,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "min_df": 3,
            "vocab_size": len(vectorizer.vocabulary_)
        },
        "classifier_config": {
            "algorithm": "LogisticRegression",
            "C": c_reg,
            "solver": "lbfgs",
            "random_state": random_state
        },
        "validation_metrics": val_metrics,
        "validation_macro_f1": round(macro_f1, 4),
        "validation_macro_roc_auc": round(macro_auc, 4),
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "trained_at_utc": datetime.utcnow().isoformat(),
        "total_train_time_sec": round(time.time() - start_time, 2)
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved vectorizer to {vec_path} ({os.path.getsize(vec_path)/(1024*1024):.2f} MB)")
    print(f"Saved multi-label classifiers to {model_path} ({os.path.getsize(model_path)/(1024*1024):.2f} MB)")
    print(f"Saved model metadata to {meta_path}")
    print(f"\nContextual training finished successfully in {time.time() - start_time:.2f}s!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Context-Aware Multi-Label Toxic Comment Classifier.")
    parser.add_argument("--train_path", type=str, default="data/processed/train_clean.csv")
    parser.add_argument("--val_path", type=str, default="data/processed/val_clean.csv")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--max_features", type=int, default=45000)
    parser.add_argument("--c_reg", type=float, default=2.0)
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    train(
        train_path=args.train_path,
        val_path=args.val_path,
        out_dir=args.out_dir,
        max_features=args.max_features,
        c_reg=args.c_reg,
        random_state=args.random_state,
        subsample=args.subsample
    )
