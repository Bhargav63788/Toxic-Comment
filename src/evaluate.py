"""
src/evaluate.py

Evaluation script for the Multi-Label Toxic Comment Classification model
using the held-out test set (63,613 official evaluated Kaggle test comments).

Computes:
  - Per-label Precision, Recall, F1-score, ROC-AUC
  - Confusion matrix counts (TP, FP, FN, TN, Support)
  - Multi-label aggregate metrics: Macro F1, Micro F1, Hamming Loss
  - Comparison between optimal validation thresholds vs. baseline 0.50
Saves metrics to models/evaluation_report.csv and models/evaluation_metrics.json.
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
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    hamming_loss,
    accuracy_score,
    confusion_matrix
)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import DEFAULT_LABELS
from src.context_analyzer import mark_negation_and_context


def evaluate_model(
    test_path: str = "data/processed/test_clean.csv",
    model_dir: str = "models",
    output_report_csv: str = "models/evaluation_report.csv",
    output_metrics_json: str = "models/evaluation_metrics.json",
    use_optimal_thresholds: bool = True
):
    print("=" * 70)
    print("Multi-Label Toxic Comment Detection - Test Evaluation")
    print(f"Test dataset: {test_path}")
    print(f"Model dir:    {model_dir}")
    print("=" * 70)

    # 1. Load test set
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test file: {test_path}. Run scripts/prepare_data.py first.")

    print("\n[1/4] Loading held-out test dataset ...")
    df_test = pd.read_csv(test_path)
    print(f"Loaded {len(df_test):,} held-out test records.")

    # 2. Load model artifacts
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    model_path = os.path.join(model_dir, "multilabel_toxicity_model.pkl")
    meta_path = os.path.join(model_dir, "model_metadata.json")

    vectorizer = joblib.load(vec_path)
    classifiers = joblib.load(model_path)

    thresholds = {l: 0.5 for l in DEFAULT_LABELS}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            if use_optimal_thresholds:
                thresholds = meta.get("optimal_thresholds", thresholds)

    print(f"Applied decision thresholds: {thresholds}")

    # 3. Vectorize test text with contextual normalization
    print("\n[2/4] Vectorizing held-out test comments ...")
    t0 = time.time()
    test_raw = df_test["comment_text"].fillna(df_test["clean_text"]).astype(str).values
    test_contextual = [mark_negation_and_context(t) for t in test_raw]
    X_test_vec = vectorizer.transform(test_contextual)
    print(f"Vectorized in {time.time() - t0:.2f} seconds.")

    # 4. Predict & Evaluate
    print("\n[3/4] Computing per-label test metrics ...")
    results = []
    y_true_all = []
    y_pred_all = []

    print("-" * 88)
    print(f"{'Label':<15} | {'Thresh':<6} | {'Precision':<9} | {'Recall':<7} | {'F1-Score':<8} | {'ROC-AUC':<7} | {'Support':<7} | {'TP':<5} | {'FP':<5}")
    print("-" * 88)

    per_label_json = {}

    for label in DEFAULT_LABELS:
        y_true = df_test[label].values
        clf = classifiers[label]
        probs = clf.predict_proba(X_test_vec)[:, 1]
        t = thresholds.get(label, 0.5)
        y_pred = (probs >= t).astype(int)

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        support = int(np.sum(y_true))

        row = {
            "label": label,
            "threshold": round(t, 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4),
            "support": support,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn)
        }
        results.append(row)
        per_label_json[label] = row

        print(f"{label:<15} | {t:<6.2f} | {prec:<9.4f} | {rec:<7.4f} | {f1:<8.4f} | {auc:<7.4f} | {support:<7} | {tp:<5} | {fp:<5}")

    print("-" * 88)

    Y_true_mat = np.column_stack(y_true_all)
    Y_pred_mat = np.column_stack(y_pred_all)

    macro_f1 = float(np.mean([r["f1"] for r in results]))
    macro_prec = float(np.mean([r["precision"] for r in results]))
    macro_rec = float(np.mean([r["recall"] for r in results]))
    macro_auc = float(np.mean([r["roc_auc"] for r in results]))

    micro_f1 = float(f1_score(Y_true_mat, Y_pred_mat, average="micro", zero_division=0))
    micro_prec = float(precision_score(Y_true_mat, Y_pred_mat, average="micro", zero_division=0))
    micro_rec = float(recall_score(Y_true_mat, Y_pred_mat, average="micro", zero_division=0))

    h_loss = float(hamming_loss(Y_true_mat, Y_pred_mat))
    exact_match = float(accuracy_score(Y_true_mat, Y_pred_mat))

    print(f"\n[SUMMARY MULTI-LABEL PERFORMANCE ON HELD-OUT TEST SET]")
    print(f"  Macro F1-Score:    {macro_f1:.4f}")
    print(f"  Macro Precision:   {macro_prec:.4f}")
    print(f"  Macro Recall:      {macro_rec:.4f}")
    print(f"  Macro ROC-AUC:     {macro_auc:.4f}")
    print(f"  Micro F1-Score:    {micro_f1:.4f}")
    print(f"  Micro Precision:   {micro_prec:.4f}")
    print(f"  Micro Recall:      {micro_rec:.4f}")
    print(f"  Hamming Loss:      {h_loss:.4f} (lower is better)")
    print(f"  Exact Match Ratio: {exact_match*100:.2f}%")

    print("\n[4/4] Saving test evaluation metrics and report ...")
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_report_csv, index=False)

    summary_metrics = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "test_dataset_path": test_path,
        "test_sample_count": len(df_test),
        "macro_metrics": {
            "f1": round(macro_f1, 4),
            "precision": round(macro_prec, 4),
            "recall": round(macro_rec, 4),
            "roc_auc": round(macro_auc, 4)
        },
        "micro_metrics": {
            "f1": round(micro_f1, 4),
            "precision": round(micro_prec, 4),
            "recall": round(micro_rec, 4)
        },
        "hamming_loss": round(h_loss, 5),
        "exact_match_ratio": round(exact_match, 4),
        "per_label": per_label_json
    }

    with open(output_metrics_json, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"Saved evaluation report CSV to {output_report_csv}")
    print(f"Saved evaluation metrics JSON to {output_metrics_json}")
    print("=" * 70)
    print("Evaluation completed successfully!")
    print("=" * 70)
    return summary_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multi-label model on held-out test data.")
    parser.add_argument("--test_data", type=str, default="data/processed/test_clean.csv")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--out_csv", type=str, default="models/evaluation_report.csv")
    parser.add_argument("--out_json", type=str, default="models/evaluation_metrics.json")
    parser.add_argument("--default_thresholds", action="store_true", help="Use 0.5 default threshold instead of tuned")
    args = parser.parse_args()

    evaluate_model(
        test_path=args.test_data,
        model_dir=args.model_dir,
        output_report_csv=args.out_csv,
        output_metrics_json=args.out_json,
        use_optimal_thresholds=not args.default_thresholds
    )
