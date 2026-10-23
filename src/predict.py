"""
src/predict.py

Inference engine for multi-label toxic comment classification.
Loads model artifacts (TF-IDF vectorizer, 6 binary classifiers, threshold metadata)
and predicts toxicity scores and binary flags for incoming text comments.
Integrates full sentence context evaluation to prevent classifications based only
on isolated trigger words.
"""

import os
import sys
import json
import time
import joblib
from typing import Dict, Any, List, Optional

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import clean_text, validate_input_comment, DEFAULT_LABELS
from src.context_analyzer import get_context_analyzer, mark_negation_and_context

_LOADED_ARTIFACTS = None


def load_model_artifacts(model_dir: str = "models", force_reload: bool = False) -> Dict[str, Any]:
    """
    Load vectorizer, classifiers, and metadata into memory.
    Cached for fast repeated inferences.
    """
    global _LOADED_ARTIFACTS
    if _LOADED_ARTIFACTS is not None and not force_reload:
        return _LOADED_ARTIFACTS

    vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    model_path = os.path.join(model_dir, "multilabel_toxicity_model.pkl")
    meta_path = os.path.join(model_dir, "model_metadata.json")

    if not os.path.exists(vec_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifacts missing in {model_dir}. Please run 'python src/train.py' first."
        )

    vectorizer = joblib.load(vec_path)
    classifiers = joblib.load(model_path)

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    thresholds = metadata.get("optimal_thresholds", {l: 0.5 for l in DEFAULT_LABELS})

    _LOADED_ARTIFACTS = {
        "vectorizer": vectorizer,
        "classifiers": classifiers,
        "metadata": metadata,
        "thresholds": thresholds,
        "labels": metadata.get("labels", DEFAULT_LABELS)
    }
    return _LOADED_ARTIFACTS


def compute_severity(flagged: List[str], probs: Dict[str, float]) -> str:
    """
    Compute overall severity rating:
      - 'High': threat, severe_toxic, or 3+ categories flagged
      - 'Medium': 2 categories flagged or obscene/insult with prob > 0.70
      - 'Low': single mild flag (e.g. toxic or insult with prob < 0.70)
      - 'Clean': 0 categories flagged
    """
    if not flagged:
        return "Clean"

    if "severe_toxic" in flagged or "threat" in flagged or "identity_hate" in flagged:
        return "High"

    if len(flagged) >= 3:
        return "High"

    if len(flagged) >= 2 or probs.get("obscene", 0) > 0.75 or probs.get("insult", 0) > 0.75:
        return "Medium"

    return "Low"


def generate_interpretation(is_toxic: bool, severity: str, flagged: List[str], probs: Dict[str, float]) -> str:
    """Generate concise, contextual explanation of the model prediction."""
    if not is_toxic:
        max_prob_label = max(probs, key=probs.get) if probs else "none"
        max_prob = probs.get(max_prob_label, 0)
        if max_prob < 0.2:
            return "This comment appears respectful and constructive with very low toxicity signals across all categories."
        else:
            return f"No category exceeded decision thresholds. Highest detected signal was '{max_prob_label}' at {max_prob*100:.1f}%."

    label_names_str = ", ".join([f"'{l.replace('_', ' ')}'" for l in flagged])
    if severity == "High":
        return f"CRITICAL: Highly toxic content detected flagged under {label_names_str}. Strong recommendation for automated moderation or immediate human review."
    elif severity == "Medium":
        return f"MODERATE: Noticeable toxicity detected flagged under {label_names_str}. Review recommended according to community guidelines."
    else:
        return f"LOW SEVERITY: Borderline or mildly toxic content detected flagged under {label_names_str}."


def predict_comment(
    raw_text: str,
    artifacts: Optional[Dict[str, Any]] = None,
    threshold_overrides: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Analyze comment and return structured multi-label classification results.
    Considers the complete sentence, context, negation, and intent rather
    than classifying based merely on isolated words.
    """
    t_start = time.perf_counter()

    # Validate input
    is_valid, err_msg = validate_input_comment(raw_text)
    if not is_valid:
        return {
            "status": "error",
            "error": err_msg,
            "raw_text": raw_text
        }

    # Load artifacts if not provided
    if artifacts is None:
        artifacts = load_model_artifacts()

    vectorizer = artifacts["vectorizer"]
    classifiers = artifacts["classifiers"]
    thresholds = dict(artifacts["thresholds"])
    if threshold_overrides:
        thresholds.update(threshold_overrides)

    labels = artifacts["labels"]

    # Preprocessing with contextual text scoping
    contextual_text = mark_negation_and_context(raw_text)
    if not contextual_text or not contextual_text.strip():
        contextual_text = clean_text(raw_text) or "unparsed"

    # ML Vectorization & raw inference
    vec = vectorizer.transform([contextual_text])

    raw_probs = {}
    for label in labels:
        clf = classifiers[label]
        prob = float(clf.predict_proba(vec)[0, 1])
        raw_probs[label] = prob

    # Contextual sentence analysis to evaluate full meaning and prevent lone-word false positives
    context_eval = get_context_analyzer().evaluate(
        raw_text=raw_text,
        ml_probs=raw_probs,
        ml_thresholds=thresholds
    )

    adjusted_probs = context_eval["adjusted_probs"]
    flagged_labels = context_eval["final_flags"]
    is_toxic = context_eval["final_is_toxic"]

    label_predictions = {}
    for label in labels:
        prob = float(adjusted_probs[label])
        thresh = float(thresholds.get(label, 0.5))
        is_flagged = bool(label in flagged_labels)

        if prob >= 0.75:
            risk = "High"
        elif prob >= thresh:
            risk = "Moderate"
        elif prob >= 0.25:
            risk = "Elevated"
        else:
            risk = "Low"

        label_predictions[label] = {
            "probability": round(prob, 4),
            "percentage": round(prob * 100, 1),
            "threshold": round(thresh, 2),
            "flagged": is_flagged,
            "risk_level": risk
        }

    severity = compute_severity(flagged_labels, adjusted_probs)
    base_interp = generate_interpretation(is_toxic, severity, flagged_labels, adjusted_probs)
    if context_eval.get("context_reasons"):
        reasons_text = "; ".join(context_eval["context_reasons"])
        interpretation = f"{base_interp} Context Analysis: {reasons_text}."
    else:
        interpretation = base_interp

    inference_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "status": "success",
        "is_toxic": is_toxic,
        "classification": "Toxic" if is_toxic else "Non-Toxic",
        "severity": severity,
        "flagged_labels": flagged_labels,
        "flagged_count": len(flagged_labels),
        "predictions": label_predictions,
        "interpretation": interpretation,
        "context_analysis": context_eval,
        "char_count": len(raw_text),
        "word_count": len(raw_text.split()),
        "cleaned_preview": contextual_text[:120] + ("..." if len(contextual_text) > 120 else ""),
        "inference_time_ms": inference_ms
    }


if __name__ == "__main__":
    test_comment = "This is a wonderful and informative article. Thank you for sharing!"
    if len(sys.argv) > 1:
        test_comment = " ".join(sys.argv[1:])

    print(f"Testing prediction for: '{test_comment}'")
    result = predict_comment(test_comment)
    print(json.dumps(result, indent=2))
