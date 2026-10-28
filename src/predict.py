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
from src.kannada_analyzer import (
    detect_language,
    load_kannada_artifacts,
    predict_kannada_comment,
    get_kannada_context_analyzer
)
from src.emoji_analyzer import get_emoji_analyzer, compare_predictions

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

    kannada_artifacts = None
    try:
        kannada_artifacts = load_kannada_artifacts(model_dir=model_dir, force_reload=force_reload)
    except Exception:
        kannada_artifacts = None

    _LOADED_ARTIFACTS = {
        "vectorizer": vectorizer,
        "classifiers": classifiers,
        "metadata": metadata,
        "thresholds": thresholds,
        "labels": metadata.get("labels", DEFAULT_LABELS),
        "kannada": kannada_artifacts
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
    threshold_overrides: Optional[Dict[str, float]] = None,
    include_emojis: bool = True,
    return_comparison: bool = False
) -> Dict[str, Any]:
    """
    Analyze comment and return structured multi-label classification results.
    Integrates full sentence context evaluation, linguistic negation, and
    comprehensive emoji semantics (hostile gestures, violent threats, sarcasm,
    repetition multipliers, and cultural Kannada idioms).
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

    # 1. Detect language (Kannada, English, or Mixed)
    lang_info = detect_language(raw_text)
    is_kannada_lang = lang_info["is_kannada"]

    # 2. Extract and analyze emojis
    emoji_analyzer = get_emoji_analyzer()
    emoji_analysis = emoji_analyzer.analyze(raw_text, language_info=lang_info)

    # 3. Base Linguistic & ML Inference
    if is_kannada_lang:
        kn_artifacts = artifacts.get("kannada") if artifacts else None
        kn_res = predict_kannada_comment(
            raw_text=raw_text,
            artifacts=kn_artifacts,
            threshold_overrides=threshold_overrides
        )

        if lang_info["language"] == "mixed":
            contextual_text = mark_negation_and_context(raw_text)
            if not contextual_text or not contextual_text.strip():
                contextual_text = clean_text(raw_text) or "unparsed"
            vec = vectorizer.transform([contextual_text])
            en_raw_probs = {l: float(classifiers[l].predict_proba(vec)[0, 1]) for l in labels}
            en_eval = get_context_analyzer().evaluate(
                raw_text=raw_text,
                ml_probs=en_raw_probs,
                ml_thresholds=thresholds
            )

            combined_probs = {}
            combined_flags = set(kn_res["flagged_labels"])
            for l in labels:
                p_kn = kn_res["adjusted_probs"].get(l, 0.0)
                p_en = en_eval["adjusted_probs"].get(l, 0.0)
                combined_probs[l] = max(p_kn, p_en)
                if l in en_eval["final_flags"]:
                    combined_flags.add(l)

            base_flagged_labels = list(combined_flags)
            base_adjusted_probs = combined_probs
            base_is_toxic = kn_res["is_toxic"] or en_eval["final_is_toxic"]
            base_context_eval = kn_res["context_eval"]
            if en_eval.get("context_reasons"):
                base_context_eval["context_reasons"].extend(en_eval["context_reasons"])
        else:
            base_adjusted_probs = kn_res["adjusted_probs"]
            base_flagged_labels = kn_res["flagged_labels"]
            base_is_toxic = kn_res["is_toxic"]
            base_context_eval = kn_res["context_eval"]
    else:
        # Standard English pipeline
        contextual_text = mark_negation_and_context(raw_text)
        if not contextual_text or not contextual_text.strip():
            contextual_text = clean_text(raw_text) or "unparsed"

        vec = vectorizer.transform([contextual_text])
        raw_probs = {}
        for label in labels:
            clf = classifiers[label]
            prob = float(clf.predict_proba(vec)[0, 1])
            raw_probs[label] = prob

        base_context_eval = get_context_analyzer().evaluate(
            raw_text=raw_text,
            ml_probs=raw_probs,
            ml_thresholds=thresholds
        )
        base_adjusted_probs = base_context_eval["adjusted_probs"]
        base_flagged_labels = base_context_eval["final_flags"]
        base_is_toxic = base_context_eval["final_is_toxic"]

    # 4. Integrate Emoji Semantics, Combinations & Sarcasm into Final Verdict
    adjusted_probs = dict(base_adjusted_probs)
    flagged_labels = list(base_flagged_labels)
    context_reasons = list(base_context_eval.get("context_reasons", []))

    if include_emojis and emoji_analysis["has_emojis"]:
        if emoji_analysis["is_emoji_only"]:
            if emoji_analysis["label_overrides"]:
                for l in labels:
                    if l in emoji_analysis["label_overrides"]:
                        adjusted_probs[l] = float(emoji_analysis["label_overrides"][l])
                thresh_map = thresholds if not is_kannada_lang else artifacts.get("kannada", {}).get("thresholds", thresholds)
                flagged_labels = [l for l in labels if adjusted_probs[l] >= thresh_map.get(l, 0.35)]
                for r in emoji_analysis["context_reasons"]:
                    if r not in context_reasons:
                        context_reasons.append(r)
            else:
                for l in labels:
                    adjusted_probs[l] = min(adjusted_probs.get(l, 0.0), 0.04)
                flagged_labels = []
                for r in emoji_analysis["context_reasons"]:
                    if r not in context_reasons:
                        context_reasons.append(r)
        else:
            # Text + Emoji interaction
            # Sarcasm inversion
            if emoji_analysis["sarcasm_detected"]:
                adjusted_probs["toxic"] = max(adjusted_probs.get("toxic", 0.0), emoji_analysis["label_deltas"]["toxic"])
                adjusted_probs["insult"] = max(adjusted_probs.get("insult", 0.0), emoji_analysis["label_deltas"]["insult"])
                if "toxic" not in flagged_labels:
                    flagged_labels.append("toxic")
                if "insult" not in flagged_labels:
                    flagged_labels.append("insult")
                for r in emoji_analysis["context_reasons"]:
                    if r not in context_reasons:
                        context_reasons.append(r)

            # Hostile / Violent / Obscene emojis
            emoji_deltas = emoji_analysis["label_deltas"]
            thresh_map = thresholds if not is_kannada_lang else artifacts.get("kannada", {}).get("thresholds", thresholds)

            for l in labels:
                d = emoji_deltas.get(l, 0.0)
                th = thresh_map.get(l, 0.35)
                if d >= th:
                    adjusted_probs[l] = max(adjusted_probs.get(l, 0.0), d)
                    if adjusted_probs[l] >= th and l not in flagged_labels:
                        flagged_labels.append(l)
                elif d > 0.15:
                    adjusted_probs[l] = min(1.0, round(adjusted_probs.get(l, 0.0) + d * 0.25, 4))
                    if adjusted_probs[l] >= th and l not in flagged_labels:
                        flagged_labels.append(l)

            # Mitigating positive emojis (when no hostile signals or attacks present)
            if (
                emoji_analysis["is_mitigating"]
                and not base_context_eval.get("is_directed_attack")
                and not base_context_eval.get("is_violent_threat")
                and not emoji_analysis["sarcasm_detected"]
            ):
                for l in ["toxic", "insult", "severe_toxic"]:
                    adjusted_probs[l] = min(adjusted_probs.get(l, 0.0), 0.20)
                    if l in flagged_labels:
                        flagged_labels.remove(l)
                context_reasons.append("Friendly/positive emojis confirm constructive, non-hostile intention")

            for r in emoji_analysis.get("context_reasons", []):
                if r not in context_reasons:
                    context_reasons.append(r)

    is_toxic = len(flagged_labels) > 0

    # 5. Format Label Predictions & 4 Severity Levels
    label_predictions = {}
    thresh_map = thresholds if not is_kannada_lang else artifacts.get("kannada", {}).get("thresholds", thresholds)

    for label in labels:
        prob = float(adjusted_probs[label])
        thresh = float(thresh_map.get(label, 0.35))
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

    if context_reasons:
        reasons_text = "; ".join(context_reasons)
        interpretation = f"[{lang_info['display_name']}] {base_interp} Context Analysis: {reasons_text}."
    else:
        interpretation = f"[{lang_info['display_name']}] {base_interp}"

    inference_ms = round((time.perf_counter() - t_start) * 1000, 2)
    cleaned_preview = clean_text(raw_text)[:120] + ("..." if len(clean_text(raw_text)) > 120 else "")

    result = {
        "status": "success",
        "raw_text": raw_text,
        "is_toxic": is_toxic,
        "classification": "Toxic" if is_toxic else "Non-Toxic",
        "severity": severity,
        "flagged_labels": flagged_labels,
        "flagged_count": len(flagged_labels),
        "predictions": label_predictions,
        "interpretation": interpretation,
        "context_analysis": {
            "final_is_toxic": is_toxic,
            "final_flags": flagged_labels,
            "adjusted_probs": adjusted_probs,
            "context_reasons": context_reasons,
            "context_summary": "; ".join(context_reasons) if context_reasons else "Linguistic & Emoji Context Evaluation"
        },
        "has_emojis": emoji_analysis["has_emojis"],
        "is_emoji_only": emoji_analysis["is_emoji_only"],
        "sarcasm_detected": emoji_analysis["sarcasm_detected"],
        "emojis_detected": emoji_analysis["emojis_detected"],
        "emoji_analysis": emoji_analysis,
        "emoji_ui_chips": emoji_analysis["ui_chips"],
        "char_count": len(raw_text),
        "word_count": len(raw_text.split()),
        "cleaned_preview": cleaned_preview,
        "inference_time_ms": inference_ms,
        "detected_language": lang_info["display_name"],
        "language_code": lang_info["language"]
    }

    if return_comparison:
        base_res = predict_comment(
            raw_text=raw_text,
            artifacts=artifacts,
            threshold_overrides=threshold_overrides,
            include_emojis=False,
            return_comparison=False
        )
        result["model_comparison"] = compare_predictions(base_res, result)

    return result


if __name__ == "__main__":
    test_comment = "This is a wonderful and informative article. Thank you for sharing!"
    if len(sys.argv) > 1:
        test_comment = " ".join(sys.argv[1:])

    print(f"Testing prediction for: '{test_comment}'")
    result = predict_comment(test_comment)
    print(json.dumps(result, indent=2))
