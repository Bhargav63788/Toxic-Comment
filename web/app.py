"""
web/app.py

Flask backend for the YouTube Toxic Comment Analyzer.
Provides:
  - Single-Page Application interface
  - YouTube video comment fetching and multi-label toxicity analysis (/api/analyze_youtube)
  - Single-comment inference endpoint (/api/predict)
  - Model metadata and held-out test evaluation metrics (/api/model_info)
  - System health check (/api/health)
  - Secure API key handling via environment variables or ephemeral request parameter
"""

import os
import sys
import json
import time
from typing import Dict, Any
from flask import Flask, request, jsonify, render_template

# Automatically load environment variables from project .env
def load_env_file():
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val

load_env_file()

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import load_model_artifacts, predict_comment
from src.preprocessing import validate_input_comment, DEFAULT_LABELS
from src.youtube_analyzer import (
    extract_video_id,
    fetch_youtube_comments_and_meta,
    analyze_youtube_comments
)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)
app.config["JSON_SORT_KEYS"] = False

# Load model artifacts once during startup
print("Initializing YouTube Toxic Comment Analyzer backend ...")
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
try:
    ARTIFACTS = load_model_artifacts(model_dir=MODEL_DIR)
    print(f"Loaded model artifacts successfully from {MODEL_DIR}")
    print(f"Supported labels: {ARTIFACTS['labels']}")
except Exception as e:
    print(f"Warning: Failed to load model artifacts: {e}")
    ARTIFACTS = None


@app.route("/")
def index():
    """Serve the single-page YouTube Toxic Comment Analyzer interface."""
    has_env_key = bool(os.environ.get("YOUTUBE_API_KEY", "").strip())
    return render_template("index.html", has_env_key=has_env_key)


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    model_ready = ARTIFACTS is not None
    kannada_ready = bool(ARTIFACTS.get("kannada")) if model_ready else False
    has_api_key = bool(os.environ.get("YOUTUBE_API_KEY", "").strip())
    return jsonify({
        "status": "healthy" if model_ready else "degraded",
        "model_loaded": model_ready,
        "kannada_model_loaded": kannada_ready,
        "supported_languages": ["English", "Kannada (ಕನ್ನಡ)", "Kanglish (Romanized Kannada)"],
        "youtube_api_key_configured": has_api_key,
        "labels": ARTIFACTS["labels"] if model_ready else [],
        "server_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }), 200 if model_ready else 503


@app.route("/api/analyze_youtube", methods=["POST"])
def analyze_youtube():
    """
    Main workflow endpoint:
      1. Receives YouTube URL or Video ID
      2. Validates URL and extracts 11-char video ID
      3. Fetches accessible comments via official YouTube Data API v3
      4. Runs multi-label toxicity inference on every comment
      5. Computes exact counts and percentages
      6. Returns structured results
    """
    if ARTIFACTS is None:
        return jsonify({
            "status": "error",
            "message": "Model artifacts are not loaded on server. Please train the model first."
        }), 503

    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Request payload must be application/json."
        }), 400

    data = request.get_json(silent=True) or {}
    url_or_id = data.get("url", "")
    api_key_override = data.get("api_key", "").strip() or None
    raw_max = data.get("max_comments")
    if raw_max is None or str(raw_max).strip().lower() in ("", "none", "null", "all", "max"):
        max_comments = 5000
    else:
        try:
            max_comments = int(raw_max)
            if max_comments <= 0:
                max_comments = 5000
        except (ValueError, TypeError):
            max_comments = 5000
    include_replies = bool(data.get("include_replies", True))

    if not url_or_id or not str(url_or_id).strip():
        return jsonify({
            "status": "error",
            "message": "Please enter a valid YouTube video URL or ID."
        }), 400

    # 1. Validate URL and extract video ID
    video_id = extract_video_id(url_or_id)
    if not video_id:
        return jsonify({
            "status": "error",
            "message": (
                "Invalid YouTube URL. Please enter a valid video link, such as: "
                "https://www.youtube.com/watch?v=... or https://youtu.be/..."
            )
        }), 400

    # 2. Fetch comments and video metadata
    t_start = time.perf_counter()
    comments, video_meta, err_msg = fetch_youtube_comments_and_meta(
        video_id=video_id,
        api_key=api_key_override,
        max_comments=max_comments,
        include_replies=include_replies,
        use_demo_if_missing_key=True
    )

    if err_msg:
        # If comments are disabled, return clean success with 0 comments and creator notification
        if "comments are disabled" in err_msg.lower():
            return jsonify({
                "status": "success",
                "video_id": video_id,
                "video_meta": video_meta,
                "total_analyzed": 0,
                "toxic_count": 0,
                "non_toxic_count": 0,
                "toxic_percentage": 0.0,
                "non_toxic_percentage": 0.0,
                "category_counts": {l: 0 for l in DEFAULT_LABELS},
                "comments": [],
                "latency_ms": round((time.perf_counter() - t_start) * 1000, 1),
                "disclaimer": "Comments are disabled on this YouTube video by the creator."
            }), 200

        # Check other error types
        status_code = 400
        if "quota" in err_msg.lower():
            status_code = 429
        elif "not configured" in err_msg.lower() or "invalid" in err_msg.lower():
            status_code = 401 if "invalid" in err_msg.lower() else 400

        return jsonify({
            "status": "error",
            "message": err_msg,
            "video_id": video_id,
            "video_meta": video_meta
        }), status_code

    # 3. Analyze comments through trained ML model
    analysis_result = analyze_youtube_comments(comments, ARTIFACTS)
    latency_ms = round((time.perf_counter() - t_start) * 1000, 1)

    total_analyzed = analysis_result["total_analyzed"]
    raw_total_comments = video_meta.get("total_comment_count")
    total_on_video = None
    if raw_total_comments is not None:
        try:
            total_on_video = int(raw_total_comments)
        except (ValueError, TypeError):
            total_on_video = None

    if total_analyzed == 0:
        disclaimer = "This video currently has 0 public comments to analyze."
    elif total_on_video is not None and total_on_video <= total_analyzed:
        disclaimer = f"Analyzed all {total_analyzed:,} available comments on this video (100% complete dataset)."
    elif total_on_video is not None:
        disclaimer = f"Analyzed {total_analyzed:,} comments out of ~{total_on_video:,} total comments on YouTube."
    else:
        disclaimer = f"Analyzed {total_analyzed:,} comments successfully retrieved via YouTube Data API."

    return jsonify({
        "status": "success",
        "video_id": video_id,
        "video_meta": video_meta,
        "total_analyzed": total_analyzed,
        "toxic_count": analysis_result["toxic_count"],
        "non_toxic_count": analysis_result["non_toxic_count"],
        "toxic_percentage": analysis_result["toxic_percentage"],
        "non_toxic_percentage": analysis_result["non_toxic_percentage"],
        "category_counts": analysis_result["category_counts"],
        "language_counts": analysis_result.get("language_counts", {}),
        "emoji_stats": analysis_result.get("emoji_stats", {}),
        "comments": analysis_result["analyzed_comments"],
        "latency_ms": latency_ms,
        "disclaimer": disclaimer
    }), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    """Single-comment prediction endpoint with full emoji context and comparison."""
    if ARTIFACTS is None:
        return jsonify({"status": "error", "message": "Model not loaded."}), 503

    if not request.is_json:
        return jsonify({"status": "error", "message": "JSON payload required."}), 400

    data = request.get_json(silent=True) or {}
    raw_text = data.get("text")
    compare_mode = bool(data.get("compare", True))
    is_valid, err_msg = validate_input_comment(raw_text, max_chars=5000)
    if not is_valid:
        status_code = 413 if "exceeds maximum" in err_msg.lower() else 400
        return jsonify({"status": "error", "message": err_msg}), status_code

    result = predict_comment(raw_text, artifacts=ARTIFACTS, return_comparison=compare_mode)
    return jsonify(result), 200


@app.route("/api/model_info", methods=["GET"])
def model_info():
    """Return model metadata and test evaluation metrics."""
    eval_json_path = os.path.join(MODEL_DIR, "evaluation_metrics.json")
    meta_json_path = os.path.join(MODEL_DIR, "model_metadata.json")

    eval_data = {}
    if os.path.exists(eval_json_path):
        try:
            with open(eval_json_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        except Exception:
            pass

    meta_data = {}
    if os.path.exists(meta_json_path):
        try:
            with open(meta_json_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
        except Exception:
            pass

    kn_meta_path = os.path.join(MODEL_DIR, "kannada_model_metadata.json")
    kn_meta_data = {}
    if os.path.exists(kn_meta_path):
        try:
            with open(kn_meta_path, "r", encoding="utf-8") as f:
                kn_meta_data = json.load(f)
        except Exception:
            pass

    return jsonify({
        "status": "success",
        "supported_languages": ["English", "Kannada (ಕನ್ನಡ)", "Kanglish (Romanized Kannada)"],
        "dataset": {
            "name": "Jigsaw Toxic Comment Classification Challenge",
            "primary_source": "https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data",
            "mirror_source": "https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments",
            "documentation": "https://huggingface.co/datasets/google/jigsaw_toxicity_pred",
            "training_samples": meta_data.get("train_samples", 135620),
            "held_out_test_samples": eval_data.get("test_sample_count", 63613),
            "labels": DEFAULT_LABELS
        },
        "model_architecture": {
            "vectorizer": "TF-IDF (1-2 N-grams, Sublinear TF, 40,000 Max Features)",
            "classifiers": "One-vs-Rest Calibrated Logistic Regression (L-BFGS, C=2.0)",
            "decision_thresholds": meta_data.get("optimal_thresholds", {})
        },
        "evaluation_metrics": eval_data,
        "kannada_model": kn_meta_data
    }), 200


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting YouTube Toxic Comment Analyzer at http://{host}:{port} ...")
    app.run(host=host, port=port, debug=False)