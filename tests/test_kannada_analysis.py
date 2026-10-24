"""
tests/test_kannada_analysis.py

Unit and integration tests for Kannada language toxic comment detection:
  - Language detection (Kannada script, Romanized Kanglish, English, Mixed)
  - Non-toxic classifications (praise, gratitude, feedback)
  - Contextual negation handling (negated insults and negative terms)
  - Self-deprecating remarks
  - Toxic classifications (targeted insults, profanity, violent threats)
  - Integration with predict_comment and Flask API endpoints
  - Batch comment processing with analyze_youtube_comments
"""

import pytest
import json
from src.kannada_analyzer import (
    detect_language,
    predict_kannada_comment,
    load_kannada_artifacts,
    get_kannada_context_analyzer
)
from src.predict import predict_comment, load_model_artifacts
from src.youtube_analyzer import analyze_youtube_comments
from web.app import app


# -----------------------------------------------------------------------------
# 1. Language Detection Tests
# -----------------------------------------------------------------------------

def test_detect_language_kannada_script():
    res = detect_language("ತುಂಬಾ ಧನ್ಯವಾದಗಳು, ಈ ವಿಡಿಯೋ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ.")
    assert res["language"] == "kannada"
    assert res["is_kannada"] is True
    assert res["script"] == "kannada"
    assert res["kannada_char_count"] > 0


def test_detect_language_kanglish():
    res = detect_language("tumba chennagide guru, dhanyavadagalu")
    assert res["language"] == "kannada"
    assert res["is_kannada"] is True
    assert res["script"] == "latin"


def test_detect_language_english():
    res = detect_language("This is a fantastic explanation of neural networks.")
    assert res["language"] == "english"
    assert res["is_kannada"] is False
    assert res["script"] == "latin"


def test_detect_language_mixed():
    res = detect_language("Hello sir, ಈ ವಿಡಿಯೋ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ, keep it up!")
    assert res["language"] == "mixed"
    assert res["is_kannada"] is True
    assert res["script"] == "mixed"


# -----------------------------------------------------------------------------
# 2. Kannada Clean / Non-Toxic Comment Tests
# -----------------------------------------------------------------------------

def test_kannada_praise_script_non_toxic():
    clean_samples = [
        "ತುಂಬಾ ಧನ್ಯವಾದಗಳು, ಈ ವಿಡಿಯೋ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ ಮತ್ತು ಉಪಯುಕ್ತವಾಗಿದೆ.",
        "ಅದ್ಭುತವಾದ ಮಾಹಿತಿ, ಧನ್ಯವಾದಗಳು ಸರ್ ನಿಮ್ಮ ಕೆಲಸಕ್ಕೆ.",
        "ಸೂಪರ್ ಮಾಹಿತಿ ಅಣ್ಣ, ಮುಂದಿನ ಭಾಗಕ್ಕಾಗಿ ಕಾಯುತ್ತಿದ್ದೇನೆ.",
        "ಜೈ ಕರ್ನಾಟಕ ಮಾತೆ! ತುಂಬಾ ಒಳ್ಳೆಯ ಪ್ರಯತ್ನ.",
        "ಬೆಂಕಿ ಗುರು! ಅದ್ಭುತವಾದ ಎಡಿಟಿಂಗ್ ಮತ್ತು ಸ್ಪಷ್ಟ ಮಾಹಿತಿ."
    ]
    for text in clean_samples:
        res = predict_comment(text)
        assert res["status"] == "success"
        assert res["is_toxic"] is False, f"Failed on '{text}' - falsely flagged as toxic"
        assert res["classification"] == "Non-Toxic"
        assert "Kannada" in res["detected_language"]


def test_kannada_praise_kanglish_non_toxic():
    kanglish_samples = [
        "tumba chennagide video, dhanyavadagalu sir!",
        "super video anna, tumba thanks for the explanation",
        "adbhuta content bro, keep it up"
    ]
    for text in kanglish_samples:
        res = predict_comment(text)
        assert res["status"] == "success"
        assert res["is_toxic"] is False, f"Failed on '{text}' - falsely flagged as toxic"
        assert res["classification"] == "Non-Toxic"


# -----------------------------------------------------------------------------
# 3. Kannada Contextual Negation & Self-Deprecation Tests
# -----------------------------------------------------------------------------

def test_kannada_negated_insults_non_toxic():
    negated_samples = [
        "ನೀನು ದಡ್ಡ ಅಲ್ಲ, ತುಂಬಾ ಚೆನ್ನಾಗಿ ಮಾಡಿದ್ದೀಯ.",
        "ಈ ವಿಡಿಯೋ ಕೆಟ್ಟದಲ್ಲ, ನನಗೆ ತುಂಬಾ ಇಷ್ಟವಾಯಿತು.",
        "ಅವರು ಮೂರ್ಖರಲ್ಲ, ಅವರಿಗೆ ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ಸ್ವಲ್ಪ ಸಮಯ ಬೇಕು.",
        "neenu dadda alla, tumba smart aagi solve madidya",
        "ee video kettadalla, super aagide"
    ]
    for text in negated_samples:
        res = predict_comment(text)
        assert res["status"] == "success"
        assert res["is_toxic"] is False, f"Failed on negated comment '{text}' - falsely flagged as toxic"
        assert res["classification"] == "Non-Toxic"
        assert "insult" not in res["flagged_labels"]


def test_kannada_self_deprecating_non_toxic():
    samples = [
        "ನಾನು ದಡ್ಡ, ನನಗೆ ಈ ವಿಷಯ ಮೊದಲೇ ತಿಳಿದಿರಲಿಲ್ಲ.",
        "ನನ್ನದೇ ತಪ್ಪು, ಕ್ಷಮಿಸಿ ನಾನು ತಪ್ಪಾಗಿ ಗ್ರಹಿಸಿದೆ.",
        "naanu dadda nanage coding artha aagthilla"
    ]
    for text in samples:
        res = predict_comment(text)
        assert res["status"] == "success"
        assert res["is_toxic"] is False, f"Failed on self-deprecating comment '{text}'"
        assert res["classification"] == "Non-Toxic"


# -----------------------------------------------------------------------------
# 4. Kannada Toxic Comments Tests
# -----------------------------------------------------------------------------

def test_kannada_targeted_insult():
    insult_samples = [
        "ನೀನು ಒಬ್ಬ ಮೂರ್ಖ ಮತ್ತು ನಿಷ್ಪ್ರಯೋಜಕ ಕತ್ತೆ.",
        "ಬಾಯಿ ಮುಚ್ಚು ದಡ್ಡ, ನಿನಗೆ ಏನೂ ಗೊತ್ತಿಲ್ಲ.",
        "ನೀನು ಬೇವರ್ಸಿ, ಕಿತ್ತೋದವನೆ ತೊಲಗು ಇಲ್ಲಿಂದ.",
        "neenu obba huccha, kelsa nodkollo baddimaga",
        "baayi mucchu dadda katte, get lost"
    ]
    for text in insult_samples:
        res = predict_comment(text)
        assert res["status"] == "success"
        assert res["is_toxic"] is True, f"Failed to identify toxicity in '{text}'"
        assert res["classification"] == "Toxic"
        assert "insult" in res["flagged_labels"] or "toxic" in res["flagged_labels"]


def test_kannada_obscene_slurs():
    obscene_samples = [
        "ಸೂಳೆಮಗನೆ ಬಾಯಿ ಮುಚ್ಚಿಕೊಂಡು ಇರು.",
        "ಬೋಳಿಮಗನೆ ನಿನ್ನ ತಿಕ ಮುಚ್ಕೊಂಡು ಹೋಗೋ.",
        "soolemaga thika muchu",
        "bolimaga thika muchkondu hogu illinda"
    ]
    for text in obscene_samples:
        res = predict_comment(text)
        assert res["status"] == "success"
        assert res["is_toxic"] is True, f"Failed to identify toxicity in '{text}'"
        assert res["classification"] == "Toxic"
        assert "obscene" in res["flagged_labels"] or "toxic" in res["flagged_labels"]


def test_kannada_violent_threat():
    threat_samples = [
        "ನಿನ್ನನ್ನು ಕೊಲ್ಲುತ್ತೇನೆ ಹುಷಾರ್, ಎಲ್ಲಾದರೂ ಕಣ್ಣಿಗೆ ಬೀಳಬೇಡ.",
        "ನಿನ್ನ ಮನೆಗೆ ಬಂದು ಹೊಡೆದು ಸಾಯಿಸ್ತೀನಿ ಮಗನೆ.",
        "ninna kolluthene hushar, kan munde barbeda",
        "ninna jeeva thegitheeni sarvanasha madtini"
    ]
    for text in threat_samples:
        res = predict_comment(text)
        assert res["status"] == "success"
        assert res["is_toxic"] is True, f"Failed to identify threat in '{text}'"
        assert res["classification"] == "Toxic"
        assert "threat" in res["flagged_labels"]
        assert res["severity"] == "High"


# -----------------------------------------------------------------------------
# 5. API & Batch YouTube Analyzer Tests
# -----------------------------------------------------------------------------

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_predict_kannada_clean(client):
    payload = {"text": "ತುಂಬಾ ಧನ್ಯವಾದಗಳು ಸರ್, ಈ ವಿಡಿಯೋ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ."}
    res = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["status"] == "success"
    assert data["is_toxic"] is False
    assert data["classification"] == "Non-Toxic"
    assert "Kannada" in data["detected_language"]


def test_api_predict_kannada_toxic(client):
    payload = {"text": "ನೀನು ಒಬ್ಬ ಮೂರ್ಖ ಮತ್ತು ನಿಷ್ಪ್ರಯೋಜಕ ಕತ್ತೆ."}
    res = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["status"] == "success"
    assert data["is_toxic"] is True
    assert data["classification"] == "Toxic"


def test_analyze_youtube_comments_multilingual_batch():
    artifacts = load_model_artifacts("models")
    comments = [
        {"comment_id": "c1", "author": "UserEN", "text": "This tutorial is really fantastic!"},
        {"comment_id": "c2", "author": "UserKN_clean", "text": "ತುಂಬಾ ಧನ್ಯವಾದಗಳು ಗುರುಗಳೇ, ಸೂಪರ್ ಮಾಹಿತಿ."},
        {"comment_id": "c3", "author": "UserEN_toxic", "text": "You are a complete idiot and moron."},
        {"comment_id": "c4", "author": "UserKN_toxic", "text": "ನೀನು ಒಬ್ಬ ಮೂರ್ಖ ಮತ್ತು ನಿಷ್ಪ್ರಯೋಜಕ ಕತ್ತೆ."}
    ]
    res = analyze_youtube_comments(comments, artifacts)
    assert res["total_analyzed"] == 4
    assert res["toxic_count"] == 2
    assert res["non_toxic_count"] == 2
    assert res["toxic_percentage"] == 50.0
    assert res["non_toxic_percentage"] == 50.0
    assert "language_counts" in res
    assert res["language_counts"]["English"] >= 1
    assert res["language_counts"]["Kannada"] >= 1
