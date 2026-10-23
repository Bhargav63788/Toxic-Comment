"""
tests/test_model.py

Unit tests for model artifact loading, prediction engine, and inference logic.
"""

import pytest
import os
from src.predict import load_model_artifacts, predict_comment, compute_severity
from src.preprocessing import DEFAULT_LABELS


def test_load_model_artifacts():
    artifacts = load_model_artifacts("models")
    assert "vectorizer" in artifacts
    assert "classifiers" in artifacts
    assert "thresholds" in artifacts
    assert len(artifacts["classifiers"]) == 6
    for label in DEFAULT_LABELS:
        assert label in artifacts["classifiers"]
        assert label in artifacts["thresholds"]


def test_predict_clean_comment():
    res = predict_comment("Thank you for your assistance, this is very helpful.")
    assert res["status"] == "success"
    assert res["is_toxic"] is False
    assert res["severity"] == "Clean"
    assert res["flagged_count"] == 0
    assert len(res["predictions"]) == 6
    for label in DEFAULT_LABELS:
        assert 0.0 <= res["predictions"][label]["probability"] <= 1.0
        assert res["predictions"][label]["flagged"] is False


def test_predict_toxic_comment():
    res = predict_comment("You are an idiot and a moron, get lost!")
    assert res["status"] == "success"
    assert res["is_toxic"] is True
    assert res["flagged_count"] >= 1
    assert "toxic" in res["flagged_labels"]
    assert res["severity"] in ["Medium", "High"]


def test_predict_threat_comment():
    res = predict_comment("I will kill you and destroy your life.")
    assert res["status"] == "success"
    assert res["is_toxic"] is True
    assert res["predictions"]["threat"]["probability"] > 0.05
    assert res["severity"] in ["High", "Medium"]


def test_compute_severity():
    assert compute_severity([], {}) == "Clean"
    assert compute_severity(["toxic"], {"toxic": 0.3}) == "Low"
    assert compute_severity(["toxic", "obscene"], {"toxic": 0.8, "obscene": 0.8}) == "Medium"
    assert compute_severity(["threat"], {"threat": 0.9}) == "High"
    assert compute_severity(["severe_toxic"], {"severe_toxic": 0.8}) == "High"
    assert compute_severity(["identity_hate"], {"identity_hate": 0.7}) == "High"
