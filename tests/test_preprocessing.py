"""
tests/test_preprocessing.py

Unit tests for preprocessing, text cleaning, and validation functions.
"""

import pytest
import pandas as pd
from src.preprocessing import clean_text, validate_input_comment, compute_label_distribution, DEFAULT_LABELS


def test_clean_text_normal():
    raw = "Hello World! This is a test comment."
    cleaned = clean_text(raw)
    assert "hello world" in cleaned
    assert "test comment" in cleaned
    assert "!" not in cleaned


def test_clean_text_html_entities():
    raw = "You &amp; me &quot;rock&quot;!"
    cleaned = clean_text(raw)
    assert "you and me" in cleaned or "you" in cleaned
    assert "&quot;" not in cleaned
    assert "&amp;" not in cleaned


def test_clean_text_urls_and_ips():
    raw = "Check this link https://example.com/bad and IP 192.168.1.1 immediately."
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "example.com" not in cleaned
    assert "192.168.1.1" not in cleaned
    assert "check this link" in cleaned


def test_clean_text_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   \n\t  ") == ""


def test_validate_input_comment_valid():
    is_valid, err = validate_input_comment("This is a clean and thoughtful remark.")
    assert is_valid is True
    assert err == ""


def test_validate_input_comment_empty():
    is_valid, err = validate_input_comment("")
    assert is_valid is False
    assert "empty" in err.lower()

    is_valid, err = validate_input_comment("    \n  ")
    assert is_valid is False
    assert "whitespace" in err.lower()


def test_validate_input_comment_null():
    is_valid, err = validate_input_comment(None)
    assert is_valid is False
    assert "null" in err.lower() or "empty" in err.lower()


def test_validate_input_comment_oversized():
    huge_text = "word " * 1500  # > 5000 chars
    is_valid, err = validate_input_comment(huge_text, max_chars=5000)
    assert is_valid is False
    assert "exceeds maximum" in err.lower()


def test_compute_label_distribution():
    data = {
        "comment_text": ["a", "b", "c", "d"],
        "toxic": [1, 0, 1, 0],
        "severe_toxic": [0, 0, 1, 0],
        "obscene": [1, 0, 0, 0],
        "threat": [0, 0, 0, 0],
        "insult": [1, 0, 1, 0],
        "identity_hate": [0, 0, 0, 0]
    }
    df = pd.DataFrame(data)
    stats = compute_label_distribution(df, DEFAULT_LABELS)
    assert stats["toxic"]["count"] == 2
    assert stats["toxic"]["percentage"] == 50.0
    assert stats["severe_toxic"]["count"] == 1
    assert stats["any_toxic"]["count"] == 2
