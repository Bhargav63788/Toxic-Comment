"""
tests/test_context_toxicity.py

Unit and integration tests verifying contextual sentence analysis, negation handling,
identity de-biasing, and prevention of lone-word false positives.
"""

import pytest
from src.predict import predict_comment
from src.context_analyzer import get_context_analyzer, mark_negation_and_context


def test_identity_self_reference_not_toxic():
    # Identity terms used constructively or self-referentially must be Non-Toxic
    samples = [
        "I am a gay man and I love cooking.",
        "He is a muslim scholar who wrote this book.",
        "As a black woman, I find this topic important.",
        "My best friend is jewish and we celebrate together."
    ]
    for s in samples:
        res = predict_comment(s)
        assert res["status"] == "success"
        assert res["is_toxic"] is False, f"Failed on '{s}' - falsely flagged as toxic"
        assert res["classification"] == "Non-Toxic"
        assert len(res["flagged_labels"]) == 0


def test_negated_insults_not_toxic():
    # Negation governing negative terms must prevent toxic classification
    samples = [
        "I do not think you are stupid at all.",
        "You are not an idiot, you are learning.",
        "This project is not terrible, it is actually quite good."
    ]
    for s in samples:
        res = predict_comment(s)
        assert res["status"] == "success"
        assert res["is_toxic"] is False, f"Failed on '{s}' - falsely flagged as toxic"
        assert res["classification"] == "Non-Toxic"
        assert "insult" not in res["flagged_labels"]


def test_colloquial_praise_and_intensifiers_not_toxic():
    # Words like damn, killed, badass used as positive intensifiers/idioms must be Non-Toxic
    samples = [
        "This movie is damn good!",
        "I killed it on my exam today!",
        "That guitar solo was badass!",
        "Hell yeah, this is an awesome tutorial!"
    ]
    for s in samples:
        res = predict_comment(s)
        assert res["status"] == "success"
        assert res["is_toxic"] is False, f"Failed on '{s}' - falsely flagged as toxic"
        assert res["classification"] == "Non-Toxic"


def test_self_deprecating_not_toxic():
    # Self-directed remarks without interpersonal hostility must be Non-Toxic
    res = predict_comment("I feel so stupid today for losing my keys.")
    assert res["status"] == "success"
    assert res["is_toxic"] is False
    assert res["classification"] == "Non-Toxic"


def test_descriptive_mention_not_toxic():
    # Benign descriptive reference to violent topics without threat
    res = predict_comment("This is a movie about a killer.")
    assert res["status"] == "success"
    assert res["is_toxic"] is False
    assert res["classification"] == "Non-Toxic"


def test_targeted_toxicity_correctly_classified():
    # Genuine targeted insults, hostility, and threats must be flagged as Toxic
    toxic_samples = [
        ("You are a stupid idiot.", "toxic"),
        ("Shut up and die, you moron!", "toxic"),
        ("I will hunt you down and hurt you.", "threat")
    ]
    for text, expected_label in toxic_samples:
        res = predict_comment(text)
        assert res["status"] == "success"
        assert res["is_toxic"] is True, f"Failed to identify toxicity in '{text}'"
        assert res["classification"] == "Toxic"
        assert expected_label in res["flagged_labels"]


def test_mark_negation_scoping():
    # Verify negation token tagging
    tagged = mark_negation_and_context("I do not think you are stupid at all.")
    assert "NOT_stupid" in tagged
    assert "NOT_think" in tagged or "not" in tagged
