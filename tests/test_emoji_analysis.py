"""
tests/test_emoji_analysis.py

Comprehensive test suite verifying:
  - Emoji preservation during preprocessing
  - Emoji-only toxic and clean comments
  - Text + emoji combinations
  - Neutral comments containing emojis
  - Sarcastic and ironic comments
  - Kannada, Kanglish, and code-mixed comments with emojis
  - Retention of all six toxicity labels and four severity levels
  - Model comparison between baseline text-only and emoji-enhanced models
  - YouTube comment batch analysis with emoji metrics
"""

import pytest
from src.preprocessing import clean_text, validate_input_comment, DEFAULT_LABELS
from src.emoji_analyzer import (
    extract_emojis,
    has_emojis,
    is_emoji_only,
    get_emoji_analyzer,
    compare_predictions
)
from src.predict import predict_comment, load_model_artifacts
from src.youtube_analyzer import analyze_youtube_comments, DEMO_COMMENTS


@pytest.fixture(scope="module")
def artifacts():
    return load_model_artifacts()


# -----------------------------------------------------------------------------
# 1. Preprocessing & Emoji Preservation
# -----------------------------------------------------------------------------

def test_clean_text_preserves_emojis():
    raw = "Hello World! 🤡 This is a test comment 🖕 with Kannada ಬೆಂಕಿ 🔥."
    cleaned = clean_text(raw)
    assert "🤡" in cleaned
    assert "🖕" in cleaned
    assert "🔥" in cleaned
    assert "ಬೆಂಕಿ" in cleaned
    assert "hello world" in cleaned
    assert "!" not in cleaned


def test_clean_text_emoji_only():
    raw = "🖕🖕🖕"
    cleaned = clean_text(raw)
    assert "🖕" in cleaned
    assert len(cleaned) > 0


def test_emoji_utility_functions():
    assert has_emojis("Good job 👍") is True
    assert has_emojis("No emojis here") is False
    assert is_emoji_only("🖕🖕🖕") is True
    assert is_emoji_only("🤡 ❤️") is True
    assert is_emoji_only("Hello 🤡") is False
    assert is_emoji_only("123 🤡") is False

    extracted = extract_emojis("You are a 🤡 and 💩")
    assert len(extracted) == 2
    assert extracted[0]["emoji"] == "🤡"
    assert extracted[1]["emoji"] == "💩"


# -----------------------------------------------------------------------------
# 2. Emoji-Only Comments (Toxic vs. Clean)
# -----------------------------------------------------------------------------

def test_emoji_only_toxic_obscene(artifacts):
    res = predict_comment("🖕🖕🖕", artifacts=artifacts)
    assert res["is_toxic"] is True
    assert res["classification"] == "Toxic"
    assert "obscene" in res["flagged_labels"]
    assert "toxic" in res["flagged_labels"]
    assert res["severity"] == "High"
    assert res["is_emoji_only"] is True


def test_emoji_only_toxic_insult(artifacts):
    res = predict_comment("🤡🤡", artifacts=artifacts)
    assert res["is_toxic"] is True
    assert "insult" in res["flagged_labels"]
    assert res["severity"] in ("Medium", "High")


def test_emoji_only_threat(artifacts):
    res = predict_comment("🔪🩸☠️", artifacts=artifacts)
    assert res["is_toxic"] is True
    assert "threat" in res["flagged_labels"]
    assert res["severity"] == "High"


def test_emoji_only_clean_hearts(artifacts):
    res = predict_comment("❤️❤️❤️", artifacts=artifacts)
    assert res["is_toxic"] is False
    assert res["classification"] == "Non-Toxic"
    assert res["severity"] == "Clean"
    assert len(res["flagged_labels"]) == 0


def test_emoji_only_clean_respect_namaskara(artifacts):
    res = predict_comment("🙏🙏", artifacts=artifacts)
    assert res["is_toxic"] is False
    assert res["severity"] == "Clean"


def test_emoji_only_clean_laughter(artifacts):
    res = predict_comment("😂🤣", artifacts=artifacts)
    assert res["is_toxic"] is False
    assert res["severity"] == "Clean"


# -----------------------------------------------------------------------------
# 3. Text + Emoji Comments
# -----------------------------------------------------------------------------

def test_text_plus_emoji_insult(artifacts):
    res = predict_comment("You are an idiot 🤡🖕", artifacts=artifacts)
    assert res["is_toxic"] is True
    assert "insult" in res["flagged_labels"]
    assert "toxic" in res["flagged_labels"]
    assert res["severity"] in ("Medium", "High")
    assert len(res["emojis_detected"]) >= 2


def test_text_plus_emoji_threat(artifacts):
    res = predict_comment("I will find you 🔪🩸", artifacts=artifacts)
    assert res["is_toxic"] is True
    assert "threat" in res["flagged_labels"]
    assert res["severity"] == "High"


# -----------------------------------------------------------------------------
# 4. Neutral Comments with Emojis
# -----------------------------------------------------------------------------

def test_neutral_comment_with_document_emoji(artifacts):
    res = predict_comment("Here is the documentation link for the dataset 📄👍", artifacts=artifacts)
    assert res["is_toxic"] is False
    assert res["classification"] == "Non-Toxic"
    assert res["severity"] == "Clean"


def test_neutral_comment_with_question_emoji(artifacts):
    res = predict_comment("Could you clarify how the loss function is computed? 🤔", artifacts=artifacts)
    assert res["is_toxic"] is False
    assert res["severity"] == "Clean"


def test_neutral_comment_with_time_emoji(artifacts):
    res = predict_comment("The meeting will start tomorrow at 10 AM ⏰", artifacts=artifacts)
    assert res["is_toxic"] is False
    assert res["severity"] == "Clean"


# -----------------------------------------------------------------------------
# 5. Sarcastic & Mocking Comments
# -----------------------------------------------------------------------------

def test_sarcastic_comment_genius_clown(artifacts):
    res = predict_comment("What a genius 🤡", artifacts=artifacts)
    assert res["is_toxic"] is True
    assert res["sarcasm_detected"] is True
    assert "insult" in res["flagged_labels"]


def test_sarcastic_comment_brilliant_trash(artifacts):
    res = predict_comment("Brilliant work 🙃🗑️ you broke everything", artifacts=artifacts)
    assert res["is_toxic"] is True
    assert res["sarcasm_detected"] is True
    assert "insult" in res["flagged_labels"]


def test_sarcastic_comment_failing_clapping(artifacts):
    res = predict_comment("Great job failing the build again 👏😂", artifacts=artifacts)
    assert res["is_toxic"] is True
    assert "insult" in res["flagged_labels"]


# -----------------------------------------------------------------------------
# 6. Kannada, Kanglish & Code-Mixed Comments with Emojis
# -----------------------------------------------------------------------------

def test_kannada_praise_benki_fire(artifacts):
    # 'ಬೆಂಕಿ' + 🔥 in Kannada is high colloquial praise, NOT a threat/arson
    res = predict_comment("ವಿಡಿಯೋ ಬೆಂಕಿ ಗುರು 🔥🔥 ಅದ್ಭುತ ವಿವರಣೆ!", artifacts=artifacts)
    assert res["is_toxic"] is False
    assert res["classification"] == "Non-Toxic"
    assert "threat" not in res["flagged_labels"]
    assert res["severity"] == "Clean"


def test_kannada_gratitude_namaskara(artifacts):
    res = predict_comment("ತುಂಬಾ ಧನ್ಯವಾದಗಳು ಗುರುಗಳೇ 🙏❤️", artifacts=artifacts)
    assert res["is_toxic"] is False
    assert res["classification"] == "Non-Toxic"
    assert res["severity"] == "Clean"


def test_kannada_negated_insult_with_friendly_emoji(artifacts):
    res = predict_comment("ನೀನು ದಡ್ಡ ಅಲ್ಲ, ತುಂಬಾ ಚೆನ್ನಾಗಿ ಮಾಡಿದ್ದೀಯ 😊❤️", artifacts=artifacts)
    assert res["is_toxic"] is False
    assert res["classification"] == "Non-Toxic"
    assert res["severity"] == "Clean"


def test_kannada_insult_with_clown_emoji(artifacts):
    res = predict_comment("ನೀನು ಒಬ್ಬ ಮೂರ್ಖ ಕತ್ತೆ 🤡 ಬಾಯಿ ಮುಚ್ಕೋ 🤫🖕", artifacts=artifacts)
    assert res["is_toxic"] is True
    assert "insult" in res["flagged_labels"]
    assert res["severity"] in ("Medium", "High")


def test_kannada_threat_with_knife_blood(artifacts):
    res = predict_comment("ಮುಗಿಸ್ತೀನಿ ನಿನ್ನ 🔪🩸", artifacts=artifacts)
    assert res["is_toxic"] is True
    assert "threat" in res["flagged_labels"]
    assert res["severity"] == "High"


def test_kanglish_insult_with_clown(artifacts):
    res = predict_comment("neenu obba huccha rascal 🤡 waste fellow", artifacts=artifacts)
    assert res["is_toxic"] is True
    assert "insult" in res["flagged_labels"]


# -----------------------------------------------------------------------------
# 7. Six Toxicity Labels & Four Severity Levels
# -----------------------------------------------------------------------------

def test_retains_all_six_toxicity_labels(artifacts):
    res = predict_comment("Test comment 🤡", artifacts=artifacts)
    predictions = res["predictions"]
    for label in DEFAULT_LABELS:
        assert label in predictions
        assert "probability" in predictions[label]
        assert "percentage" in predictions[label]
        assert "threshold" in predictions[label]
        assert "flagged" in predictions[label]
        assert "risk_level" in predictions[label]


def test_retains_four_severity_levels(artifacts):
    valid_severities = {"Clean", "Low", "Medium", "High"}

    # Clean
    res_clean = predict_comment("Great tutorial! ❤️", artifacts=artifacts)
    assert res_clean["severity"] in valid_severities
    assert res_clean["severity"] == "Clean"

    # High
    res_high = predict_comment("I will hunt and murder you 🔪🩸☠️", artifacts=artifacts)
    assert res_high["severity"] in valid_severities
    assert res_high["severity"] == "High"

    # Medium
    res_med = predict_comment("Shut up 🤫", artifacts=artifacts)
    assert res_med["severity"] in valid_severities
    assert res_med["severity"] == "Medium"


# -----------------------------------------------------------------------------
# 8. Model Comparison Utility
# -----------------------------------------------------------------------------

def test_model_comparison_recovers_emoji_only_false_negative(artifacts):
    res = predict_comment("🖕🖕🖕", artifacts=artifacts, return_comparison=True)
    assert "model_comparison" in res
    comp = res["model_comparison"]
    assert comp["verdict_changed"] is True
    assert comp["base_model"]["is_toxic"] is False
    assert comp["enhanced_model"]["is_toxic"] is True
    assert "False Negative" in comp["impact_type"]


def test_model_comparison_detects_sarcasm(artifacts):
    res = predict_comment("What a genius 🤡", artifacts=artifacts, return_comparison=True)
    comp = res["model_comparison"]
    assert comp["enhanced_model"]["is_toxic"] is True
    assert comp["base_model"]["is_toxic"] is False
    assert comp["verdict_changed"] is True


# -----------------------------------------------------------------------------
# 9. YouTube Batch Comment Analysis with Emojis
# -----------------------------------------------------------------------------

def test_youtube_analyzer_with_emojis(artifacts):
    comments_sample = [
        {"comment_id": "c1", "text": "Great tutorial! ❤️", "author": "User1"},
        {"comment_id": "c2", "text": "🖕🖕🖕", "author": "Hater1"},
        {"comment_id": "c3", "text": "What a genius 🤡", "author": "Sarcastic1"},
        {"comment_id": "c4", "text": "ಬೆಂಕಿ ಗುರು 🔥🔥", "author": "UserKN"}
    ]
    results = analyze_youtube_comments(comments_sample, artifacts)
    assert results["total_analyzed"] == 4
    assert results["toxic_count"] >= 2  # c2 (🖕) and c3 (🤡)
    assert "emoji_stats" in results
    assert results["emoji_stats"]["comments_with_emojis"] == 4
    assert results["emoji_stats"]["emoji_toxic_count"] >= 2

    # Verify per-comment emoji chips are attached
    analyzed = results["analyzed_comments"]
    c2_res = next(c for c in analyzed if c["comment_id"] == "c2")
    assert c2_res["has_emojis"] is True
    assert c2_res["is_emoji_only"] is True
    assert len(c2_res["emoji_ui_chips"]) > 0


def test_emoji_and_text_comments_classified_across_sections(artifacts):
    """
    Test with text-only, emoji-only, and text-plus-emoji comments.
    Verify:
      - Every analyzed comment is classified as Toxic or Non-Toxic based on the model.
      - Emoji detection works independently (has_emojis=True/False).
      - Comments containing emojis appear in Toxic or Non-Toxic according to prediction.
      - Comment IDs and analysis results are strictly preserved without duplicates.
    """
    mixed_comments = [
        # 1. Text-only clean
        {"comment_id": "comm_1", "author": "Alice", "text": "This tutorial is well structured and very informative."},
        # 2. Text-only toxic
        {"comment_id": "comm_2", "author": "Bob", "text": "You are a complete idiot and your work is garbage."},
        # 3. Emoji-only clean
        {"comment_id": "comm_3", "author": "Carol", "text": "❤️❤️❤️"},
        # 4. Emoji-only toxic
        {"comment_id": "comm_4", "author": "Dave", "text": "🖕🖕🖕"},
        # 5. Text + emoji clean
        {"comment_id": "comm_5", "author": "Eve", "text": "Thank you so much! Really helped me pass my ML exam ❤️✨👍"},
        # 6. Text + emoji toxic
        {"comment_id": "comm_6", "author": "Frank", "text": "What an absolute genius 🤡🗑️ you should be proud of this disaster"}
    ]

    res = analyze_youtube_comments(mixed_comments, artifacts)
    analyzed = res["analyzed_comments"]
    assert len(analyzed) == 6

    # Verify unique comment IDs are preserved
    ids = [c["comment_id"] for c in analyzed]
    assert len(ids) == len(set(ids))
    assert ids == ["comm_1", "comm_2", "comm_3", "comm_4", "comm_5", "comm_6"]

    # Verify classification and emoji independence
    toxic_comments = [c for c in analyzed if c["is_toxic"] is True]
    clean_comments = [c for c in analyzed if c["is_toxic"] is False]
    emoji_comments = [c for c in analyzed if c["has_emojis"] is True]

    # Every comment is either Toxic or Non-Toxic
    assert len(toxic_comments) + len(clean_comments) == len(analyzed)

    # Toxic section must include both text-only (comm_2) and emoji-containing (comm_4, comm_6)
    toxic_ids = [c["comment_id"] for c in toxic_comments]
    assert "comm_2" in toxic_ids  # Text-only toxic
    assert "comm_4" in toxic_ids  # Emoji-only toxic
    assert "comm_6" in toxic_ids  # Text+emoji toxic

    # Clean section must include both text-only (comm_1) and emoji-containing (comm_3, comm_5)
    clean_ids = [c["comment_id"] for c in clean_comments]
    assert "comm_1" in clean_ids  # Text-only clean
    assert "comm_3" in clean_ids  # Emoji-only clean
    assert "comm_5" in clean_ids  # Text+emoji clean

    # Emojis section contains all emoji-bearing comments, each with Toxic or Non-Toxic label
    emoji_ids = [c["comment_id"] for c in emoji_comments]
    assert emoji_ids == ["comm_3", "comm_4", "comm_5", "comm_6"]
    for ec in emoji_comments:
        assert ec["classification"] in ["Toxic", "Non-Toxic"]
        assert ec["comment_id"] in ["comm_3", "comm_4", "comm_5", "comm_6"]


def test_mixed_comments_derisive_mockery_classification(artifacts):
    """
    Verify mixed comments such as 'You are useless 😂' are analyzed using both
    text and emoji context, appearing in both Toxic and Emojis sections.
    """
    res_plain = predict_comment("You are useless", artifacts=artifacts)
    assert res_plain["is_toxic"] is True
    assert res_plain["classification"] == "Toxic"
    assert "insult" in res_plain["flagged_labels"]

    res_mixed = predict_comment("You are useless 😂", artifacts=artifacts)
    assert res_mixed["is_toxic"] is True
    assert res_mixed["classification"] == "Toxic"
    assert res_mixed["has_emojis"] is True
    assert res_mixed["sarcasm_detected"] is True
    assert "insult" in res_mixed["flagged_labels"]
    assert "toxic" in res_mixed["flagged_labels"]

    # Verify batch placement in both Toxic and Emojis sections
    batch = [
        {"comment_id": "u1", "author": "User", "text": "You are useless 😂"}
    ]
    batch_res = analyze_youtube_comments(batch, artifacts)
    c = batch_res["analyzed_comments"][0]
    assert c["is_toxic"] is True
    assert c["classification"] == "Toxic"
    assert c["has_emojis"] is True
    assert batch_res["toxic_count"] == 1
    assert batch_res["emoji_stats"]["comments_with_emojis"] == 1
    assert batch_res["emoji_stats"]["emoji_toxic_count"] == 1


