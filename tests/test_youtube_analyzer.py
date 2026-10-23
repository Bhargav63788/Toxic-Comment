"""
tests/test_youtube_analyzer.py

Unit and integration tests for the YouTube Toxic Comment Analyzer:
  - Video ID extraction from multiple YouTube URL formats
  - Exact percentage calculations and zero-division safety
  - Batch comment analysis with the trained multi-label ML model
  - Flask endpoint integration (/api/analyze_youtube)
"""

import pytest
import json
from src.youtube_analyzer import extract_video_id, analyze_youtube_comments
from src.predict import load_model_artifacts
from web.app import app


# -----------------------------------------------------------------------------
# 1. Video ID Extraction Tests
# -----------------------------------------------------------------------------

def test_extract_video_id_standard_watch():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_extract_video_id_short_link():
    url = "https://youtu.be/dQw4w9WgXcQ"
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_extract_video_id_embed():
    url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_extract_video_id_shorts():
    url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_extract_video_id_with_query_params():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&feature=share"
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_extract_video_id_raw_id():
    raw_id = "dQw4w9WgXcQ"
    assert extract_video_id(raw_id) == "dQw4w9WgXcQ"


def test_extract_video_id_invalid():
    assert extract_video_id("https://google.com") is None
    assert extract_video_id("https://example.com/not-youtube") is None
    assert extract_video_id("") is None
    assert extract_video_id(None) is None


# -----------------------------------------------------------------------------
# 2. Percentage and Calculation Tests
# -----------------------------------------------------------------------------

def test_analyze_empty_comments():
    artifacts = load_model_artifacts("models")
    res = analyze_youtube_comments([], artifacts)
    assert res["total_analyzed"] == 0
    assert res["toxic_count"] == 0
    assert res["non_toxic_count"] == 0
    assert res["toxic_percentage"] == 0.0
    assert res["non_toxic_percentage"] == 0.0
    assert len(res["analyzed_comments"]) == 0


def test_analyze_batch_comments():
    artifacts = load_model_artifacts("models")
    sample_comments = [
        {"comment_id": "1", "author": "Alice", "text": "This is great, thank you!"},
        {"comment_id": "2", "author": "Bob", "text": "You are a total idiot and a moron!"},
        {"comment_id": "3", "author": "Charlie", "text": "Wonderful tutorial, very clear."}
    ]
    res = analyze_youtube_comments(sample_comments, artifacts)
    assert res["total_analyzed"] == 3
    assert res["toxic_count"] >= 1
    assert res["non_toxic_count"] >= 1
    # Check that percentages sum to 100.0 (or rounded to 100.0)
    assert abs((res["toxic_percentage"] + res["non_toxic_percentage"]) - 100.0) < 0.1
    # Ensure each comment has required fields
    for c in res["analyzed_comments"]:
        assert "is_toxic" in c
        assert "classification" in c
        assert c["classification"] in ["Toxic", "Non-Toxic"]
        assert "category_scores" in c


# -----------------------------------------------------------------------------
# 3. API Integration Tests
# -----------------------------------------------------------------------------

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_analyze_youtube_invalid_url(client):
    res = client.post(
        "/api/analyze_youtube",
        data=json.dumps({"url": "https://google.com"}),
        content_type="application/json"
    )
    assert res.status_code == 400
    data = json.loads(res.data)
    assert data["status"] == "error"
    assert "invalid" in data["message"].lower()


def test_api_analyze_youtube_missing_url(client):
    res = client.post(
        "/api/analyze_youtube",
        data=json.dumps({"url": ""}),
        content_type="application/json"
    )
    assert res.status_code == 400


def test_api_analyze_youtube_demo_execution(client):
    # Test built-in demo video processing
    res = client.post(
        "/api/analyze_youtube",
        data=json.dumps({"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "max_comments": 10}),
        content_type="application/json"
    )
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["status"] == "success"
    assert data["video_id"] == "dQw4w9WgXcQ"
    assert data["total_analyzed"] > 0
    assert "toxic_percentage" in data
    assert "non_toxic_percentage" in data
    assert "comments" in data
    assert "disclaimer" in data


def test_api_analyze_youtube_high_volume(client):
    # Test high-volume retrieval (e.g. 50 comments) with reply extraction
    res = client.post(
        "/api/analyze_youtube",
        data=json.dumps({
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "max_comments": 50,
            "include_replies": True
        }),
        content_type="application/json"
    )
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["status"] == "success"
    assert data["total_analyzed"] <= 50
    assert data["total_analyzed"] > 0
    assert len(data["comments"]) == data["total_analyzed"]
    assert "is_reply" in data["comments"][0]


def test_fetch_youtube_comments_up_to_5000():
    from src.youtube_analyzer import fetch_youtube_comments_and_meta
    comments, meta, err = fetch_youtube_comments_and_meta(
        video_id="demo_test",
        api_key=None,
        max_comments=5000,
        include_replies=True,
        use_demo_if_missing_key=True
    )
    assert err is None
    assert len(comments) == 5000
    assert meta["is_demo"] is True
