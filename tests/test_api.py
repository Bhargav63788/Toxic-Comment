"""
tests/test_api.py

Integration tests for the Flask web application and API endpoints.
"""

import pytest
import json
from web.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Toxic Comment" in response.data


def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert len(data["labels"]) == 6


def test_api_model_info(client):
    response = client.get("/api/model_info")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    assert "dataset" in data
    assert "model_architecture" in data
    assert "evaluation_metrics" in data
    assert "https://www.kaggle.com" in data["dataset"]["primary_source"]
    assert "https://huggingface.co" in data["dataset"]["mirror_source"]


def test_api_predict_clean(client):
    payload = {"text": "What a beautiful and sunny day outside."}
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    assert data["is_toxic"] is False
    assert data["severity"] == "Clean"
    assert len(data["predictions"]) == 6


def test_api_predict_toxic(client):
    payload = {"text": "You are stupid, disgusting, and horrible."}
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    assert data["is_toxic"] is True
    assert len(data["flagged_labels"]) >= 1


def test_api_predict_empty_text(client):
    payload = {"text": "   "}
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data["status"] == "error"


def test_api_predict_missing_text_field(client):
    payload = {"wrong_key": "some text"}
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data["status"] == "error"


def test_api_predict_oversized_text(client):
    payload = {"text": "a " * 3000}  # 6000 chars > 5000 limit
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 413
    data = json.loads(response.data)
    assert data["status"] == "error"


def test_api_predict_non_json(client):
    response = client.post(
        "/api/predict",
        data="plain text body",
        content_type="text/plain"
    )
    assert response.status_code == 400
