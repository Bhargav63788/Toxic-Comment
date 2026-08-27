"""
web/app.py

All-in-one Flask backend:
  - Serves the dashboard page
  - Runs a background thread that polls YouTube comments for
    whichever video the user pastes in, and scores them live
  - Auto-recovers from network errors instead of crashing
  - /api/start lets the browser change which video is being watched

Run:
  python web/app.py
Then open:
  http://127.0.0.1:5000
"""

import os
import re
import csv
import time
import threading
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from googleapiclient.discovery import build

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "toxicity_model_mitigated.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "models", "tfidf_vectorizer_mitigated.pkl")
CSV_PATH = os.path.join(BASE_DIR, "..", "dashboard", "live_predictions.csv")
POLL_INTERVAL = 15

DEFAULT_API_KEY = "AIzaSyC18uJp6Vxa7zSBGHL72m9FPUaFQ7aQqNM"

IDENTITY_KEYWORDS = {
    "female": ["woman", "women", "girl", "she", "her", "female"],
    "male": ["man", "men", "boy", "he", "him", "male"],
    "muslim": ["muslim", "islam", "islamic"],
    "christian": ["christian", "christianity", "catholic"],
    "jewish": ["jewish", "jew", "judaism"],
    "homosexual_gay_or_lesbian": ["gay", "lesbian", "homosexual", "lgbt"],
    "black": ["black person", "black people", "african american"],
    "white": ["white person", "white people"],
    "psychiatric_or_mental_illness": ["mental illness", "depression", "schizophrenia", "psychiatric"],
}

app = Flask(__name__)

print("Loading trained model ...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
print("Model loaded.")

# ---------------- Global live state ----------------
state = {
    "video_id": None,
    "api_key": None,
    "seen_ids": set(),
    "status": "idle",   # idle | polling | error
    "last_error": None,
}
state_lock = threading.Lock()


def extract_video_id(url_or_id):
    url_or_id = url_or_id.strip()
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url_or_id)
        if m:
            return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    return None


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_identities(text):
    text_lower = text.lower()
    flags = []
    for identity, keywords in IDENTITY_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(k)}\b", text_lower) for k in keywords):
            flags.append(identity)
    return flags


def ensure_csv():
    if not os.path.exists(CSV_PATH):
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "comment_id", "video_id", "author", "text", "clean_text",
                "toxicity_prob", "is_toxic", "identity_flags", "scored_at",
            ])


def background_poller():
    ensure_csv()
    while True:
        with state_lock:
            video_id = state["video_id"]
            api_key = state["api_key"]

        if not video_id or not api_key:
            time.sleep(2)
            continue

        try:
            youtube = build("youtube", "v3", developerKey=api_key)
            response = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=50,
                order="time", textFormat="plainText",
            ).execute()

            new_count = 0
            for item in response.get("items", []):
                comment_id = item["snippet"]["topLevelComment"]["id"]
                with state_lock:
                    if video_id != state["video_id"]:
                        break  # user switched videos mid-fetch
                    if comment_id in state["seen_ids"]:
                        continue
                    state["seen_ids"].add(comment_id)

                snippet = item["snippet"]["topLevelComment"]["snippet"]
                raw_text = snippet.get("textDisplay", "")
                cleaned = clean_text(raw_text)

                vec = vectorizer.transform([cleaned])
                toxicity_prob = float(model.predict_proba(vec)[0, 1])
                is_toxic = toxicity_prob >= 0.5
                identity_flags = detect_identities(raw_text)

                with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        comment_id, video_id, snippet.get("authorDisplayName"),
                        raw_text, cleaned, round(toxicity_prob, 4), is_toxic,
                        ",".join(identity_flags), datetime.utcnow().isoformat(),
                    ])
                new_count += 1

            with state_lock:
                state["status"] = "polling"
                state["last_error"] = None
            print(f"[{datetime.utcnow().isoformat()}] Scored {new_count} new comments for {video_id}")

        except Exception as e:
            with state_lock:
                state["status"] = "error"
                state["last_error"] = str(e)
            print(f"Polling error (will retry): {e}")

        time.sleep(POLL_INTERVAL)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(force=True)
    url_or_id = data.get("url", "")

    video_id = extract_video_id(url_or_id)
    if not video_id:
        return jsonify({"ok": False, "error": "Could not find a valid video ID in that link."}), 400

    with state_lock:
        state["video_id"] = video_id
        state["api_key"] = DEFAULT_API_KEY
        state["seen_ids"] = set()
        state["status"] = "polling"
        state["last_error"] = None

    return jsonify({"ok": True, "video_id": video_id})


@app.route("/api/data")
def api_data():
    with state_lock:
        active_video_id = state["video_id"]
        status = state["status"]
        last_error = state["last_error"]

    if not active_video_id or not os.path.exists(CSV_PATH):
        return jsonify({
            "total": 0, "toxicity_rate": 0, "identity_count": 0,
            "video_id": active_video_id, "comments": [], "identity_rates": [],
            "status": status, "last_error": last_error,
        })

    df = pd.read_csv(CSV_PATH)
    df["identity_flags"] = df["identity_flags"].fillna("")
    df = df[df["video_id"] == active_video_id]

    filter_mode = request.args.get("filter", "all")
    total = len(df)
    toxicity_rate = round(float(df["is_toxic"].mean()) * 100, 1) if total else 0
    identity_count = int((df["identity_flags"] != "").sum())

    view_df = df.sort_values("scored_at", ascending=False)
    if filter_mode == "toxic":
        view_df = view_df[view_df["is_toxic"] == True]
    elif filter_mode == "clean":
        view_df = view_df[view_df["is_toxic"] == False]

    comments = view_df.head(50)[[
        "author", "text", "toxicity_prob", "is_toxic", "identity_flags", "scored_at"
    ]].to_dict(orient="records")

    exploded = df.assign(identity_flags=df["identity_flags"].str.split(",")).explode("identity_flags")
    exploded = exploded[exploded["identity_flags"].fillna("") != ""]
    identity_rates = []
    if not exploded.empty:
        grouped = exploded.groupby("identity_flags")["is_toxic"].mean().sort_values(ascending=False)
        identity_rates = [{"identity": k, "rate": round(float(v) * 100, 1)} for k, v in grouped.items()]

    return jsonify({
        "total": total,
        "toxicity_rate": toxicity_rate,
        "identity_count": identity_count,
        "video_id": active_video_id,
        "comments": comments,
        "identity_rates": identity_rates,
        "status": status,
        "last_error": last_error,
    })


@app.route("/api/video_meta")
def api_video_meta():
    with state_lock:
        video_id = state["video_id"]
        api_key = state["api_key"]

    if not video_id or not api_key:
        return jsonify({})

    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        resp = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
        items = resp.get("items", [])
        if not items:
            return jsonify({})
        snippet = items[0]["snippet"]
        stats = items[0].get("statistics", {})
        return jsonify({
            "title": snippet.get("title"),
            "description": snippet.get("description", ""),
            "channel": snippet.get("channelTitle"),
            "views": stats.get("viewCount"),
            "likes": stats.get("likeCount"),
        })
    except Exception:
        return jsonify({})


if __name__ == "__main__":
    poller_thread = threading.Thread(target=background_poller, daemon=True)
    poller_thread.start()
    app.run(debug=False, port=5000)