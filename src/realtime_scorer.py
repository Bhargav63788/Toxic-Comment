"""
realtime_scorer.py

Real-time pipeline WITHOUT Kafka (direct polling version):
  1. Poll YouTube Data API for new comments on a video
  2. Score each comment with the trained model (toxicity_model_mitigated.pkl)
  3. Tag identity mentions (for fairness monitoring)
  4. Append results to a CSV that the dashboard reads live

This does the same job Kafka would (decouple ingestion from scoring)
but in one simple script — good fit for limited local resources.

Run:
  python src/realtime_scorer.py --video_id VIDEO_ID --api_key YOUR_KEY
"""

import argparse
import re
import time
import os
import csv
import joblib
from datetime import datetime

from googleapiclient.discovery import build

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


def ensure_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "comment_id", "video_id", "author", "text", "clean_text",
                "toxicity_prob", "is_toxic", "identity_flags", "scored_at",
            ])


def fetch_comments(youtube, video_id, page_token=None, max_results=50):
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        pageToken=page_token,
        order="time",
        textFormat="plainText",
    )
    return request.execute()


def main(args):
    print("Loading trained model ...")
    model = joblib.load(args.model_path)
    vectorizer = joblib.load(args.vectorizer_path)

    youtube = build("youtube", "v3", developerKey=args.api_key)
    ensure_csv(args.output_csv)

    seen_comment_ids = set()
    print(f"Polling video {args.video_id} every {args.poll_interval}s ... (Ctrl+C to stop)")

    try:
        while True:
            response = fetch_comments(youtube, args.video_id)
            new_count = 0

            for item in response.get("items", []):
                comment_id = item["snippet"]["topLevelComment"]["id"]
                if comment_id in seen_comment_ids:
                    continue
                seen_comment_ids.add(comment_id)

                snippet = item["snippet"]["topLevelComment"]["snippet"]
                raw_text = snippet.get("textDisplay", "")
                cleaned = clean_text(raw_text)

                vec = vectorizer.transform([cleaned])
                toxicity_prob = float(model.predict_proba(vec)[0, 1])
                is_toxic = toxicity_prob >= args.threshold
                identity_flags = detect_identities(raw_text)

                row = [
                    comment_id, args.video_id, snippet.get("authorDisplayName"),
                    raw_text, cleaned, round(toxicity_prob, 4), is_toxic,
                    ",".join(identity_flags), datetime.utcnow().isoformat(),
                ]

                with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                new_count += 1
                tag = f" [identities: {', '.join(identity_flags)}]" if identity_flags else ""
                print(f"toxic={is_toxic} prob={toxicity_prob:.3f}{tag} | {raw_text[:80]}")

            print(f"[{datetime.utcnow().isoformat()}] Scored {new_count} new comments "
                  f"(total seen: {len(seen_comment_ids)})")
            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", type=str, required=True,
                         help="YouTube video ID (the part after v= in the URL)")
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="models/toxicity_model_mitigated.pkl")
    parser.add_argument("--vectorizer_path", type=str, default="models/tfidf_vectorizer_mitigated.pkl")
    parser.add_argument("--output_csv", type=str, default="dashboard/live_predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--poll_interval", type=int, default=15)
    args = parser.parse_args()
    main(args)