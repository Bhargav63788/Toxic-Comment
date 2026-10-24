"""
src/youtube_analyzer.py

Dedicated service for YouTube Toxic Comment Detection.
Handles:
  - Video URL validation and Video ID extraction
  - Fetching comments and video metadata via official YouTube Data API v3
  - Multi-label toxicity classification using the trained ML model with sentence context
  - High-volume pagination allowing analysis of all accessible comments without arbitrary limits
  - Exact percentage calculations and statistical summaries
  - Graceful error handling (disabled comments, quota exceeded, invalid links)
"""

import os
import re
import html
from typing import Dict, Any, List, Optional, Tuple
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.predict import load_model_artifacts
from src.preprocessing import clean_text, DEFAULT_LABELS
from src.context_analyzer import get_context_analyzer, mark_negation_and_context
from src.kannada_analyzer import (
    detect_language,
    load_kannada_artifacts,
    get_kannada_context_analyzer
)

# Built-in demo comments for testing without requiring an immediate Google API key
DEMO_COMMENTS = [
    {
        "comment_id": "demo_1",
        "author": "TechExplorer",
        "author_avatar": "",
        "text": "This tutorial was exceptionally well explained! I finally understand how gradient descent works under the hood. Thank you so much for putting this together.",
        "like_count": 42
    },
    {
        "comment_id": "demo_2",
        "author": "AngryViewer99",
        "author_avatar": "",
        "text": "You are a complete idiot and your code is absolute garbage. Stop making videos you moron, nobody wants to watch this trash.",
        "like_count": 2
    },
    {
        "comment_id": "demo_3",
        "author": "CodeNewbie",
        "author_avatar": "",
        "text": "Great pacing and clear slides. Could you please cover transformers or multi-head attention in the next session?",
        "like_count": 15
    },
    {
        "comment_id": "demo_4",
        "author": "ToxicTroll_X",
        "author_avatar": "",
        "text": "Shut your mouth you stupid clown, I hope someone tracks you down and beats you up.",
        "like_count": 0
    },
    {
        "comment_id": "demo_5",
        "author": "DataScientist101",
        "author_avatar": "",
        "text": "Very clean implementation with scikit-learn! Loved the threshold tuning breakdown.",
        "like_count": 28
    },
    {
        "comment_id": "demo_6",
        "author": "Hater44",
        "author_avatar": "",
        "text": "Worst explanation ever. People from your country are so dumb and shouldn't be allowed on YouTube.",
        "like_count": 1
    },
    {
        "comment_id": "demo_7",
        "author": "CuriousDev",
        "author_avatar": "",
        "text": "Does this approach generalize to multi-lingual comments or is it strictly English?",
        "like_count": 8
    },
    {
        "comment_id": "demo_8",
        "author": "AggressiveGuy",
        "author_avatar": "",
        "text": "You are sickeningly incompetent. Disgusting waste of time.",
        "like_count": 0
    },
    {
        "comment_id": "demo_9",
        "author": "ElenaR",
        "author_avatar": "",
        "text": "Bookmarked! Running the scripts locally on Windows right now and it works like a charm.",
        "like_count": 19
    },
    {
        "comment_id": "demo_10",
        "author": "SkepticalViewer",
        "author_avatar": "",
        "text": "I will destroy you and erase your channel from the internet permanently.",
        "like_count": 0
    },
    {
        "comment_id": "demo_11",
        "author": "TechKannadiga",
        "author_avatar": "",
        "text": "ತುಂಬಾ ಧನ್ಯವಾದಗಳು ಗುರುಗಳೇ, ವಿಷಯವನ್ನು ಅತ್ಯಂತ ಸರಳವಾಗಿ ಮತ್ತು ಸ್ಪಷ್ಟವಾಗಿ ವಿವರಿಸಿದ್ದೀರಿ. ಸೂಪರ್ ಮಾಹಿತಿ!",
        "like_count": 34
    },
    {
        "comment_id": "demo_12",
        "author": "KopaGonda_99",
        "author_avatar": "",
        "text": "ನೀನು ಒಬ್ಬ ಮೂರ್ಖ ಮತ್ತು ನಿಷ್ಪ್ರಯೋಜಕ ಕತ್ತೆ. ನಿನಗೆ ಏನೂ ಗೊತ್ತಿಲ್ಲ ಬಾಯಿ ಮುಚ್ಕೋ ಲೋಫರ್.",
        "like_count": 1
    },
    {
        "comment_id": "demo_13",
        "author": "BangaloreCoder",
        "author_avatar": "",
        "text": "tumba chennagide sir! Next video dalli neural networks bagge detailed tutorial madi please.",
        "like_count": 22
    },
    {
        "comment_id": "demo_14",
        "author": "AngryTroll_KN",
        "author_avatar": "",
        "text": "ನಿನ್ನನ್ನು ಕೊಲ್ಲುತ್ತೇನೆ ಹುಷಾರ್, ಎಲ್ಲಾದರೂ ಕಣ್ಣಿಗೆ ಬಿದ್ದರೆ ಮುಗಿಸ್ತೀನಿ ನಿನ್ನ ಕಥೆ.",
        "like_count": 0
    },
    {
        "comment_id": "demo_15",
        "author": "NaveenKumar",
        "author_avatar": "",
        "text": "ನೀನು ದಡ್ಡ ಅಲ್ಲ, ಜನ ಏನೇ ನೆಗೆಟಿವ್ ಹೇಳಿದ್ರೂ ಈ ವಿಡಿಯೋ ಕೆಟ್ಟದಲ್ಲ. ತುಂಬಾ ಚೆನ್ನಾಗಿ ಮಾಡಿದ್ದೀಯ ಮುಂದುವರಿಸಿ.",
        "like_count": 18
    },
    {
        "comment_id": "demo_16",
        "author": "RudeUser55",
        "author_avatar": "",
        "text": "neenu obba huccha rascal, thika muchkondu hogu illinda wasting time.",
        "like_count": 0
    }
]


def extract_video_id(url_or_id: str) -> Optional[str]:
    """
    Extract standard 11-character YouTube video ID from various URL formats:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - Raw 11-char ID
    Returns None if URL is invalid or ID cannot be identified.
    """
    if not url_or_id or not isinstance(url_or_id, str):
        return None

    cleaned = url_or_id.strip()

    # If it is a URL, extract via regex
    if "youtube.com" in cleaned or "youtu.be" in cleaned or cleaned.startswith(("http://", "https://", "www.")):
        # Match standard query parameter: ?v=XXXXXXXXXXX or &v=XXXXXXXXXXX
        v_match = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", cleaned)
        if v_match:
            return v_match.group(1)

        # Match youtu.be/XXXXXXXXXXX, embed/XXXXXXXXXXX, shorts/XXXXXXXXXXX
        path_match = re.search(r"(?:youtu\.be/|/embed/|/shorts/|/v/)([A-Za-z0-9_-]{11})", cleaned)
        if path_match:
            return path_match.group(1)

        # If it was a URL but no YouTube pattern matched, return None
        return None

    # Check if raw string is an 11-char ID
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", cleaned):
        return cleaned

    return None


def fetch_youtube_comments_and_meta(
    video_id: str,
    api_key: Optional[str] = None,
    max_comments: Optional[int] = 5000,
    include_replies: bool = True,
    use_demo_if_missing_key: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    """
    Fetch comments and video metadata using official YouTube Data API v3.
    Extracts up to 5,000 comments per video using dual-strategy pagination (relevance + time)
    to overcome the YouTube API ~2,525 relevance cutoff.
    If a video has fewer than 5,000 accessible comments, retrieves all available comments.
    If it has more than 5,000, retrieves exactly 5,000 comments.
    Returns: (comments_list, video_meta_dict, error_message)
    """
    # Check for API key in args or environment
    key = api_key or os.environ.get("YOUTUBE_API_KEY", "").strip()

    target_limit = int(max_comments) if (max_comments is not None and int(max_comments) > 0) else 5000

    if video_id.startswith("demo") or (not key and use_demo_if_missing_key and video_id == "dQw4w9WgXcQ"):
        # Provide sample demonstration comments up to target_limit
        demo_meta = {
            "title": "Demonstration Video: Machine Learning Content Moderation",
            "channel": "AI Research Lab",
            "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            "description": (
                "Welcome to our deep dive into automated toxicity detection and real-time comment moderation! "
                "In this video, we explore how natural language processing models classify aggressive speech, insults, "
                "and severe toxicity in large online communities. Learn how TF-IDF vectorization and multi-label Logistic "
                "Regression achieve balanced precision and recall on real-world datasets."
            ),
            "view_count": "154200",
            "like_count": "8950",
            "total_comment_count": "1250",
            "published_at": "2024-03-15T12:00:00Z",
            "tags": ["Machine Learning", "NLP", "Content Moderation", "AI", "Toxic Comments"],
            "is_demo": True
        }
        demo_list = []
        while len(demo_list) < target_limit:
            for c in DEMO_COMMENTS:
                c_copy = dict(c)
                c_copy["comment_id"] = f"demo_{len(demo_list) + 1}"
                c_copy["is_reply"] = False
                demo_list.append(c_copy)
                if len(demo_list) >= target_limit:
                    break
        return demo_list[:target_limit], demo_meta, None

    if not key:
        return [], {}, (
            "YouTube Data API Key is not configured. Please set the YOUTUBE_API_KEY "
            "environment variable or provide your API key in the input form."
        )

    try:
        youtube = build("youtube", "v3", developerKey=key, cache_discovery=False)

        # 1. Fetch Video Metadata
        video_meta = {
            "title": f"YouTube Video ({video_id})",
            "channel": "YouTube Channel",
            "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            "description": "",
            "view_count": None,
            "like_count": None,
            "total_comment_count": None,
            "published_at": None,
            "tags": [],
            "is_demo": False
        }

        try:
            video_resp = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
            items = video_resp.get("items", [])
            if items:
                snippet = items[0].get("snippet", {})
                stats = items[0].get("statistics", {})
                video_meta["title"] = snippet.get("title", video_meta["title"])
                video_meta["channel"] = snippet.get("channelTitle", video_meta["channel"])
                video_meta["description"] = snippet.get("description", "")
                video_meta["published_at"] = snippet.get("publishedAt")
                video_meta["tags"] = snippet.get("tags", [])[:8]
                thumbs = snippet.get("thumbnails", {})
                video_meta["thumbnail"] = (
                    thumbs.get("high", {}).get("url") or
                    thumbs.get("medium", {}).get("url") or
                    video_meta["thumbnail"]
                )
                video_meta["view_count"] = stats.get("viewCount")
                video_meta["like_count"] = stats.get("likeCount")
                video_meta["total_comment_count"] = stats.get("commentCount")
            else:
                return [], {}, "Video not found or is private/removed."
        except HttpError:
            pass

        # 2. Fetch Comments up to target_limit (5,000 default) using dual-order pagination
        raw_comments = []
        seen_ids = set()
        part_param = "snippet,replies" if include_replies else "snippet"

        def _add_comment(item_id: str, author: str, avatar: str, text: str, likes: int, published_at: str, is_reply: bool, parent_id: Optional[str] = None) -> bool:
            if not text or not str(text).strip() or item_id in seen_ids:
                return False
            seen_ids.add(item_id)
            raw_comments.append({
                "comment_id": item_id,
                "author": author or "Anonymous",
                "author_avatar": avatar or "",
                "text": text,
                "like_count": likes or 0,
                "published_at": published_at or "",
                "is_reply": is_reply,
                "parent_id": parent_id
            })
            return True

        # YouTube Data API imposes an indexing cap of ~2,525 comments when order='relevance'.
        # By querying 'relevance' first, we get the most engaging comments up to YouTube's relevance cutoff.
        # If the video has more comments and len(raw_comments) < target_limit, we seamlessly transition
        # to order='time' (chronological), deduplicating by ID, to fetch up to exactly target_limit (5,000).
        order_strategies = ["relevance", "time"]

        for order_strategy in order_strategies:
            if len(raw_comments) >= target_limit:
                break

            next_page_token = None
            while len(raw_comments) < target_limit:
                batch_size = min(target_limit - len(raw_comments), 100)
                request = youtube.commentThreads().list(
                    part=part_param,
                    videoId=video_id,
                    maxResults=batch_size,
                    order=order_strategy,
                    textFormat="plainText",
                    pageToken=next_page_token
                )
                response = request.execute()

                items = response.get("items", [])
                if not items:
                    break

                for item in items:
                    top_comment = item.get("snippet", {}).get("topLevelComment", {})
                    snippet = top_comment.get("snippet", {})
                    raw_text = snippet.get("textDisplay", "")
                    _add_comment(
                        top_comment.get("id", f"c_{len(raw_comments)}"),
                        snippet.get("authorDisplayName", "Anonymous"),
                        snippet.get("authorProfileImageUrl", ""),
                        raw_text,
                        snippet.get("likeCount", 0),
                        snippet.get("publishedAt", ""),
                        False
                    )
                    if len(raw_comments) >= target_limit:
                        break

                    if include_replies:
                        replies = item.get("replies", {}).get("comments", [])
                        for rep in replies:
                            rep_snippet = rep.get("snippet", {})
                            _add_comment(
                                rep.get("id", f"r_{len(raw_comments)}"),
                                rep_snippet.get("authorDisplayName", "Anonymous"),
                                rep_snippet.get("authorProfileImageUrl", ""),
                                rep_snippet.get("textDisplay", ""),
                                rep_snippet.get("likeCount", 0),
                                rep_snippet.get("publishedAt", ""),
                                True,
                                item.get("id")
                            )
                            if len(raw_comments) >= target_limit:
                                break

                next_page_token = response.get("nextPageToken")
                if not next_page_token or len(raw_comments) >= target_limit:
                    break

        if not raw_comments:
            return [], video_meta, None

        return raw_comments[:target_limit], video_meta, None

    except HttpError as e:
        status_code = e.resp.status
        reason = str(e)
        if "commentsDisabled" in reason:
            return [], video_meta, "Comments are disabled for this YouTube video."
        elif "videoNotFound" in reason:
            return [], {}, "The requested YouTube video was not found or is private."
        elif "quotaExceeded" in reason:
            return [], {}, "YouTube Data API quota exceeded. Please check your Google Cloud quota limits."
        elif status_code == 400 or "keyInvalid" in reason:
            return [], {}, "Invalid YouTube Data API key provided. Please verify your API key."
        else:
            return [], video_meta, f"YouTube API error (HTTP {status_code}): {e.reason or 'Unable to fetch comments'}"
    except Exception as e:
        return [], {}, f"Unexpected error connecting to YouTube API: {str(e)}"


def analyze_youtube_comments(
    comments: List[Dict[str, Any]],
    artifacts: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process fetched YouTube comments through the trained multi-label model
    combined with full sentence contextual semantic evaluation.
    Computes exact percentages and category distributions.
    """
    total = len(comments)
    if total == 0:
        return {
            "total_analyzed": 0,
            "toxic_count": 0,
            "non_toxic_count": 0,
            "toxic_percentage": 0.0,
            "non_toxic_percentage": 0.0,
            "category_counts": {l: 0 for l in DEFAULT_LABELS},
            "analyzed_comments": []
        }

    vectorizer = artifacts["vectorizer"]
    classifiers = artifacts["classifiers"]
    thresholds = artifacts["thresholds"]
    labels = artifacts.get("labels", DEFAULT_LABELS)
    context_analyzer = get_context_analyzer()

    # Load Kannada model artifacts if available
    kn_artifacts = artifacts.get("kannada")
    if kn_artifacts is None:
        try:
            kn_artifacts = load_kannada_artifacts()
        except Exception:
            kn_artifacts = None

    kn_vectorizer = kn_artifacts["vectorizer"] if kn_artifacts else None
    kn_classifiers = kn_artifacts["classifiers"] if kn_artifacts else None
    kn_thresholds = kn_artifacts.get("thresholds", {l: 0.35 for l in labels}) if kn_artifacts else thresholds
    kn_context_analyzer = get_kannada_context_analyzer()

    # 1. Classify comments by language
    kannada_indices = []
    english_indices = []
    lang_info_list = []

    for idx, c in enumerate(comments):
        info = detect_language(c.get("text", ""))
        lang_info_list.append(info)
        if info["is_kannada"] and kn_artifacts is not None:
            kannada_indices.append(idx)
        else:
            english_indices.append(idx)

    # 2. Batch vectorization and ML prediction
    raw_label_probs = {l: [0.0] * total for l in labels}

    # Batch process English comments
    if english_indices:
        en_texts = [mark_negation_and_context(comments[i]["text"]) for i in english_indices]
        en_texts = [t if t.strip() else (clean_text(comments[i]["text"]) or "comment") for t, i in zip(en_texts, english_indices)]
        en_vec = vectorizer.transform(en_texts)
        for l in labels:
            probs = classifiers[l].predict_proba(en_vec)[:, 1]
            for pos, orig_idx in enumerate(english_indices):
                raw_label_probs[l][orig_idx] = float(probs[pos])

    # Batch process Kannada comments
    if kannada_indices and kn_artifacts is not None:
        kn_texts = [clean_text(comments[i]["text"]) or "comment" for i in kannada_indices]
        kn_vec = kn_vectorizer.transform(kn_texts)
        for l in labels:
            probs = kn_classifiers[l].predict_proba(kn_vec)[:, 1]
            for pos, orig_idx in enumerate(kannada_indices):
                raw_label_probs[l][orig_idx] = float(probs[pos])

    analyzed_list = []
    toxic_count = 0
    category_counts = {l: 0 for l in labels}
    language_counts = {"English": 0, "Kannada": 0, "Mixed": 0}

    for idx, c in enumerate(comments):
        raw_scores = {l: float(raw_label_probs[l][idx]) for l in labels}
        lang_info = lang_info_list[idx]

        # Route to appropriate language context analyzer
        if lang_info["is_kannada"] and kn_artifacts is not None:
            context_eval = kn_context_analyzer.evaluate(
                raw_text=c.get("text", ""),
                ml_probs=raw_scores,
                ml_thresholds=kn_thresholds
            )
            c_flags = context_eval["final_flags"]
            c_scores = {l: round(context_eval["adjusted_probs"][l], 4) for l in labels}
            is_toxic = context_eval["final_is_toxic"]
            if lang_info["language"] == "mixed":
                language_counts["Mixed"] += 1
            else:
                language_counts["Kannada"] += 1
        else:
            context_eval = context_analyzer.evaluate(
                raw_text=c.get("text", ""),
                ml_probs=raw_scores,
                ml_thresholds=thresholds
            )
            c_flags = context_eval["final_flags"]
            c_scores = {l: round(context_eval["adjusted_probs"][l], 4) for l in labels}
            is_toxic = context_eval["final_is_toxic"]
            language_counts["English"] += 1

        if is_toxic:
            toxic_count += 1
            for flagged_label in c_flags:
                category_counts[flagged_label] += 1

        # Determine severity rating
        if not is_toxic:
            severity = "Non-Toxic"
        elif "severe_toxic" in c_flags or "threat" in c_flags or "identity_hate" in c_flags or len(c_flags) >= 3:
            severity = "High Severity"
        elif len(c_flags) >= 2 or c_scores.get("obscene", 0) > 0.70 or c_scores.get("insult", 0) > 0.70:
            severity = "Moderate Toxicity"
        else:
            severity = "Mild Toxicity"

        analyzed_list.append({
            "comment_id": c.get("comment_id", f"c_{idx}"),
            "author": c.get("author", "User"),
            "author_avatar": c.get("author_avatar", ""),
            "text": c.get("text", ""),
            "like_count": c.get("like_count", 0),
            "published_at": c.get("published_at", ""),
            "is_toxic": is_toxic,
            "classification": "Toxic" if is_toxic else "Non-Toxic",
            "severity": severity,
            "flagged_categories": c_flags,
            "category_scores": c_scores,
            "context_summary": context_eval.get("context_summary", ""),
            "is_reply": c.get("is_reply", False),
            "parent_id": c.get("parent_id"),
            "language": lang_info["display_name"],
            "language_code": lang_info["language"]
        })

    non_toxic_count = total - toxic_count
    toxic_percentage = round((toxic_count / total) * 100, 1)
    non_toxic_percentage = round((non_toxic_count / total) * 100, 1)

    return {
        "total_analyzed": total,
        "toxic_count": toxic_count,
        "non_toxic_count": non_toxic_count,
        "toxic_percentage": toxic_percentage,
        "non_toxic_percentage": non_toxic_percentage,
        "category_counts": category_counts,
        "language_counts": language_counts,
        "analyzed_comments": analyzed_list
    }
