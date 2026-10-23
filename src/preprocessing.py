"""
src/preprocessing.py

Text cleaning, preprocessing utilities, and schema validation for
multi-label toxic comment classification.
"""

import re
import html
import unicodedata
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

DEFAULT_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def clean_text(text: Any) -> str:
    """
    Standardize and clean input comment text:
      - Decode HTML entities
      - Normalize unicode characters (NFKD)
      - Lowercase
      - Strip URLs and email addresses
      - Remove IP addresses (frequent in Wikipedia revision comments)
      - Strip non-alphabet characters while retaining standard contractions
      - Collapse multiple whitespace and trim
    """
    if text is None or pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities like &quot;, &amp;, etc.
    text = html.unescape(text)

    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove IP addresses
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", " ", text)

    # Remove usernames often in Wikipedia talk pages (e.g., [[User:Foo]])
    text = re.sub(r"\[\[user:[^\]]+\]\]", " ", text)

    # Keep only letters, digits, apostrophes and spaces
    text = re.sub(r"[^a-z0-9'\s]", " ", text)

    # Normalize contractions and apostrophes
    text = re.sub(r"'+", "'", text)

    # Collapse consecutive whitespace characters
    text = re.sub(r"\s+", " ", text).strip()

    return text


def contextual_clean_text(text: Any) -> str:
    """
    Context-aware text normalization preserving sentence meaning,
    negation scoping, intensifier-praise idioms, and identity context.
    """
    from src.context_analyzer import mark_negation_and_context
    return mark_negation_and_context(text)


def compute_label_distribution(df: pd.DataFrame, labels: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Calculate counts, positive rates, and co-occurrence for multi-label targets.
    """
    if labels is None:
        labels = DEFAULT_LABELS

    total = len(df)
    stats = {}
    for col in labels:
        if col in df.columns:
            pos_count = int(df[col].sum())
            pct = (pos_count / total * 100) if total > 0 else 0.0
            stats[col] = {
                "count": pos_count,
                "percentage": round(pct, 2),
                "total": total
            }

    # Also calculate any_toxic count (at least one label is 1)
    available_labels = [c for c in labels if c in df.columns]
    if available_labels:
        any_toxic_count = int((df[available_labels].sum(axis=1) > 0).sum())
        stats["any_toxic"] = {
            "count": any_toxic_count,
            "percentage": round((any_toxic_count / total * 100) if total > 0 else 0.0, 2),
            "total": total
        }
    return stats


def validate_input_comment(text: Any, max_chars: int = 5000) -> Tuple[bool, str]:
    """
    Validate incoming text from API or user input.
    Returns (is_valid, error_message).
    """
    if text is None:
        return False, "Comment text must not be empty or null."

    if not isinstance(text, str):
        return False, "Comment must be a text string."

    stripped = text.strip()
    if len(stripped) == 0:
        return False, "Comment cannot be empty or contain only whitespace."

    if len(stripped) > max_chars:
        return False, f"Comment exceeds maximum length of {max_chars} characters (received {len(stripped)})."

    return True, ""
