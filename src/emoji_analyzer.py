"""
src/emoji_analyzer.py

Comprehensive Emoji Toxicity, Sentiment, Combination, and Sarcasm Analyzer.
Preserves and analyzes emoji meaning, combinations, repetition, and semantic context
across English, Kannada (ಕನ್ನಡ Script), Romanized Kannada (Kanglish), and Code-Mixed text.

Provides:
  - Granular detection and parsing of Unicode emojis (single, multi-byte, variation selectors).
  - Multi-label toxicity contributions across all 6 standard labels:
      ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'].
  - Sarcasm and irony detection (derisive emojis paired with superficial praise words).
  - Multiplier dynamics for emoji repetition and compound threat/hate combinations.
  - Contextual resolution for Kannada cultural tokens (e.g., 'ಬೆಂಕಿ' + 🔥, 'ನಮಸ್ಕಾರ' + 🙏).
  - Model comparison utilities evaluating text-only vs. text-plus-emoji predictions.
"""

import re
import html
import unicodedata
from typing import Dict, Any, List, Optional, Tuple, Set
import emoji

# Default multi-label toxicity categories
DEFAULT_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# -----------------------------------------------------------------------------
# 1. Comprehensive Emoji Lexicon with Multi-Label Probabilities & Semantic Roles
# -----------------------------------------------------------------------------

EMOJI_LEXICON = {
    # --- Obscene Gestures & Explicit Profanity ---
    "🖕": {
        "name": "middle finger",
        "category": "obscene_gesture",
        "sentiment": "hostile",
        "base_probs": {"toxic": 0.94, "obscene": 0.92, "insult": 0.88, "severe_toxic": 0.45, "threat": 0.10, "identity_hate": 0.05},
        "primary_label": "obscene",
        "contribution_label": "+92% Obscene / +94% Toxic",
        "explanation": "Offensive middle-finger gesture denoting explicit vulgarity and hostility"
    },
    "🤬": {
        "name": "face with symbols on mouth",
        "category": "anger_aggression",
        "sentiment": "profane_rage",
        "base_probs": {"toxic": 0.88, "obscene": 0.82, "insult": 0.76, "severe_toxic": 0.35, "threat": 0.12, "identity_hate": 0.02},
        "primary_label": "obscene",
        "contribution_label": "+82% Obscene / Profanity Outburst",
        "explanation": "Swearing face representing explicit profanity, vulgar outburst, and aggressive rage"
    },

    # --- Hostile Insult, Disparagement & Mockery ---
    "🤡": {
        "name": "clown face",
        "category": "hostile_insult",
        "sentiment": "mocking_insult",
        "base_probs": {"toxic": 0.82, "insult": 0.85, "obscene": 0.20, "severe_toxic": 0.08, "threat": 0.02, "identity_hate": 0.02},
        "primary_label": "insult",
        "contribution_label": "+85% Insult / Derisive Mockery",
        "explanation": "Clown emoji used to demean, ridicule, or brand someone a fool"
    },
    "💩": {
        "name": "pile of poo",
        "category": "hostile_insult",
        "sentiment": "disparaging",
        "base_probs": {"toxic": 0.74, "insult": 0.78, "obscene": 0.45, "severe_toxic": 0.05, "threat": 0.02, "identity_hate": 0.02},
        "primary_label": "insult",
        "contribution_label": "+78% Insult / Disparaging",
        "explanation": "Feces emoji used to belittle work, code, or personal character as worthless filth"
    },
    "🤮": {
        "name": "face vomiting",
        "category": "hostile_insult",
        "sentiment": "disgust_revulsion",
        "base_probs": {"toxic": 0.72, "insult": 0.74, "obscene": 0.38, "severe_toxic": 0.05, "threat": 0.02, "identity_hate": 0.02},
        "primary_label": "insult",
        "contribution_label": "+74% Insult / Revulsion",
        "explanation": "Vomiting emoji expressing visceral disgust or targeted repulsion"
    },
    "🤢": {
        "name": "nauseated face",
        "category": "hostile_insult",
        "sentiment": "disgust",
        "base_probs": {"toxic": 0.66, "insult": 0.68, "obscene": 0.25, "severe_toxic": 0.02, "threat": 0.02, "identity_hate": 0.02},
        "primary_label": "insult",
        "contribution_label": "+68% Insult / Contempt",
        "explanation": "Nauseated face expressing contempt and disparaging repulsion"
    },
    "🗑️": {
        "name": "wastebasket / trash",
        "category": "hostile_insult",
        "sentiment": "disparaging",
        "base_probs": {"toxic": 0.68, "insult": 0.72, "obscene": 0.20, "severe_toxic": 0.05, "threat": 0.02, "identity_hate": 0.02},
        "primary_label": "insult",
        "contribution_label": "+72% Insult / Trash Label",
        "explanation": "Trash can emoji labeling content, author, or creator as garbage"
    },
    "🐷": {
        "name": "pig face",
        "category": "hostile_insult",
        "sentiment": "dehumanizing",
        "base_probs": {"toxic": 0.75, "insult": 0.80, "obscene": 0.30, "severe_toxic": 0.10, "threat": 0.05, "identity_hate": 0.05},
        "primary_label": "insult",
        "contribution_label": "+80% Insult / Demeaning",
        "explanation": "Pig emoji deployed as a derogatory animal slur or demeaning body insult"
    },
    "🐖": {
        "name": "pig",
        "category": "hostile_insult",
        "sentiment": "dehumanizing",
        "base_probs": {"toxic": 0.75, "insult": 0.80, "obscene": 0.30, "severe_toxic": 0.10, "threat": 0.05, "identity_hate": 0.05},
        "primary_label": "insult",
        "contribution_label": "+80% Insult / Demeaning",
        "explanation": "Pig animal slur used to demean or insult a person"
    },
    "🐒": {
        "name": "monkey",
        "category": "hostile_insult",
        "sentiment": "derogatory",
        "base_probs": {"toxic": 0.78, "insult": 0.82, "identity_hate": 0.68, "obscene": 0.25, "severe_toxic": 0.15, "threat": 0.05},
        "primary_label": "insult",
        "contribution_label": "+82% Insult / Identity Slur Risk",
        "explanation": "Monkey emoji frequently utilized in online abuse as an identity or dehumanizing slur"
    },
    "🐵": {
        "name": "monkey face",
        "category": "hostile_insult",
        "sentiment": "derogatory",
        "base_probs": {"toxic": 0.76, "insult": 0.80, "identity_hate": 0.65, "obscene": 0.25, "severe_toxic": 0.15, "threat": 0.05},
        "primary_label": "insult",
        "contribution_label": "+80% Insult / Identity Slur Risk",
        "explanation": "Monkey face used in hostile contexts as a racial or personal insult"
    },
    "🤫": {
        "name": "shushing face",
        "category": "hostile_insult",
        "sentiment": "silencing",
        "base_probs": {"toxic": 0.58, "insult": 0.55, "obscene": 0.15, "severe_toxic": 0.02, "threat": 0.05, "identity_hate": 0.02},
        "primary_label": "insult",
        "contribution_label": "+55% Insult / Aggressive Silencing",
        "explanation": "Shushing emoji aggressively ordering the target to shut up"
    },
    "👎": {
        "name": "thumbs down",
        "category": "disapproval",
        "sentiment": "negative",
        "base_probs": {"toxic": 0.25, "insult": 0.20, "obscene": 0.05, "severe_toxic": 0.02, "threat": 0.01, "identity_hate": 0.01},
        "primary_label": "toxic",
        "contribution_label": "Mild Disapproval",
        "explanation": "Thumbs down expressing basic negative disapproval"
    },

    # --- Violent Threats & Physical Harm ---
    "🔪": {
        "name": "kitchen knife",
        "category": "violent_threat",
        "sentiment": "lethal_threat",
        "base_probs": {"toxic": 0.92, "threat": 0.90, "severe_toxic": 0.72, "insult": 0.25, "obscene": 0.15, "identity_hate": 0.05},
        "primary_label": "threat",
        "contribution_label": "+90% Violent Threat / Weapon",
        "explanation": "Knife weapon emoji conveying explicit threat of physical assault or stabbings"
    },
    "🗡️": {
        "name": "dagger",
        "category": "violent_threat",
        "sentiment": "lethal_threat",
        "base_probs": {"toxic": 0.88, "threat": 0.85, "severe_toxic": 0.68, "insult": 0.20, "obscene": 0.10, "identity_hate": 0.05},
        "primary_label": "threat",
        "contribution_label": "+85% Violent Threat / Weapon",
        "explanation": "Dagger weapon emoji indicating physical threat or violence"
    },
    "🪓": {
        "name": "axe",
        "category": "violent_threat",
        "sentiment": "lethal_threat",
        "base_probs": {"toxic": 0.88, "threat": 0.85, "severe_toxic": 0.68, "insult": 0.20, "obscene": 0.10, "identity_hate": 0.05},
        "primary_label": "threat",
        "contribution_label": "+85% Violent Threat / Weapon",
        "explanation": "Axe weapon emoji representing lethal force or physical assault"
    },
    "🔫": {
        "name": "water pistol / handgun",
        "category": "violent_threat",
        "sentiment": "firearm_threat",
        "base_probs": {"toxic": 0.84, "threat": 0.82, "severe_toxic": 0.62, "insult": 0.15, "obscene": 0.10, "identity_hate": 0.05},
        "primary_label": "threat",
        "contribution_label": "+82% Threat / Firearm Symbol",
        "explanation": "Gun weapon emoji conveying shooting violence or intimidation"
    },
    "💣": {
        "name": "bomb",
        "category": "violent_threat",
        "sentiment": "explosive_threat",
        "base_probs": {"toxic": 0.82, "threat": 0.78, "severe_toxic": 0.60, "insult": 0.10, "obscene": 0.10, "identity_hate": 0.05},
        "primary_label": "threat",
        "contribution_label": "+78% Threat / Explosive Violence",
        "explanation": "Bomb emoji conveying catastrophic attack or physical harm"
    },
    "⚰️": {
        "name": "coffin",
        "category": "violent_threat",
        "sentiment": "death_threat",
        "base_probs": {"toxic": 0.85, "threat": 0.82, "severe_toxic": 0.65, "insult": 0.30, "obscene": 0.10, "identity_hate": 0.05},
        "primary_label": "threat",
        "contribution_label": "+82% Threat / Death Imagery",
        "explanation": "Coffin emoji implying murder, fatality, or death wishes"
    },
    "☠️": {
        "name": "skull and crossbones",
        "category": "violent_threat",
        "sentiment": "lethal_symbol",
        "base_probs": {"toxic": 0.82, "threat": 0.76, "severe_toxic": 0.58, "insult": 0.25, "obscene": 0.10, "identity_hate": 0.05},
        "primary_label": "threat",
        "contribution_label": "+76% Threat / Lethality Symbol",
        "explanation": "Skull & crossbones denoting lethal toxicity, poison, or fatality"
    },
    "🩸": {
        "name": "drop of blood",
        "category": "violent_threat",
        "sentiment": "bloodshed",
        "base_probs": {"toxic": 0.78, "threat": 0.75, "severe_toxic": 0.58, "insult": 0.15, "obscene": 0.10, "identity_hate": 0.05},
        "primary_label": "threat",
        "contribution_label": "+75% Threat / Bloodshed Imagery",
        "explanation": "Blood drop emoji reinforcing physical harm, bloodshed, or injury"
    },

    # --- Anger, Aggression & Malevolence ---
    "😡": {
        "name": "pouting face / enraged",
        "category": "anger_aggression",
        "sentiment": "intense_anger",
        "base_probs": {"toxic": 0.68, "insult": 0.52, "obscene": 0.25, "severe_toxic": 0.12, "threat": 0.10, "identity_hate": 0.02},
        "primary_label": "toxic",
        "contribution_label": "+68% Toxic / Hostile Rage",
        "explanation": "Red enraged face indicating intense animosity and belligerent anger"
    },
    "😠": {
        "name": "angry face",
        "category": "anger_aggression",
        "sentiment": "anger",
        "base_probs": {"toxic": 0.62, "insult": 0.46, "obscene": 0.18, "severe_toxic": 0.08, "threat": 0.06, "identity_hate": 0.02},
        "primary_label": "toxic",
        "contribution_label": "+62% Toxic / Agitated Anger",
        "explanation": "Aggressive frown conveying overt frustration and hostility"
    },
    "👿": {
        "name": "angry devil",
        "category": "anger_aggression",
        "sentiment": "malevolent",
        "base_probs": {"toxic": 0.72, "insult": 0.58, "threat": 0.38, "obscene": 0.22, "severe_toxic": 0.18, "identity_hate": 0.02},
        "primary_label": "toxic",
        "contribution_label": "+72% Toxic / Malevolent Intent",
        "explanation": "Horned devil face signifying malicious, wicked, or aggressive intent"
    },
    "💢": {
        "name": "anger symbol",
        "category": "anger_aggression",
        "sentiment": "annoyance",
        "base_probs": {"toxic": 0.45, "insult": 0.30, "obscene": 0.10, "severe_toxic": 0.05, "threat": 0.05, "identity_hate": 0.01},
        "primary_label": "toxic",
        "contribution_label": "Aggressive Agitation",
        "explanation": "Comic anger vein denoting acute irritation and simmering aggression"
    },

    # --- Sarcasm, Irony, Contempt & Dismissal ---
    "🙃": {
        "name": "upside-down face",
        "category": "sarcastic_mockery",
        "sentiment": "irony_sarcasm",
        "base_probs": {"toxic": 0.40, "insult": 0.42, "obscene": 0.05, "severe_toxic": 0.02, "threat": 0.01, "identity_hate": 0.01},
        "primary_label": "insult",
        "contribution_label": "Sarcasm / Passive-Aggression Marker",
        "explanation": "Upside-down face signaling irony, passive-aggressive sarcasm, or mocked compliance"
    },
    "🙄": {
        "name": "face with rolling eyes",
        "category": "sarcastic_mockery",
        "sentiment": "dismissive_contempt",
        "base_probs": {"toxic": 0.45, "insult": 0.46, "obscene": 0.05, "severe_toxic": 0.02, "threat": 0.01, "identity_hate": 0.01},
        "primary_label": "insult",
        "contribution_label": "Contempt / Dismissive Eye-Roll",
        "explanation": "Eye-rolling face expressing open disdain, condescension, or dismissive contempt"
    },
    "🥱": {
        "name": "yawning face",
        "category": "sarcastic_mockery",
        "sentiment": "bored_dismissal",
        "base_probs": {"toxic": 0.38, "insult": 0.42, "obscene": 0.05, "severe_toxic": 0.02, "threat": 0.01, "identity_hate": 0.01},
        "primary_label": "insult",
        "contribution_label": "Disparaging Boredom / Dismissal",
        "explanation": "Yawning emoji deployed to demean an argument or speaker as tedious garbage"
    },
    "😏": {
        "name": "smirking face",
        "category": "sarcastic_mockery",
        "sentiment": "smugness",
        "base_probs": {"toxic": 0.30, "insult": 0.32, "obscene": 0.05, "severe_toxic": 0.01, "threat": 0.01, "identity_hate": 0.01},
        "primary_label": "insult",
        "contribution_label": "Smug Derision",
        "explanation": "Smirking face conveying superior arrogance or mocking amusement"
    },
    "😒": {
        "name": "unamused face",
        "category": "sarcastic_mockery",
        "sentiment": "displeasure",
        "base_probs": {"toxic": 0.32, "insult": 0.30, "obscene": 0.05, "severe_toxic": 0.01, "threat": 0.01, "identity_hate": 0.01},
        "primary_label": "toxic",
        "contribution_label": "Unamused Disdain",
        "explanation": "Unamused look registering sour dissatisfaction or disapproval"
    },
    "🎭": {
        "name": "performing arts / masks",
        "category": "sarcastic_mockery",
        "sentiment": "duplicity",
        "base_probs": {"toxic": 0.35, "insult": 0.38, "obscene": 0.05, "severe_toxic": 0.02, "threat": 0.01, "identity_hate": 0.01},
        "primary_label": "insult",
        "contribution_label": "Two-Faced / Hypocrisy Accusation",
        "explanation": "Masks used in debate to brand someone as a fake, actor, or two-faced fraud"
    },
    "😂": {
        "name": "face with tears of joy",
        "category": "sarcastic_mockery",
        "sentiment": "mocking_laughter",
        "base_probs": {"toxic": 0.35, "insult": 0.35, "obscene": 0.05, "severe_toxic": 0.01, "threat": 0.01, "identity_hate": 0.01},
        "primary_label": "insult",
        "contribution_label": "Mocking Laughter / Taunt",
        "explanation": "Laughing emoji used contextually to mock, ridicule, or deride"
    },
    "🤣": {
        "name": "rolling on the floor laughing",
        "category": "sarcastic_mockery",
        "sentiment": "mocking_laughter",
        "base_probs": {"toxic": 0.38, "insult": 0.38, "obscene": 0.05, "severe_toxic": 0.01, "threat": 0.01, "identity_hate": 0.01},
        "primary_label": "insult",
        "contribution_label": "Derisive Mockery / Laughter",
        "explanation": "Rolling laughing emoji deployed to taunt or demean"
    },
    "💀": {
        "name": "skull",
        "category": "sarcastic_mockery",
        "sentiment": "derisive_mockery",
        "base_probs": {"toxic": 0.35, "insult": 0.35, "threat": 0.15, "obscene": 0.05, "severe_toxic": 0.02, "identity_hate": 0.01},
        "primary_label": "insult",
        "contribution_label": "Mocking Contempt / Derision",
        "explanation": "Skull emoji used contextually for derision or humiliation"
    }
}

# Positive / Praise / Respectful Emojis that mitigate toxicity and signal constructive intent
POSITIVE_PRAISE_EMOJIS = {
    # Hearts
    "❤️", "💖", "💕", "💓", "💗", "💘", "💙", "💚", "💛", "💜", "🤍", "🤎", "🖤", "❣️", "💞",
    # Thumbs & gestures
    "👍", "👏", "🙌", "🤝", "👌", "✌️", "🤞", "👊",
    # Cultural & Gratitude
    "🙏", "🚩", "💐", "🌸", "🌺", "🌻", "🌹", "🌷", "🌼",
    # Enthusiasm & Achievement
    "🔥", "💯", "⭐", "🌟", "✨", "🏆", "🥇", "🥈", "🥉", "👑", "🎯", "🎉", "🎊", "🥂", "🚀",
    # Smiles & Love
    "😊", "😀", "😃", "😄", "😁", "🥰", "😍", "🤩", "😇", " warm", "🤗", "😎", "🥳"
}

# Cultural Kannada positive combinations
KANNADA_POSITIVE_EMOJI_WORDS = {
    "ಬೆಂಕಿ": "🔥",
    "benki": "🔥",
    "ಧನ್ಯವಾದಗಳು": "🙏",
    "dhanyavada": "🙏",
    "dhanyavadagalu": "🙏",
    "ನಮಸ್ಕಾರ": "🙏",
    "namaskara": "🙏",
    "ಸೂಪರ್": "👍",
    "super": "👍",
    "ರಾಜ": "👑",
    "boss": "👑"
}

# Superficial praise words used in sarcastic insults
SUPERFICIAL_PRAISE_WORDS = {
    "genius", "brilliant", "smart", "expert", "masterpiece", "great job", "amazing",
    "well done", "impressive", "such talent", "love it", "wonderful", "legend",
    "hero", "bravo", "intellectual", "scholar", "gifted", "nice job", "good job",
    "fantastic", "award winning", "best ever", "outstanding", "pro"
}


# -----------------------------------------------------------------------------
# 2. Compound Multiplier Pairs
# -----------------------------------------------------------------------------

COMPOUND_EMOJI_MULTIPLIERS = {
    # Violent Threat Compounds
    ("🔪", "🩸"): {
        "elevate_labels": {"threat": 0.95, "toxic": 0.96, "severe_toxic": 0.88},
        "reason": "Compound bladed weapon + bloodshed sequence indicates acute violent threat"
    },
    ("🩸", "🔪"): {
        "elevate_labels": {"threat": 0.95, "toxic": 0.96, "severe_toxic": 0.88},
        "reason": "Compound bloodshed + bladed weapon sequence indicates acute violent threat"
    },
    ("💀", "⚰️"): {
        "elevate_labels": {"threat": 0.92, "toxic": 0.94, "severe_toxic": 0.82},
        "reason": "Compound skull + coffin sequence signifies lethal mortality threat"
    },
    ("⚰️", "💀"): {
        "elevate_labels": {"threat": 0.92, "toxic": 0.94, "severe_toxic": 0.82},
        "reason": "Compound coffin + skull sequence signifies lethal mortality threat"
    },

    # Obscene & Insult Compounds
    ("🖕", "🤡"): {
        "elevate_labels": {"obscene": 0.95, "toxic": 0.96, "insult": 0.92},
        "reason": "Compound middle finger + clown face combines obscene profanity with demeaning mockery"
    },
    ("🤡", "🖕"): {
        "elevate_labels": {"obscene": 0.95, "toxic": 0.96, "insult": 0.92},
        "reason": "Compound clown face + middle finger combines demeaning mockery with obscene profanity"
    },
    ("🤡", "💩"): {
        "elevate_labels": {"insult": 0.90, "toxic": 0.90, "obscene": 0.45},
        "reason": "Compound clown + feces sequence represents intensive disparaging insult"
    },
    ("💩", "🤡"): {
        "elevate_labels": {"insult": 0.90, "toxic": 0.90, "obscene": 0.45},
        "reason": "Compound feces + clown sequence represents intensive disparaging insult"
    },
    ("🖕", "🔥"): {
        "elevate_labels": {"toxic": 0.96, "obscene": 0.94, "insult": 0.88},
        "reason": "Middle finger intensified by fire signifies furious hostility"
    },
    ("🐵", "🍌"): {
        "elevate_labels": {"identity_hate": 0.92, "toxic": 0.94, "insult": 0.90},
        "reason": "Monkey and banana combination deployed as explicit racial/identity hate slur"
    },
    ("🐒", "🍌"): {
        "elevate_labels": {"identity_hate": 0.92, "toxic": 0.94, "insult": 0.90},
        "reason": "Monkey and banana combination deployed as explicit racial/identity hate slur"
    }
}


# -----------------------------------------------------------------------------
# 3. Core Emoji Extraction & Parsing
# -----------------------------------------------------------------------------

def extract_emojis(text: Any) -> List[Dict[str, Any]]:
    """
    Extract all emojis from text with character, starting index, ending index,
    and standard Unicode description.
    """
    if not text:
        return []
    text_str = str(text)
    raw_list = emoji.emoji_list(text_str)
    results = []
    for item in raw_list:
        em = item["emoji"]
        # Determine human-friendly name
        em_name = emoji.demojize(em).strip(":").replace("_", " ")
        results.append({
            "emoji": em,
            "start": item["match_start"],
            "end": item["match_end"],
            "name": em_name
        })
    return results


def has_emojis(text: Any) -> bool:
    """Return True if text contains at least one Unicode emoji."""
    if not text:
        return False
    return len(emoji.emoji_list(str(text))) > 0


def is_emoji_only(text: Any) -> bool:
    """
    Check if a comment consists exclusively of emojis, whitespace, and punctuation,
    with zero alphabetical or numerical words.
    """
    if not text:
        return False
    text_str = str(text).strip()
    emojis = emoji.emoji_list(text_str)
    if not emojis:
        return False
    # Strip emojis out of string and check if any alphanumerics remain
    remaining = emoji.replace_emoji(text_str, replace="").strip()
    # Also strip punctuation
    alphanumeric_chars = re.findall(r"[a-zA-Z0-9\u0C80-\u0CFF]", remaining)
    return len(alphanumeric_chars) == 0


# -----------------------------------------------------------------------------
# 4. Contextual Emoji Analyzer Engine
# -----------------------------------------------------------------------------

class EmojiContextAnalyzer:
    """
    Semantic Sentence & Emoji Context Evaluator.
    Analyzes emoji meaning, compound combinations, repetition, sarcasm,
    and contextual interaction with text across English, Kannada, and Kanglish.
    """

    def __init__(self):
        pass

    def analyze(
        self,
        raw_text: str,
        language_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze comment text for emoji presence, semantic intent, sarcasm,
        and toxicity contribution across the 6 labels.

        Returns:
          {
            "has_emojis": bool,
            "is_emoji_only": bool,
            "emojis_detected": List[Dict],
            "emoji_count": int,
            "unique_emoji_count": int,
            "sarcasm_detected": bool,
            "is_mitigating": bool,
            "emoji_toxicity_score": float,
            "primary_sentiment": str,
            "label_deltas": Dict[str, float],
            "label_overrides": Dict[str, float],
            "context_reasons": List[str],
            "ui_chips": List[Dict[str, str]]
          }
        """
        if not raw_text:
            return self._empty_result()

        raw_str = str(raw_text)
        detected_raw = extract_emojis(raw_str)
        if not detected_raw:
            return self._empty_result()

        emoji_chars = [item["emoji"] for item in detected_raw]
        unique_emojis = list(dict.fromkeys(emoji_chars))
        emoji_only = is_emoji_only(raw_str)

        text_lower = raw_str.lower().strip()
        text_without_emojis = emoji.replace_emoji(text_lower, replace=" ").strip()
        words_in_text = re.findall(r"[a-z0-9\u0C80-\u0CFF']+", text_without_emojis)

        # 1. Inspect detected emojis against lexicon
        detected_details = []
        hostile_emojis = []
        threat_emojis = []
        obscene_emojis = []
        sarcastic_emojis = []
        positive_emojis = []
        neutral_emojis = []

        label_deltas = {l: 0.0 for l in DEFAULT_LABELS}
        label_overrides = {}
        context_reasons = []
        ui_chips = []

        # Count occurrences of each emoji
        emoji_counts = {}
        for em in emoji_chars:
            emoji_counts[em] = emoji_counts.get(em, 0) + 1

        for em in unique_emojis:
            count = emoji_counts[em]
            lex = EMOJI_LEXICON.get(em)

            if lex:
                cat = lex["category"]
                sent = lex["sentiment"]
                prim = lex["primary_label"]
                base_probs = lex["base_probs"]
                name = lex["name"]

                repetition_mult = 1.0 + min((count - 1) * 0.08, 0.35) if count > 1 else 1.0

                if cat == "obscene_gesture":
                    obscene_emojis.append(em)
                    hostile_emojis.append(em)
                elif cat == "violent_threat":
                    threat_emojis.append(em)
                    hostile_emojis.append(em)
                elif cat in ("hostile_insult", "anger_aggression"):
                    hostile_emojis.append(em)
                elif cat == "sarcastic_mockery":
                    sarcastic_emojis.append(em)

                chip_text = f"{em} {lex['contribution_label']}"
                if count > 1:
                    chip_text += f" (x{count})"

                ui_chips.append({
                    "emoji": em,
                    "name": name,
                    "category": cat,
                    "chip_text": chip_text,
                    "is_toxic": True
                })

                detected_details.append({
                    "emoji": em,
                    "name": name,
                    "count": count,
                    "category": cat,
                    "sentiment": sent,
                    "primary_label": prim,
                    "contribution": lex["contribution_label"],
                    "explanation": lex["explanation"]
                })

                # Accumulate deltas
                for lbl, prob in base_probs.items():
                    effective = min(1.0, round(prob * repetition_mult, 4))
                    label_deltas[lbl] = max(label_deltas[lbl], effective)

            elif em in POSITIVE_PRAISE_EMOJIS:
                positive_emojis.append(em)
                name = emoji.demojize(em).strip(":").replace("_", " ")
                ui_chips.append({
                    "emoji": em,
                    "name": name,
                    "category": "positive_praise",
                    "chip_text": f"{em} Praise / Friendly",
                    "is_toxic": False
                })
                detected_details.append({
                    "emoji": em,
                    "name": name,
                    "count": count,
                    "category": "positive_praise",
                    "sentiment": "positive",
                    "primary_label": "clean",
                    "contribution": "Constructive / Mitigating",
                    "explanation": "Friendly or appreciative emoji expressing positive goodwill"
                })
            else:
                neutral_emojis.append(em)
                name = emoji.demojize(em).strip(":").replace("_", " ")
                ui_chips.append({
                    "emoji": em,
                    "name": name,
                    "category": "neutral",
                    "chip_text": f"{em} Neutral",
                    "is_toxic": False
                })
                detected_details.append({
                    "emoji": em,
                    "name": name,
                    "count": count,
                    "category": "neutral",
                    "sentiment": "neutral",
                    "primary_label": "clean",
                    "contribution": "Neutral",
                    "explanation": "Informational or ambient emoji without toxic intent"
                })

        # 2. Check for Compound Multiplier Pairs
        for i in range(len(emoji_chars) - 1):
            pair = (emoji_chars[i], emoji_chars[i + 1])
            if pair in COMPOUND_EMOJI_MULTIPLIERS:
                comp = COMPOUND_EMOJI_MULTIPLIERS[pair]
                context_reasons.append(comp["reason"])
                for lbl, val in comp["elevate_labels"].items():
                    label_deltas[lbl] = max(label_deltas[lbl], val)

        # 3. Kannada & Kanglish Cultural Resolution
        is_kannada = False
        if language_info:
            is_kannada = language_info.get("is_kannada", False)
        else:
            is_kannada = bool(re.search(r"[\u0C80-\u0CFF]", raw_str))

        # "ಬೆಂಕಿ" / "benki" + 🔥 is supreme colloquial praise in Kannada
        has_kannada_fire_praise = bool(re.search(r"(?:^|[^\w\u0C80-\u0CFF])(ಬೆಂಕಿ|benki)(?:[^\w\u0C80-\u0CFF]|$)", text_lower)) and ("🔥" in emoji_chars)
        has_kannada_respect_namaskara = bool(re.search(r"(?:^|[^\w\u0C80-\u0CFF])(ಧನ್ಯವಾದ|ಧನ್ಯವಾದಗಳು|ನಮಸ್ಕಾರ|dhanyavada|namaskara)(?:[^\w\u0C80-\u0CFF]|$)", text_lower)) and ("🙏" in emoji_chars)

        if has_kannada_fire_praise:
            context_reasons.append("Kannada colloquial praise idiom ('ಬೆಂಕಿ' + 🔥) recognized as extreme enthusiasm, not threat")
            # Clear false threat/arson interpretations
            label_deltas["threat"] = 0.0
            label_deltas["toxic"] = min(label_deltas["toxic"], 0.10)

        if has_kannada_respect_namaskara:
            context_reasons.append("Kannada cultural respect marker paired with folded hands ('🙏') reinforces polite constructive tone")
            for l in DEFAULT_LABELS:
                label_deltas[l] = min(label_deltas[l], 0.10)

        # 4. Sarcasm & Mockery Detection
        # Check if superficial praise words appear alongside derisive mocking emojis
        sarcasm_detected = False
        found_praise_words = [w for w in SUPERFICIAL_PRAISE_WORDS if re.search(rf"\b{re.escape(w)}\b", text_lower)]
        has_mocking_emoji = any(em in emoji_chars for em in ["🤡", "🙃", "🗑️", "🤮", "💩", "🙄", "🥱"])

        # Sarcastic praise inversion: e.g. "What a genius 🤡", "Brilliant work 🙃🗑️"
        if found_praise_words and has_mocking_emoji:
            sarcasm_detected = True
            mocking_char = next(em for em in ["🤡", "🙃", "🗑️", "🤮", "💩", "🙄", "🥱"] if em in emoji_chars)
            context_reasons.append(
                f"Sarcastic mockery detected: Apparent praise ('{found_praise_words[0]}') contradicted by derisive emoji '{mocking_char}'"
            )
            label_deltas["toxic"] = max(label_deltas["toxic"], 0.84)
            label_deltas["insult"] = max(label_deltas["insult"], 0.82)

        # Sarcastic slow-clap / negative outcome mockery: e.g. "Great job failing the build again 👏😂"
        has_negative_outcome = bool(re.search(r"\b(fail|failing|failed|broke|broken|ruin|ruined|ruining|mess|disaster|loser|lose|terrible|worst)\b", text_lower))
        if found_praise_words and has_negative_outcome and any(em in emoji_chars for em in ["👏", "😂", "🤣", "🙃", "🤡", "🙄"]):
            sarcasm_detected = True
            context_reasons.append(
                "Sarcastic slow-clap / mockery detected: Praise phrasing applied to a failure or negative outcome"
            )
            label_deltas["toxic"] = max(label_deltas["toxic"], 0.82)
            label_deltas["insult"] = max(label_deltas["insult"], 0.80)

        # Standalone upside-down face or eye-roll with brief text
        if ("🙃" in emoji_chars or "🙄" in emoji_chars) and len(words_in_text) <= 6 and not sarcasm_detected:
            if any(w in text_lower for w in ["sure", "okay", "yeah", "right", "great", "thanks", "wow"]):
                sarcasm_detected = True
                context_reasons.append("Passive-aggressive sarcasm indicated by upside-down / eye-roll emoji with affirmative token")
                label_deltas["toxic"] = max(label_deltas["toxic"], 0.72)
                label_deltas["insult"] = max(label_deltas["insult"], 0.70)

        # Derisive mockery / taunting paired with insult or attack (e.g. "You are useless 😂", "What an idiot 🤣", "You have no brain 💀")
        has_insult_term = bool(re.search(
            r"\b(useless|worthless|idiot|moron|stupid|dumb|loser|trash|garbage|pathetic|clown|failure|incompetent|ugly|scum|disgusting|joke|terrible|worst|shut\s+up|hate\s+you|no\s+(brain|brains|sense|talent|life|clue)|suck|sucks|stink|bitch|bastard|asshole|jerk|pig|dumbass|dadda|huccha|katte)\b",
            text_lower
        ))
        has_mocking_laughter = any(em in emoji_chars for em in ["😂", "🤣", "😆", "😹", "💀"])

        if has_insult_term and has_mocking_laughter:
            sarcasm_detected = True
            mock_char = next(em for em in ["😂", "🤣", "😆", "😹", "💀"] if em in emoji_chars)
            context_reasons.append(
                f"Derisive mockery / taunting detected: Targeted insult combined with ridiculing emoji '{mock_char}'"
            )
            label_deltas["toxic"] = max(label_deltas["toxic"], 0.88)
            label_deltas["insult"] = max(label_deltas["insult"], 0.86)
            for chip in ui_chips:
                if chip["emoji"] == mock_char:
                    chip["category"] = "sarcastic_mockery"
                    chip["chip_text"] = f"{mock_char} Mocking / Insult"
                    chip["is_toxic"] = True

        # 5. Emoji-Only Evaluation
        if emoji_only:
            if hostile_emojis or threat_emojis or obscene_emojis:
                context_reasons.append("Comment consists solely of hostile or threatening emojis")
                # Direct override of model output for emoji-only toxic posts
                for lbl in DEFAULT_LABELS:
                    label_overrides[lbl] = min(1.0, round(label_deltas[lbl], 4))
            elif positive_emojis and not hostile_emojis:
                context_reasons.append("Comment consists solely of positive praise or friendly emojis")
                for lbl in DEFAULT_LABELS:
                    label_overrides[lbl] = 0.02
            else:
                context_reasons.append("Comment consists solely of neutral or expressive emojis")
                for lbl in DEFAULT_LABELS:
                    label_overrides[lbl] = 0.05

        # 6. Mitigating vs Amplifying Context
        is_mitigating = False
        if len(positive_emojis) > 0 and len(hostile_emojis) == 0 and not sarcasm_detected:
            is_mitigating = True
            # Friendly emojis in otherwise ambiguous sentence mitigate borderline triggers
            # (e.g., 'Stop it haha 😂❤️')

        # Overall emoji toxicity score
        max_toxic_signal = max(label_deltas.values()) if label_deltas else 0.0

        if threat_emojis:
            primary_sentiment = "threatening"
        elif obscene_emojis:
            primary_sentiment = "obscene"
        elif sarcasm_detected:
            primary_sentiment = "sarcastic"
        elif hostile_emojis:
            primary_sentiment = "hostile"
        elif positive_emojis:
            primary_sentiment = "positive"
        else:
            primary_sentiment = "neutral"

        return {
            "has_emojis": True,
            "is_emoji_only": emoji_only,
            "emojis_detected": detected_details,
            "emoji_count": len(emoji_chars),
            "unique_emoji_count": len(unique_emojis),
            "sarcasm_detected": sarcasm_detected,
            "is_mitigating": is_mitigating,
            "emoji_toxicity_score": round(max_toxic_signal, 4),
            "primary_sentiment": primary_sentiment,
            "label_deltas": label_deltas,
            "label_overrides": label_overrides,
            "context_reasons": context_reasons,
            "ui_chips": ui_chips
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "has_emojis": False,
            "is_emoji_only": False,
            "emojis_detected": [],
            "emoji_count": 0,
            "unique_emoji_count": 0,
            "sarcasm_detected": False,
            "is_mitigating": False,
            "emoji_toxicity_score": 0.0,
            "primary_sentiment": "none",
            "label_deltas": {l: 0.0 for l in DEFAULT_LABELS},
            "label_overrides": {},
            "context_reasons": [],
            "ui_chips": []
        }


# Singleton instance
_GLOBAL_EMOJI_ANALYZER = EmojiContextAnalyzer()

def get_emoji_analyzer() -> EmojiContextAnalyzer:
    return _GLOBAL_EMOJI_ANALYZER


def compare_predictions(
    base_result: Dict[str, Any],
    enhanced_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare baseline text-only prediction with emoji-enhanced prediction.
    Highlights shifts in toxicity classification, severity, probability deltas,
    and explains the emoji contribution.
    """
    base_toxic = base_result.get("is_toxic", False)
    enh_toxic = enhanced_result.get("is_toxic", False)
    base_sev = base_result.get("severity", "Clean")
    enh_sev = enhanced_result.get("severity", "Clean")
    base_flags = base_result.get("flagged_labels", [])
    enh_flags = enhanced_result.get("flagged_labels", [])

    base_preds = base_result.get("predictions", {})
    enh_preds = enhanced_result.get("predictions", {})

    prob_deltas = {}
    for lbl in DEFAULT_LABELS:
        p_base = base_preds.get(lbl, {}).get("probability", 0.0)
        p_enh = enh_preds.get(lbl, {}).get("probability", 0.0)
        prob_deltas[lbl] = round(p_enh - p_base, 4)

    verdict_changed = (base_toxic != enh_toxic)
    severity_changed = (base_sev != enh_sev)

    emoji_meta = enhanced_result.get("emoji_analysis", {})
    emojis_detected = enhanced_result.get("emojis_detected", [])

    if verdict_changed:
        if enh_toxic and not base_toxic:
            impact_type = "False Negative Resolved"
            summary = (
                f"Toxicity detected by emoji integration: Baseline text model classified comment as Non-Toxic "
                f"({base_sev}), but emoji analysis identified toxicity via {', '.join([e['name'] for e in emojis_detected])}. "
                f"Upgraded to Toxic ({enh_sev})."
            )
        else:
            impact_type = "False Positive Mitigated"
            summary = (
                f"False positive mitigated: Baseline model flagged toxic ({base_sev}), but constructive/mitigating "
                f"emojis confirmed friendly or benign intent. Corrected to Non-Toxic."
            )
    elif severity_changed:
        impact_type = "Severity Calibrated"
        summary = (
            f"Severity updated from {base_sev} to {enh_sev} based on emoji intensifiers / combination signals."
        )
    elif emoji_meta.get("has_emojis"):
        impact_type = "Consistent with Emoji Corroboration"
        summary = (
            f"Both models agree on '{'Toxic' if enh_toxic else 'Non-Toxic'}', with emoji semantics providing "
            f"corroborating contextual signals ({len(emojis_detected)} emoji(s) analyzed)."
        )
    else:
        impact_type = "Identical (Text-Only Comment)"
        summary = "No emojis present; output driven by linguistic text models."

    return {
        "text": enhanced_result.get("raw_text", ""),
        "verdict_changed": verdict_changed,
        "severity_changed": severity_changed,
        "impact_type": impact_type,
        "summary": summary,
        "base_model": {
            "is_toxic": base_toxic,
            "classification": "Toxic" if base_toxic else "Non-Toxic",
            "severity": base_sev,
            "flagged_labels": base_flags,
            "probabilities": {lbl: base_preds.get(lbl, {}).get("probability", 0.0) for lbl in DEFAULT_LABELS}
        },
        "enhanced_model": {
            "is_toxic": enh_toxic,
            "classification": "Toxic" if enh_toxic else "Non-Toxic",
            "severity": enh_sev,
            "flagged_labels": enh_flags,
            "probabilities": {lbl: enh_preds.get(lbl, {}).get("probability", 0.0) for lbl in DEFAULT_LABELS}
        },
        "prob_deltas": prob_deltas,
        "emojis_detected": emojis_detected,
        "ui_chips": enhanced_result.get("emoji_ui_chips", [])
    }

