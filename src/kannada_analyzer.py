"""
src/kannada_analyzer.py

Contextual NLP and Machine Learning engine for Kannada toxic comment detection.
Handles:
  - Language detection for Kannada script (ಕನ್ನಡ), Romanized Kannada (Kanglish), and English.
  - Multi-label ML toxicity inference using trained Kannada classifiers.
  - Deep semantic sentence evaluation for Kannada:
      * Negation scoping (e.g., 'ನೀನು ದಡ್ಡ ಅಲ್ಲ' / 'neenu dadda alla' -> non-toxic)
      * Colloquial praise and enthusiasm (e.g., 'ಬೆಂಕಿ ಗುರು', 'ತುಂಬಾ ಅದ್ಭುತ')
      * Self-referential remarks (e.g., 'ನಾನು ದಡ್ಡ, ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ')
      * Gratitude and constructive engagement (e.g., 'ಧನ್ಯವಾದಗಳು')
      * Targeted personal attacks, obscenity, and violent threats.
"""

import os
import re
import json
import html
import unicodedata
import joblib
from typing import Dict, Any, List, Optional, Tuple

from src.preprocessing import clean_text, DEFAULT_LABELS

# -----------------------------------------------------------------------------
# 1. Kannada Semantic Lexicons (Script & Romanized Kanglish)
# -----------------------------------------------------------------------------

# Negation terms in Kannada (script and transliterated)
KN_NEGATION_TERMS = {
    # Script
    "ಇಲ್ಲ", "ಅಲ್ಲ", "ಅಲ್ವೇ", "ಅಲ್ಲವೇ", "ಬೇಡ", "ಮಾಡಬೇಡಿ", "ಮಾಡಬೇಡ", "ಮಾಡ್ದೇ",
    "ಮಾಡಬೇಡಿ", "ತಪ್ಪಲ್ಲ", "ಇಲ್ಲವೇ", "ಇಲ್ಲವೇಇಲ್ಲ", "ಅಲ್ಲವೇಅಲ್ಲ", "ಇರ್ಲಿಲ್ಲ",
    # Romanized
    "illa", "alla", "allave", "allve", "beda", "madbeda", "madbedi",
    "tappalla", "illave", "allavealla", "irlilla", "iralla"
}

# Praise and appreciation terms in Kannada
KN_PRAISE_WORDS = {
    # Script
    "ಚೆನ್ನಾಗಿದೆ", "ಚೆನ್ನಾಗಿ", "ಅದ್ಭುತ", "ಅದ್ಭುತವಾಗಿದೆ", "ಅತ್ಯುತ್ತಮ", "ಸೂಪರ್",
    "ಸುಂದರ", "ಸುಂದರವಾಗಿದೆ", "ಹೆಮ್ಮೆ", "ಹೆಮ್ಮೆಯ", "ಶ್ರೇಷ್ಠ", "ಮುದ್ದಾಗಿದೆ",
    "ಇಷ್ಟವಾಯಿತು", "ಇಷ್ಟ", "ಖುಷಿಯಾಯಿತು", "ಖುಷಿ", "ಬೆಂಕಿ", "ಗ್ರೇಟ್", "ಉಪಯುಕ್ತ",
    "ಉಪಯುಕ್ತವಾಗಿದೆ", "ಒಳ್ಳೆಯದು", "ಒಳ್ಳೆಯ", "ಮುಂದುವರಿಸಿ", "ಗೆಲುವು", "ಸಾಧನೆ",
    # Romanized
    "chennagide", "chennagi", "adbhuta", "adbhutha", "adbhutavagide", "uttama",
    "atyuttama", "super", "sundara", "hemme", "shrestha", "muddagide",
    "istavayithu", "ishta", "kushi", "kushiyayithu", "benki", "great",
    "upayukta", "olleyadu", "olleyadagali", "munduvarisi", "geluvu"
}

# Gratitude and respectful markers
KN_GRATITUDE_MARKERS = {
    # Script
    "ಧನ್ಯವಾದಗಳು", "ಧನ್ಯವಾದ", "ಕೃತಜ್ಞತೆಗಳು", "ಕೃತಜ್ಞತೆ", "ನಮಸ್ಕಾರ", "ನಮಸ್ಕಾರಗಳು",
    "ಅಭಿನಂದನೆಗಳು", "ಶುಭಾಶಯಗಳು", "ಹಾರೈಕೆಗಳು", "ಗೌರವ", "ಪ್ರೀತಿ", "ಆಶೀರ್ವಾದ",
    "ಶುಭವಾಗಲಿ", "ಜೈ", "ಕರ್ನಾಟಕ", "ಧನ್ಯೋಸ್ಮಿ", "ಪ್ರಣಾಮಗಳು",
    # Romanized
    "dhanyavadagalu", "dhanyavada", "dhanyavadha", "krutajnategalu", "namaskara",
    "namaskaragalu", "abhinandhanegalu", "shubhashayagalu", "gaurava", "priti",
    "aashirvada", "shubhavagali", "jai", "karnataka", "pranamagalu"
}

# Insult and abusive terms
KN_INSULT_WORDS = {
    # Script
    "ಮೂರ್ಖ", "ಮೂರ್ಖರು", "ಮೂರ್ಖತನ", "ಕತ್ತೆ", "ಹುಚ್ಚ", "ಹುಚ್ಚಿ", "ಹುಚ್ಚರು",
    "ಹುಚ್ಚಾಟ", "ನಾಯಿ", "ಹಂದಿ", "ಭಿಕಾರಿ", "ಭಿಕ್ಷುಕ", "ಲೋಫರ್", "ರಾಸ್ಕಲ್",
    "ಕಿತ್ತೋದವನೆ", "ಕಿತ್ತೋದವಳು", "ಕಿತ್ತೋದವ್ರೆ", "ದರಿದ್ರ", "ದರಿದ್ರದವನೆ",
    "ಅಯೋಗ್ಯ", "ಅಯೋಗ್ಯರು", "ನೀಚ", "ನೀಚರು", "ಹೇಡಿ", "ಪಾಪಿ", "ದಡ್ಡ", "ದಡ್ಡಿ",
    "ದಡ್ಡರು", "ಕೆಟ್ಟವನು", "ಕೆಟ್ಟವಳು", "ಮಾನಗೆಟ್ಟವನೆ", "ಬುದ್ಧಿಹೀನ", "ತಿಕ್ಕಲು",
    "ಗೂಬೆ", "ಕೋತಿ", "ಚಿಲ್ಲರೆ", "ನಿಷ್ಪ್ರಯೋಜಕ", "ಸೋಮಾರಿ", "ಲಜ್ಜೆಗೆಟ್ಟ",
    "ಬೇವರ್ಸಿ", "ಮುಸುಡಿ", "ನಾಲಾಯಕ್", "ತೊಲಗು", "ಬಾಯಿಮುಚ್ಚು", "ಬಾಯಿಮುಚ್ಕೋ",
    # Romanized
    "murcha", "moorkha", "moorukha", "katte", "huccha", "huchha", "huchhi",
    "naayi", "nayi", "handi", "bhikari", "loafer", "rascal", "kittodavne",
    "kittodavle", "daridra", "ayogya", "ayoggya", "neecha", "hedi", "dadda",
    "daddi", "kettavne", "buddihina", "thiklu", "tiklu", "goobe", "kothi",
    "chillare", "nishprayojaka", "somari", "lajjegetta", "bevarsi", "musudi",
    "nalayak", "tholagu", "baayimucchu", "baayimuchko"
}

# Obscene and vulgar slurs
KN_OBSCENE_WORDS = {
    # Script
    "ಸೂಳೆ", "ಸೂಳೆಮಗನೆ", "ಸೂಳೆಮಗ", "ಸೂಳೆಮಕ್ಕಳ", "ಬೋಳಿಮಗನೆ", "ಬೋಳಿಮಗ",
    "ಬೋಳಿಮಕ್ಕಳ", "ತಿಕ", "ತಿಕಮುಚ್ಕೊಂಡು", "ತಿಕಮುಚ್ಚು", "ತುಣ್ಣೆ", "ತುಲ್ಲು",
    "ಹಲ್ಕಟ್", "ಹಲ್ಕಾ", "ಹರಾಮ್ಖೋರ್", "ಹರಾಮಿ", "ರಾಂಡಿ", "ಕಾಮಪಿಶಾಚಿ",
    "ಚ್ಯೂತ್ಯ", "ಚುತ್ಯ", "ಲಫಂಗ",
    # Romanized
    "soole", "soolemaga", "soolemagane", "soolemakla", "bolimaga",
    "bolimagane", "bolimakla", "thika", "tika", "thika muchu",
    "thika muchkondu", "thunne", "thullu", "halkat", "halka", "haramkhor",
    "harami", "randi", "kamapisachi", "chutiya", "lafang"
}

# Violent threat terms
KN_THREAT_WORDS = {
    # Script
    "ಕೊಲ್ಲುತ್ತೇನೆ", "ಕೊಲ್ತೀನಿ", "ಕೊಲೆ", "ಮುಗಿಸ್ತೀನಿ", "ಮುಗಿಸಿಬಿಡ್ತೀನಿ",
    "ಹೊಡಿತೀನಿ", "ಹೊಡೆದು", "ಸಾಯಿಸ್ತೀನಿ", "ಸಾಯಿಸುತ್ತೇನೆ", "ಜೀವ", "ತೆಗೀತೀನಿ",
    "ರಕ್ತ", "ಕುಡೀತೀನಿ", "ಕೈಕಾಲು", "ಮುರೀತೀನಿ", "ಸರ್ವನಾಶ", "ಹಾಳುಮಾಡ谑್ತೀನಿ",
    "ಸುಟ್ಟು", "ಜೀವಂತ", "ಕತ್ತು", "ಹಿಸುಕಿ",
    # Romanized
    "kolluthene", "koltini", "kole", "mugisteeni", "mugisbittini",
    "hodithini", "hodedu", "sayistini", "sayisuttene", "thegitheeni",
    "raktha", "kudithini", "kaikaalu", "murithini", "sarvanasha", "suttu",
    "haaktini", "madtini", "madthini", "hushar", "jeeva"
}

# Distinctive Kanglish lexicon used to detect romanized Kannada
# Excludes general English words like 'great', 'super', 'awesome', 'bad'
DISTINCTIVE_KANGLISH_WORDS = {
    # Pronouns & common particles in Kannada
    "neenu", "neen", "neevu", "naanu", "naan", "nanna", "nanage", "ninge",
    "nimage", "namage", "beku", "beda", "madbeda", "madbedi", "nodkollo",
    "yavaga", "yenu", "hege", "idu", "adhu", "kelsa", "makla", "gothilla",
    "dayavittu", "kannada", "kannadiga", "karnataka", "alla", "illa", "allve",
    "allave", "tappalla", "illave", "irlilla", "iralla", "guru", "magane",
    "chennagide", "chennagi", "dhanyavada", "dhanyavadagalu", "dhanyavadha",
    "namaskara", "namaskaragalu", "krutajnategalu", "abhinandhanegalu",
    "shubhashayagalu", "shubhavagali", "adbhuta", "adbhutha", "adbhutavagide",
    "uttama", "atyuttama", "muddagide", "istavayithu", "kushiyayithu", "benki",
    "olleyadu", "olleyadagali", "munduvarisi", "geluvu", "thumba", "tumba",
    # Insults & slurs in Kanglish
    "huccha", "huchha", "huchhi", "katte", "baddimaga", "baddimagane", "soole",
    "soolemaga", "soolemagane", "soolemakla", "bolimaga", "bolimagane",
    "bolimakla", "thika", "tika", "thunne", "thullu", "halkat", "halka",
    "haramkhor", "harami", "randi", "bevarsi", "kittodavne", "kittodavle",
    "daridra", "ayogya", "ayoggya", "neecha", "hedi", "dadda", "daddi",
    "kettavne", "buddihina", "thiklu", "tiklu", "goobe", "kothi", "chillare",
    "nishprayojaka", "somari", "lajjegetta", "musudi", "nalayak", "tholagu",
    "baayimucchu", "baayimuchko", "kamapisachi", "chutiya", "lafang",
    # Threats in Kanglish
    "kolluthene", "koltini", "kole", "mugisteeni", "mugisbittini",
    "hodithini", "hodedu", "sayistini", "sayisuttene", "thegitheeni",
    "raktha", "kudithini", "kaikaalu", "murithini", "sarvanasha", "suttu",
    "haaktini", "madtini", "madthini", "hushar", "jeeva"
}


# -----------------------------------------------------------------------------
# 2. Language Detection
# -----------------------------------------------------------------------------

def detect_language(text: Any) -> Dict[str, Any]:
    """
    Detect whether text is Kannada (Kannada script or Kanglish), English, or Mixed.
    Returns:
      {
        "language": "kannada" | "english" | "mixed",
        "is_kannada": bool,
        "script": "kannada" | "latin" | "mixed",
        "kannada_char_count": int,
        "kannada_ratio": float,
        "display_name": str
      }
    """
    if not text:
        return {
            "language": "english",
            "is_kannada": False,
            "script": "latin",
            "kannada_char_count": 0,
            "kannada_ratio": 0.0,
            "display_name": "English"
        }

    text_str = str(text)
    total_chars = len(text_str.strip())
    if total_chars == 0:
        return {
            "language": "english",
            "is_kannada": False,
            "script": "latin",
            "kannada_char_count": 0,
            "kannada_ratio": 0.0,
            "display_name": "English"
        }

    # Count Kannada Unicode characters (U+0C80 to U+0CFF)
    kannada_chars = len(re.findall(r"[\u0C80-\u0CFF]", text_str))
    latin_chars = len(re.findall(r"[a-zA-Z]", text_str))
    kannada_ratio = kannada_chars / total_chars if total_chars > 0 else 0.0

    # 1. Native Kannada script detected
    if kannada_chars > 0:
        if latin_chars > 0 and kannada_chars >= 3 and latin_chars >= 5:
            return {
                "language": "mixed",
                "is_kannada": True,
                "script": "mixed",
                "kannada_char_count": kannada_chars,
                "kannada_ratio": round(kannada_ratio, 3),
                "display_name": "Mixed (ಕನ್ನಡ / English)"
            }
        return {
            "language": "kannada",
            "is_kannada": True,
            "script": "kannada",
            "kannada_char_count": kannada_chars,
            "kannada_ratio": round(kannada_ratio, 3),
            "display_name": "Kannada (ಕನ್ನಡ)"
        }

    # 2. Check for Romanized Kannada (Kanglish) markers
    text_lower = text_str.lower()
    words = set(re.findall(r"[a-z']+", text_lower))
    kanglish_matches = words.intersection(DISTINCTIVE_KANGLISH_WORDS)

    if len(kanglish_matches) >= 2 or (len(kanglish_matches) >= 1 and len(words) <= 7):
        # Check if text is predominantly English with a single incidental marker
        english_particles = words.intersection({
            "the", "is", "this", "are", "you", "and", "in", "it", "to", "for",
            "of", "a", "an", "on", "with", "video", "tutorial", "great", "awesome", "good"
        })
        if len(english_particles) >= 3 and len(kanglish_matches) == 1:
            return {
                "language": "mixed",
                "is_kannada": True,
                "script": "latin",
                "kannada_char_count": 0,
                "kannada_ratio": 0.0,
                "display_name": "Mixed (ಕನ್ನಡ / English)"
            }
        return {
            "language": "kannada",
            "is_kannada": True,
            "script": "latin",
            "kannada_char_count": 0,
            "kannada_ratio": 0.0,
            "display_name": "Kannada (Kanglish)"
        }

    return {
        "language": "english",
        "is_kannada": False,
        "script": "latin",
        "kannada_char_count": 0,
        "kannada_ratio": 0.0,
        "display_name": "English"
    }


# -----------------------------------------------------------------------------
# 3. Kannada Semantic Sentence Context Evaluator
# -----------------------------------------------------------------------------

class KannadaContextAnalyzer:
    """
    Evaluates complete sentence context in Kannada (script and Kanglish).
    Correctly recognizes:
      - Negation (e.g. 'ನೀನು ದಡ್ಡ ಅಲ್ಲ' / 'not stupid' -> non-toxic)
      - Colloquial praise (e.g. 'ಬೆಂಕಿ ಗುರು', 'ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ' -> non-toxic)
      - Gratitude & respect (e.g. 'ಧನ್ಯವಾದಗಳು' -> non-toxic)
      - Self-deprecation (e.g. 'ನಾನು ದಡ್ಡ ನನಗೆ ತಿಳಿಯಲಿಲ್ಲ' -> non-toxic)
      - Targeted insults, obscenity, and violent threats -> toxic.
    """

    def evaluate(
        self,
        raw_text: str,
        ml_probs: Dict[str, float],
        ml_thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        text_clean = str(raw_text).lower().strip()
        tokens = set(re.findall(r"[a-z0-9\u0C80-\u0CFF']+", text_clean))

        # Lexical detection of components
        found_insults = [w for w in KN_INSULT_WORDS if re.search(rf"\b{re.escape(w)}\b", text_clean)]
        found_obscene = [w for w in KN_OBSCENE_WORDS if re.search(rf"\b{re.escape(w)}\b", text_clean)]
        found_threats = [w for w in KN_THREAT_WORDS if re.search(rf"\b{re.escape(w)}\b", text_clean)]
        found_praise = [w for w in KN_PRAISE_WORDS if re.search(rf"\b{re.escape(w)}\b", text_clean)]
        found_gratitude = [w for w in KN_GRATITUDE_MARKERS if re.search(rf"\b{re.escape(w)}\b", text_clean)]

        # Context indicators
        has_praise = len(found_praise) > 0
        has_gratitude = len(found_gratitude) > 0

        # Negation check: e.g. 'ದಡ್ಡ ಅಲ್ಲ', 'ಕೆಟ್ಟದಲ್ಲ', 'not stupid'
        has_negated_insult = bool(re.search(
            r"\b(ದಡ್ಡ|ಮೂರ್ಖ|ಹುಚ್ಚ|ಕೆಟ್ಟ|ಕತ್ತೆ|ನಾಯಿ|ಹಂದಿ|ಭಿಕಾರಿ|dadda|huccha|moorkha|katte|kettadu|kettadalla|bad|terrible)\s+(ಅಲ್ಲ|ಅಲ್ವೇ\s+ಅಲ್ಲ|ಅಲ್ಲವೇ\s+ಅಲ್ಲ|ಇಲ್ಲ|ತಪ್ಪಲ್ಲ|alla|allave\s+alla|illa)\b",
            text_clean
        )) or bool(re.search(
            r"\b(ಅಲ್ಲ|ಇಲ್ಲ|not|never|no)\b[^.,;:!?]{1,35}?\b(ದಡ್ಡ|ಮೂರ್ಖ|ಹುಚ್ಚ|ಕತ್ತೆ|dadda|huccha|moorkha|katte)\b",
            text_clean
        ))

        # Self-deprecation / self-reference: 'ನಾನು ದಡ್ಡ', 'naanu dadda'
        is_self_deprecating = bool(re.search(
            r"\b(ನಾನು|ನಾನ್|ನನ್ನ|ನನಗೆ|naanu|nanage|nanna)\s+(ದಡ್ಡ|ಮೂರ್ಖ|ಹುಚ್ಚ|ತಪ್ಪು|dadda|moorkha|huccha|thappu)\b",
            text_clean
        ))

        # Directed 2nd person attack: 'ನೀನು ಒಬ್ಬ ಮೂರ್ಖ', 'neenu obba huccha', 'ನೀನು ಕತ್ತೆ'
        attack_match = re.search(
            r"\b(ನೀನು|ನೀನ್|ನೀವು|ನಿನ್ನ|ನಿಂಗೆ|neenu|neen|neevu|ninna|ninge)\s+(ಒಬ್ಬ\s+|ದೊಡ್ಡ\s+|ತುಂಬಾ\s+|obba\s+)?(ಮೂರ್ಖ|ದಡ್ಡ|ಹುಚ್ಚ|ಕತ್ತೆ|ನಾಯಿ|ಹಂದಿ|ಬೇವರ್ಸಿ|ಕಿತ್ತೋದವನೆ|ದರಿದ್ರ|ಅಯೋಗ್ಯ|ಲೋಫರ್|ರಾಸ್ಕಲ್|ಕೋತಿ|ಗೂಬೆ|ಚಿಲ್ಲರೆ|huccha|dadda|katte|bevarsi|kittodavne|daridra|ayogya|rascal|loafer)\b",
            text_clean
        )
        is_directed_attack = False
        if attack_match and not has_negated_insult:
            is_directed_attack = True

        if bool(re.search(
            r"\b(ಬಾಯಿ\s+ಮುಚ್ಚು|ಬಾಯಿ\s+ಮುಚ್ಕೋ|ತೊಲಗು|baayi\s+mucchu|baayi\s+muchko|tholagu|get\s+lost)\b",
            text_clean
        )):
            is_directed_attack = True

        # Directed violent threat: 'ನಿನ್ನ ಕೊಲ್ಲುತ್ತೇನೆ', 'ninna mugistheeni'
        is_violent_threat = bool(re.search(
            r"\b(ನಿನ್ನ|ನಿನ್ನನ್ನು|ನಿನಗೆ|ninna|ninage)\s+[^.,;:!?]{0,30}?(ಕೊಲ್ಲುತ್ತೇನೆ|ಕೊಲ್ತೀನಿ|ಮುಗಿಸ್ತೀನಿ|ಹೊಡೆದು\s+ಸಾಯಿಸ್ತೀನಿ|ಜೀವ\s+ತೆಗೀತೀನಿ|ರಕ್ತ\s+ಕುಡೀತೀನಿ|ಸುಟ್ಟು\s+ಹಾಕ್ತೀನಿ|kolluthene|koltini|mugisteeni|sayistini|thegitheeni|sarvanasha)\b",
            text_clean
        )) or bool(re.search(
            r"\b(ಕೊಲ್ಲುತ್ತೇನೆ|ಕೊಲೆ\s+ಮಾಡ್ತೀನಿ|ಹೊಡೆದು\s+ಸಾಯಿಸ್ತೀನಿ|ಸಾಯಿಸುತ್ತೇನೆ|kolluthene|sayistini|jeeva\s+thegitheeni|thegitheeni|sarvanasha)\b",
            text_clean
        ))

        # Obscene insult: 'ಸೂಳೆಮಗನೆ', 'ಬೋಳಿಮಗ', 'thika muchu'
        is_obscene = len(found_obscene) > 0

        # Start with ML model probabilities
        adjusted_probs = dict(ml_probs)
        context_reasons = []

        # RULE 1: Negated negative comment (e.g. 'ನೀನು ದಡ್ಡ ಅಲ್ಲ', 'neenu dadda alla')
        if has_negated_insult and not is_directed_attack and not is_violent_threat:
            context_reasons.append("Kannada negation modifies negative term into non-hostile statement")
            for l in ["toxic", "insult", "severe_toxic", "obscene"]:
                adjusted_probs[l] = min(adjusted_probs.get(l, 0.0), 0.12)

        # RULE 2: Praise idioms & gratitude
        if (has_praise or has_gratitude) and not is_directed_attack and not is_violent_threat and not is_obscene:
            context_reasons.append("Kannada praise, gratitude or cultural enthusiasm context")
            for l in ["toxic", "obscene", "threat", "insult", "severe_toxic"]:
                adjusted_probs[l] = min(adjusted_probs.get(l, 0.0), 0.12)

        # RULE 3: Self-deprecation
        if is_self_deprecating and not is_directed_attack and not is_violent_threat:
            context_reasons.append("Self-referential or self-deprecating remark in Kannada without hostility to others")
            for l in ["toxic", "insult", "severe_toxic", "obscene"]:
                adjusted_probs[l] = min(adjusted_probs.get(l, 0.0), 0.12)

        # RULE 4: Directed attack enforcement
        if is_directed_attack and not has_negated_insult:
            context_reasons.append("Direct targeted personal insult in Kannada")
            adjusted_probs["toxic"] = max(adjusted_probs.get("toxic", 0), 0.94)
            adjusted_probs["insult"] = max(adjusted_probs.get("insult", 0), 0.91)

        # RULE 5: Violent threat enforcement
        if is_violent_threat:
            context_reasons.append("Violent threat or physical harm statement in Kannada")
            adjusted_probs["toxic"] = max(adjusted_probs.get("toxic", 0), 0.96)
            adjusted_probs["threat"] = max(adjusted_probs.get("threat", 0), 0.92)
            adjusted_probs["severe_toxic"] = max(adjusted_probs.get("severe_toxic", 0), 0.82)

        # RULE 6: Obscenity enforcement
        if is_obscene:
            context_reasons.append("Vulgarity or obscene slur in Kannada")
            adjusted_probs["toxic"] = max(adjusted_probs.get("toxic", 0), 0.95)
            adjusted_probs["obscene"] = max(adjusted_probs.get("obscene", 0), 0.92)

        # Determine flagged labels based on adjusted probabilities and thresholds
        flagged_labels = [
            l for l, p in adjusted_probs.items()
            if p >= ml_thresholds.get(l, 0.35)
        ]

        # Prevent single ambiguous trigger word without hostile intent
        if len(flagged_labels) > 0 and not is_directed_attack and not is_violent_threat and not is_obscene:
            if len(found_insults) <= 1 and (has_praise or has_gratitude or has_negated_insult or is_self_deprecating):
                flagged_labels = []
                for l in adjusted_probs:
                    adjusted_probs[l] = min(adjusted_probs[l], 0.15)
                context_reasons.append("Isolated sensitive term mitigated by benign Kannada context")

        final_is_toxic = len(flagged_labels) > 0

        summary = "; ".join(context_reasons) if context_reasons else "Evaluated with Kannada linguistic contextual engine"

        return {
            "final_is_toxic": final_is_toxic,
            "final_flags": flagged_labels,
            "adjusted_probs": adjusted_probs,
            "context_reasons": context_reasons,
            "context_summary": summary,
            "is_directed_attack": is_directed_attack,
            "is_violent_threat": is_violent_threat,
            "is_obscene": is_obscene,
            "has_negated_insult": has_negated_insult,
            "has_praise": has_praise
        }


# -----------------------------------------------------------------------------
# 4. Kannada ML Model Artifact Loader & Predictor
# -----------------------------------------------------------------------------

_LOADED_KN_ARTIFACTS = None
_GLOBAL_KN_CONTEXT_ANALYZER = KannadaContextAnalyzer()


def get_kannada_context_analyzer() -> KannadaContextAnalyzer:
    return _GLOBAL_KN_CONTEXT_ANALYZER


def load_kannada_artifacts(model_dir: str = "models", force_reload: bool = False) -> Dict[str, Any]:
    """
    Load Kannada TF-IDF vectorizer, 6 binary classifiers, and metadata into memory.
    Cached for fast repeated inferences.
    """
    global _LOADED_KN_ARTIFACTS
    if _LOADED_KN_ARTIFACTS is not None and not force_reload:
        return _LOADED_KN_ARTIFACTS

    vec_path = os.path.join(model_dir, "kannada_tfidf_vectorizer.pkl")
    model_path = os.path.join(model_dir, "kannada_multilabel_model.pkl")
    meta_path = os.path.join(model_dir, "kannada_model_metadata.json")

    if not os.path.exists(vec_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Kannada model artifacts missing in {model_dir}. Please run 'python scripts/train_kannada_model.py' first."
        )

    vectorizer = joblib.load(vec_path)
    classifiers = joblib.load(model_path)

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    thresholds = metadata.get("optimal_thresholds", {l: 0.35 for l in DEFAULT_LABELS})

    _LOADED_KN_ARTIFACTS = {
        "vectorizer": vectorizer,
        "classifiers": classifiers,
        "metadata": metadata,
        "thresholds": thresholds,
        "labels": metadata.get("labels", DEFAULT_LABELS)
    }
    return _LOADED_KN_ARTIFACTS


def predict_kannada_comment(
    raw_text: str,
    artifacts: Optional[Dict[str, Any]] = None,
    threshold_overrides: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Perform multi-label toxicity classification for a Kannada comment.
    Combines trained ML model with Kannada semantic context analyzer.
    """
    if artifacts is None:
        artifacts = load_kannada_artifacts()

    vectorizer = artifacts["vectorizer"]
    classifiers = artifacts["classifiers"]
    thresholds = dict(artifacts["thresholds"])
    if threshold_overrides:
        thresholds.update(threshold_overrides)

    labels = artifacts["labels"]

    # Preprocess text
    cleaned_text = clean_text(raw_text) or "unparsed"

    # ML Vectorization & raw probability prediction
    vec = vectorizer.transform([cleaned_text])

    raw_probs = {}
    for label in labels:
        clf = classifiers[label]
        prob = float(clf.predict_proba(vec)[0, 1])
        raw_probs[label] = prob

    # Kannada contextual evaluation
    context_analyzer = get_kannada_context_analyzer()
    context_eval = context_analyzer.evaluate(
        raw_text=raw_text,
        ml_probs=raw_probs,
        ml_thresholds=thresholds
    )

    adjusted_probs = context_eval["adjusted_probs"]
    flagged_labels = context_eval["final_flags"]
    is_toxic = context_eval["final_is_toxic"]

    label_predictions = {}
    for label in labels:
        prob = float(adjusted_probs[label])
        thresh = float(thresholds.get(label, 0.35))
        is_flagged = bool(label in flagged_labels)

        if prob >= 0.75:
            risk = "High"
        elif prob >= thresh:
            risk = "Moderate"
        elif prob >= 0.25:
            risk = "Elevated"
        else:
            risk = "Low"

        label_predictions[label] = {
            "probability": round(prob, 4),
            "percentage": round(prob * 100, 1),
            "threshold": round(thresh, 2),
            "flagged": is_flagged,
            "risk_level": risk
        }

    return {
        "is_toxic": is_toxic,
        "flagged_labels": flagged_labels,
        "adjusted_probs": adjusted_probs,
        "label_predictions": label_predictions,
        "context_eval": context_eval
    }
