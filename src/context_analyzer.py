"""
src/context_analyzer.py

Contextual NLP and semantic sentence analysis engine for toxic comment detection.
Ensures comments are evaluated holistically based on complete sentence meaning,
syntactic context, negation scope, intent, and conversational cues, rather than
classifying based merely on the presence of isolated sensitive words.
"""

import re
import html
import unicodedata
from typing import Dict, Any, List, Optional, Tuple

NEGATION_TERMS = {
    "not", "never", "no", "neither", "nor", "hardly", "scarcely",
    "barely", "cannot", "cant", "can't", "dont", "don't", "doesnt",
    "doesn't", "didnt", "didn't", "wont", "won't", "wouldnt", "wouldn't",
    "shouldnt", "shouldn't", "isnt", "isn't", "arent", "aren't", "wasnt",
    "wasn't", "werent", "weren't", "aint", "ain't", "without"
}

PRAISE_WORDS = {
    "good", "great", "awesome", "cool", "nice", "amazing", "love", "loved",
    "legend", "legendary", "fantastic", "masterpiece", "fire", "dope", "brilliant",
    "beautiful", "wonderful", "funny", "hilarious", "talented", "skilled",
    "helpful", "gem", "classic", "clean", "perfect", "underrated", "genius", "banger",
    "impressive", "outstanding", "wholesome", "favourite", "favorite"
}

INTENSIFIERS = {
    "damn", "damned", "fucking", "freaking", "bloody", "hell", "badass",
    "sick", "crazy", "insane", "wicked"
}

IDENTITY_WORDS = {
    "gay", "lesbian", "homosexual", "lgbt", "lgbtq", "trans", "transgender",
    "queer", "muslim", "islam", "islamic", "jewish", "jew", "christian",
    "christianity", "catholic", "black", "white", "african", "asian",
    "hispanic", "latino", "woman", "women", "female", "man", "men", "male",
    "mental illness", "depression", "autistic", "autism", "schizophrenia"
}

INSULT_WORDS = {
    "idiot", "moron", "stupid", "dumb", "retard", "clown", "loser",
    "jerk", "asshole", "bitch", "bastard", "scumbag", "trash", "garbage",
    "pig", "cunt", "ugly", "pathetic", "disgusting", "incompetent", "imbecile",
    "useless", "worthless", "failure", "hopeless", "freak", "scum", "creep",
    "suck", "sucks", "annoying", "clueless", "horrible", "terrible", "shitty", "dumbass"
}

THREAT_WORDS = {
    "kill", "murder", "destroy", "hunt", "hurt", "strangle", "beat", "die", "slaughter"
}

GRATITUDE_MARKERS = [
    "thank", "thanks", "helpful", "informative", "tutorial", "well done",
    "kudos", "appreciate", "great work", "loved this", "bless", "congrats",
    "congratulations", "keep it up", "respect"
]


def mark_negation_and_context(text: Any) -> str:
    """
    Context-aware text normalization and syntactic tagging:
      1. Resolves colloquial praise idioms (e.g. 'killed it', 'damn good', 'badass')
         so they are not misinterpreted as violence or profanity.
      2. Normalizes 1st-person self-statements and self-identity mentions.
      3. Scopes negation across clauses (tagging governed tokens as NOT_token)
         so linear and n-gram classifiers do not falsely trigger on negated terms
         (e.g., 'not an idiot' -> 'not NOT_an NOT_idiot').
    """
    if text is None:
        return ""
    text = str(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKD", text)

    # 1. Capture colloquial praise idioms
    text = re.sub(r"\b(killed|slayed|nailed)\s+it\b", " praise_idiom_killed_it ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(damn|damned)\s+(good|fine|cool|awesome|great|impressive|well)\b", r" praise_intensified_\2 ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(holy\s+shit|hell\s+yeah)\b", " praise_enthusiasm ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(fucking|freaking)\s+(awesome|amazing|great|good|cool|brilliant)\b", r" praise_intensified_\2 ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bbadass\b", " praise_badass ", text, flags=re.IGNORECASE)

    # 2. Capture constructive self-identification
    text = re.sub(
        r"\b(i\s+am|i'm|im|as\s+a)\s+(a\s+|an\s+)?(gay|lesbian|homosexual|muslim|jewish|christian|black|white|trans|woman|man)\b",
        r" self_identity_\3 ",
        text,
        flags=re.IGNORECASE
    )

    # 3. Propagate negation across clauses while preserving emojis
    import emoji
    em_list = emoji.emoji_list(text)
    placeholders = {}
    temp_text = text.lower()
    for i, m in enumerate(reversed(em_list)):
        idx = len(em_list) - 1 - i
        token = f"emojitok{idx}"
        placeholders[token] = m["emoji"]
        temp_text = temp_text[:m["match_start"]] + f" {token} " + temp_text[m["match_end"]:]

    raw_tokens = re.findall(r"[a-z0-9\u0C80-\u0CFF'_]+|[.,;:!?]", temp_text)
    tokens = []
    in_negation = False

    for tok in raw_tokens:
        if tok in [".", ",", ";", ":", "!", "?"]:
            in_negation = False
            continue
        if tok in ("but", "however", "although", "though", "yet", "except"):
            in_negation = False
            tokens.append(tok)
            continue
        if tok in NEGATION_TERMS:
            in_negation = True
            tokens.append(tok)
            continue

        if tok in placeholders:
            tokens.append(placeholders[tok])
        elif in_negation:
            tokens.append(f"NOT_{tok}")
        else:
            tokens.append(tok)

    return " ".join(tokens)


class SentenceContextAnalyzer:
    """
    Semantic Sentence Context Evaluator.
    Analyzes the complete sentence context, interpersonal direction, negation,
    and semantic intent. Prevents false positive classifications triggered by
    lone isolated words in benign, appreciative, or descriptive contexts.
    """

    def __init__(self):
        pass

    def evaluate(
        self,
        raw_text: str,
        ml_probs: Dict[str, float],
        ml_thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Evaluate sentence context against raw ML outputs.
        Returns dictionary containing:
          - final_is_toxic: bool
          - final_flags: List[str]
          - adjusted_probs: Dict[str, float]
          - context_reasons: List[str]
          - context_summary: str
        """
        text_clean = str(raw_text).lower().strip()
        tokens = re.findall(r"[a-z0-9']+", text_clean)
        num_words = len(tokens)

        # Lexical detection of components
        found_insults = [w for w in INSULT_WORDS if re.search(rf"\b{re.escape(w)}\b", text_clean)]
        found_threats = [w for w in THREAT_WORDS if re.search(rf"\b{re.escape(w)}\b", text_clean)]
        found_identities = [w for w in IDENTITY_WORDS if re.search(rf"\b{re.escape(w)}\b", text_clean)]
        found_intensifiers = [w for w in INTENSIFIERS if re.search(rf"\b{re.escape(w)}\b", text_clean)]
        found_praise = [w for w in PRAISE_WORDS if re.search(rf"\b{re.escape(w)}\b", text_clean)]

        # Context indicators
        has_gratitude = any(g in text_clean for g in GRATITUDE_MARKERS)
        has_praise_idiom = bool(re.search(
            r"\b(killed it|slayed|nailed it|damn good|damn fine|hell yeah|badass|holy shit this is|fucking awesome|fucking amazing)\b",
            text_clean
        ))

        # Check negation of negative terms within the same clause
        has_negated_negative = bool(re.search(
            r"\b(not|never|no|don't|dont|didn't|didnt|isn't|isnt|aren't|arent|wasn't|wasnt|hardly|scarcely|without)\b[^.,;:!?]{1,50}?\b(stupid|idiot|moron|dumb|bad|terrible|horrible|loser|hate|hater|threat|criminal|killer|ugly|trash|garbage|jerk|cunt|bitch)\b",
            text_clean
        ))

        # Self-identity / constructive identity reference
        is_self_identity = bool(re.search(
            r"\b(i\s+am|i'm|im|as\s+a|my|i\s+feel)\s+(a\s+|an\s+)?(gay|lesbian|homosexual|muslim|jewish|christian|black|white|trans|woman|man|stupid|idiot)\b",
            text_clean
        ))

        # Self-deprecation / self-directed negative remark (e.g. "I feel so stupid today")
        is_self_deprecating = bool(re.search(
            r"\b(i\s+am|i'm|im|i\s+feel|i\s+was|made\s+myself\s+look|i've\s+been|call\s+myself|my\s+mistake)\s+(so\s+|really\s+|pretty\s+|such\s+a\s+|a\s+|an\s+)?(stupid|dumb|idiot|moron|fool|loser|crazy|bad)\b",
            text_clean
        ))

        # Descriptive / educational context (e.g. books, movies, history)
        is_descriptive = bool(re.search(
            r"\b(movie|book|film|song|video|documentary|article|story|character|history|actor|director|tutorial|lesson)\s+(about|is|has|was|shows|explains)\b",
            text_clean
        ))

        # Directed personal attack (ensuring it is not preceded by negation)
        attack_match = re.search(
            r"\b(you|u|ur|your)\s+(are|r|is|being|look|sound)?\s*(a\s+|an\s+)?(complete|total|real|such\s+a|fucking|absolute|so)?\s*(idiot|moron|stupid|dumb|loser|bitch|bastard|asshole|clown|trash|garbage|jerk|cunt|pig|useless|worthless|pathetic|failure|incompetent|disgusting|ugly|hopeless|scum|creep|horrible|terrible|shitty|dumbass|imbecile)\b",
            text_clean
        ) or re.search(
            r"\b(you|u)\s+(suck|sucks|stink|have\s+no\s+(talent|brain|brains|sense|life|clue))\b",
            text_clean
        ) or re.search(
            r"\b(what\s+an?|such\s+an?|good\s+job|nice\s+one|way\s+to\s+go)\s+(idiot|moron|stupid|loser|clown|failure|joke|disaster|dumbass)\b",
            text_clean
        )
        is_directed_attack = False
        if attack_match:
            match_start = attack_match.start()
            prefix = text_clean[max(0, match_start - 35):match_start]
            if not any(neg in prefix for neg in ["not", "never", "no", "don't", "dont", "didn't", "didnt", "isn't", "isnt"]):
                is_directed_attack = True

        if re.search(r"\b(shut\s+up|get\s+lost|kill\s+yourself|go\s+to\s+hell|fuck\s+off|fuck\s+you|stfu)\b", text_clean):
            is_directed_attack = True

        # Directed violent threat
        is_violent_threat = bool(re.search(
            r"\b(i\s+will|i'll|imma|gonna)\s+(kill|hunt|hurt|beat|destroy|murder)\s+(you|u)\b",
            text_clean
        ))

        # Raw flags based on input ML probabilities
        raw_flags = [l for l, p in ml_probs.items() if p >= ml_thresholds.get(l, 0.5)]

        total_hostile_words = len(found_insults) + len(found_threats)
        has_single_word_trigger = (total_hostile_words <= 1) and (not is_directed_attack) and (not is_violent_threat)

        adjusted_probs = dict(ml_probs)
        adjusted_flags = list(raw_flags)
        context_reasons = []

        # RULE 1: Negated negative term (e.g. "I do not think you are stupid at all")
        if has_negated_negative and not is_directed_attack and not is_violent_threat:
            context_reasons.append("Negation modifies negative term into non-hostile statement")
            for l in ["toxic", "insult", "severe_toxic", "obscene"]:
                adjusted_probs[l] = min(adjusted_probs.get(l, 0.0), 0.15)
                if l in adjusted_flags:
                    adjusted_flags.remove(l)

        # RULE 2: Praise idioms & intensifiers (e.g. "This movie is damn good!")
        if (has_praise_idiom or (found_intensifiers and found_praise)) and not is_directed_attack and not is_violent_threat:
            context_reasons.append("Intensifier or idiom used in praise/enthusiastic context")
            for l in ["toxic", "obscene", "threat", "insult", "severe_toxic"]:
                adjusted_probs[l] = min(adjusted_probs.get(l, 0.0), 0.15)
                if l in adjusted_flags:
                    adjusted_flags.remove(l)

        # RULE 3: Self-referential or self-deprecating remarks
        if is_self_deprecating and not is_directed_attack and not is_violent_threat:
            context_reasons.append("Self-referential or self-deprecating remark without hostility towards others")
            for l in ["toxic", "insult", "severe_toxic", "obscene"]:
                adjusted_probs[l] = min(adjusted_probs.get(l, 0.0), 0.15)
                if l in adjusted_flags:
                    adjusted_flags.remove(l)

        # RULE 4: Identity self-reference or neutral mention without hate
        if (is_self_identity or found_identities) and not is_directed_attack and not is_violent_threat:
            has_hate_speech = bool(re.search(
                r"\b(hate|kill|subhuman|scum|burn|destroy)\s+(all\s+)?(gays|muslims|jews|blacks|women|trans)\b",
                text_clean
            ))
            if not has_hate_speech and len(found_insults) == 0:
                context_reasons.append("Identity term mentioned constructively or self-referentially without hate")
                for l in ["identity_hate", "toxic", "insult", "threat"]:
                    adjusted_probs[l] = min(adjusted_probs.get(l, 0.0), 0.12)
                    if l in adjusted_flags:
                        adjusted_flags.remove(l)

        # RULE 5: Descriptive / educational / discursive context
        if is_descriptive and not is_directed_attack and not is_violent_threat and total_hostile_words <= 1:
            context_reasons.append("Neutral descriptive or informational reference")
            for l in ["toxic", "threat", "severe_toxic"]:
                adjusted_probs[l] = min(adjusted_probs.get(l, 0.0), 0.15)
                if l in adjusted_flags:
                    adjusted_flags.remove(l)

        # RULE 6: Overall gratitude / respectful tone with isolated ambiguity
        if has_gratitude and not is_directed_attack and not is_violent_threat and total_hostile_words <= 1:
            context_reasons.append("Overall comment expresses gratitude or respect")
            for l in ["toxic", "insult", "obscene"]:
                adjusted_probs[l] = min(adjusted_probs.get(l, 0.0), 0.15)
                if l in adjusted_flags:
                    adjusted_flags.remove(l)

        # RULE 7: Lone-word trigger check (mitigate single sensitive term in benign/descriptive contexts)
        if has_single_word_trigger and not is_directed_attack and not is_violent_threat:
            if len(adjusted_flags) > 0 and len(found_insults) == 1 and len(found_threats) == 0:
                if has_gratitude or has_praise_idiom or is_self_identity or is_descriptive or is_self_deprecating:
                    context_reasons.append("Single isolated word evaluated in sentence context without hostile intent")
                    adjusted_flags = []
                    for l in ml_probs:
                        adjusted_probs[l] = min(adjusted_probs[l], ml_thresholds.get(l, 0.5) * 0.75)

        # RULE 7: Confirm true targeted toxicity
        if is_directed_attack or is_violent_threat:
            context_reasons.append("Direct targeted hostility detected in sentence structure")
            if is_directed_attack:
                adjusted_probs["toxic"] = max(adjusted_probs.get("toxic", 0), 0.92)
                adjusted_probs["insult"] = max(adjusted_probs.get("insult", 0), 0.88)
                if "toxic" not in adjusted_flags:
                    adjusted_flags.append("toxic")
                if "insult" not in adjusted_flags:
                    adjusted_flags.append("insult")
            if is_violent_threat:
                adjusted_probs["threat"] = max(adjusted_probs.get("threat", 0), 0.90)
                adjusted_probs["toxic"] = max(adjusted_probs.get("toxic", 0), 0.92)
                if "threat" not in adjusted_flags:
                    adjusted_flags.append("threat")
                if "toxic" not in adjusted_flags:
                    adjusted_flags.append("toxic")

        final_is_toxic = len(adjusted_flags) > 0

        # Construct explanation
        if context_reasons:
            context_summary = "; ".join(context_reasons)
        else:
            context_summary = "Classified based on overall linguistic context"

        return {
            "final_is_toxic": final_is_toxic,
            "final_flags": adjusted_flags,
            "adjusted_probs": adjusted_probs,
            "context_reasons": context_reasons,
            "context_summary": context_summary,
            "is_directed_attack": is_directed_attack,
            "is_violent_threat": is_violent_threat,
            "has_negated_negative": has_negated_negative,
            "has_praise_idiom": has_praise_idiom,
            "is_self_identity": is_self_identity
        }


# Singleton instance for efficient reuse
_GLOBAL_CONTEXT_ANALYZER = SentenceContextAnalyzer()

def get_context_analyzer() -> SentenceContextAnalyzer:
    return _GLOBAL_CONTEXT_ANALYZER
