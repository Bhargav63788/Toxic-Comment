"""
scripts/train_kannada_model.py

Training pipeline for Kannada Toxic Comment Classification.
Trains a contextual multi-label machine learning model supporting:
  - Kannada script comments (ಕನ್ನಡ ಲಿಪಿ)
  - Romanized Kannada comments (Kanglish)
  - Code-mixed Kannada + English comments
Across 6 target labels: toxic, severe_toxic, obscene, threat, insult, identity_hate.
Outputs model artifacts to models/ for production inference.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import clean_text, DEFAULT_LABELS

# Comprehensive, balanced training corpus for Kannada (Script, Kanglish, and Code-mixed)
KANNADA_TRAINING_DATA = [
    # -------------------------------------------------------------------------
    # 1. NON-TOXIC: Praise, appreciation, gratitude, and encouragement (Kannada Script)
    # -------------------------------------------------------------------------
    {"text": "ತುಂಬಾ ಧನ್ಯವಾದಗಳು, ಈ ವಿಡಿಯೋ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ ಮತ್ತು ಉಪಯುಕ್ತವಾಗಿದೆ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಅದ್ಭುತವಾದ ಮಾಹಿತಿ, ಧನ್ಯವಾದಗಳು ಸರ್ ನಿಮ್ಮ ಈ ಒಳ್ಳೆಯ ಕೆಲಸಕ್ಕೆ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ತುಂಬಾ ಚೆನ್ನಾಗಿ ವಿವರಣೆ ನೀಡಿದ್ದೀರಿ, ನಮಗೆ ತುಂಬಾ ಸಹಾಯವಾಯಿತು.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಅತ್ಯುತ್ತಮ ವಿಡಿಯೋ! ಕನ್ನಡದಲ್ಲಿ ಇಂತಹ ತಂತ್ರಜ್ಞಾನ ವಿಷಯಗಳು ಬರುತ್ತಿರುವುದು ಹೆಮ್ಮೆಯ ವಿಷಯ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಜೈ ಕರ್ನಾಟಕ ಮಾತೆ! ತುಂಬಾ ಒಳ್ಳೆಯ ಪ್ರಯತ್ನ, ನಿಮ್ಮ ಕೆಲಸ ಮುಂದುವರಿಯಲಿ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಸೂಪರ್ ಮಾಹಿತಿ ಅಣ್ಣ, ಧನ್ಯವಾದಗಳು. ಮುಂದಿನ ಭಾಗಕ್ಕಾಗಿ ಕಾಯುತ್ತಿದ್ದೇನೆ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ನಮಸ್ಕಾರ ಸರ್, ನಿಮ್ಮ ಬೋಧನಾ ಶೈಲಿ ನಮಗೆ ತುಂಬಾ ಇಷ್ಟವಾಯಿತು.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಬೆಂಕಿ ಗುರು! ಅದ್ಭುತವಾದ ಎಡಿಟಿಂಗ್ ಮತ್ತು ಸ್ಪಷ್ಟ ಮಾಹಿತಿ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಹೃತ್ಪೂರ್ವಕ ಧನ್ಯವಾದಗಳು, ಪರೀಕ್ಷೆಗೆ ಈ ವಿಡಿಯೋ ತುಂಬಾ ನೆರವಾಯಿತು.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಸುಂದರವಾದ ಹಾಡು ಮತ್ತು ಸಂಗೀತ, ಕೇಳಲು ತುಂಬಾ ಹಿತವಾಗಿದೆ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ನಿಮ್ಮ ಎಲ್ಲಾ ವಿಡಿಯೋಗಳನ್ನು ನಾನು ನಿಯಮಿತವಾಗಿ ನೋಡುತ್ತೇನೆ, ಗ್ರೇಟ್ ವರ್ಕ್.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಕನ್ನಡ ನಾಡಿನ ಎಲ್ಲಾ ಜನರಿಗೆ ಶುಭಾಶಯಗಳು, ಒಗ್ಗಟ್ಟಿನಲ್ಲಿ ಬಲವಿದೆ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಖುಷಿಯಾಯಿತು ಈ ವಿಷಯ ತಿಳಿದು. ಇನ್ನು ಹೆಚ್ಚಿನ ಮಾಹಿತಿಯನ್ನು ಹಂಚಿಕೊಳ್ಳಿ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಅಭಿನಂದನೆಗಳು ನಿಮ್ಮ 1 ಲಕ್ಷ ಸಬ್ಸ್ಕ್ರೈಬರ್ ಸಾಧನೆಗೆ!", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಶ್ರೇಷ್ಠ ಮಟ್ಟದ ಕೆಲಸ, ಪ್ರತಿಯೊಬ್ಬ ಕನ್ನಡಿಗನೂ ನೋಡಲೇಬೇಕಾದ ವಿಡಿಯೋ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ತುಂಬಾ ಉಪಯುಕ್ತ ಮಾಹಿತಿ ಸರ್, ನನ್ನ ಎಲ್ಲಾ ಸ್ನೇಹಿತರಿಗೂ ಶೇರ್ ಮಾಡಿದ್ದೇನೆ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ದೇವರು ನಿಮಗೆ ಒಳ್ಳೆಯದು ಮಾಡಲಿ, ಸದಾ ಕಾಲ ಸಂತೋಷವಾಗಿರಿ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಧನ್ಯವಾದಗಳು ಗುರುಗಳೇ, ನಿಮ್ಮ ಮಾರ್ಗದರ್ಶನ ಅಮೂಲ್ಯವಾದದ್ದು.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಅತ್ಯುತ್ತಮ ಪ್ರಸ್ತುತಿ, ಯಾವುದೇ ಗೊಂದಲವಿಲ್ಲದೆ ಸ್ಪಷ್ಟವಾಗಿ ಅರ್ಥವಾಯಿತು.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಖಂಡಿತವಾಗಿಯೂ ಇದು ಒಂದು ಅದ್ಭುತ ವಿಡಿಯೋ, ಧನ್ಯವಾದಗಳು.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},

    # -------------------------------------------------------------------------
    # 2. NON-TOXIC: Questions, feedback, discussion (Kannada Script)
    # -------------------------------------------------------------------------
    {"text": "ಮುಂದಿನ ಭಾಗ ಯಾವಾಗ ಅಪ್ಲೋಡ್ ಮಾಡುತ್ತೀರಿ ಸರ್? ದಯವಿಟ್ಟು ತಿಳಿಸಿ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಇದರ ಪಿಡಿಎಫ್ ನೋಟ್ಸ್ ಎಲ್ಲಿ ಸಿಗುತ್ತದೆ? ಡಿಸ್ಕ್ರಿಪ್ಷನ್ ನಲ್ಲಿ ಲಿಂಕ್ ಕೊಡಿ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಧ್ವನಿ ಸ್ವಲ್ಪ ಕಡಿಮೆಯಿದೆ, ಮುಂದಿನ ವಿಡಿಯೋದಲ್ಲಿ ಮೈಕ್ ಸರಿಪಡಿಸಿಕೊಳ್ಳಿ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಪೈಥಾನ್ ಪ್ರೋಗ್ರಾಮಿಂಗ್ ಬಗ್ಗೆ ಕನ್ನಡದಲ್ಲೇ ಪೂರ್ಣ ಟ್ಯುಟೋರಿಯಲ್ ಮಾಡಿ ಪ್ಲೀಸ್.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ನನಗೆ 3ನೇ ಹಂತ ಅರ್ಥವಾಗಲಿಲ್ಲ, ಇನ್ನೊಮ್ಮೆ ವಿವರಣೆ ನೀಡುವಿರಾ?", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಈ ಪುಸ್ತಕ ಮಾರುಕಟ್ಟೆಯಲ್ಲಿ ಲಭ್ಯವಿದೆಯೇ ಅಥವಾ ಆನ್ಲೈನ್ ನಲ್ಲಿ ಕೊಳ್ಳಬೇಕೇ?", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ವಿಡಿಯೋ ಸಮಯ ತುಂಬಾ ಕಡಿಮೆಯಾಯಿತು, ದಯವಿಟ್ಟು ದೀರ್ಘ ವಿಡಿಯೋಗಳನ್ನು ಮಾಡಿ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಬೆಂಗಳೂರಿನಲ್ಲಿ ಈ ತರಬೇತಿ ಕೇಂದ್ರ ಎಲ್ಲಿ ಬರುತ್ತದೆ? ವಿಳಾಸ ತಿಳಿಸಿ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},

    # -------------------------------------------------------------------------
    # 3. NON-TOXIC: Praise & questions in Kanglish (Romanized Kannada)
    # -------------------------------------------------------------------------
    {"text": "tumba chennagide video, dhanyavadagalu sir!", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "thumba helpful content guru, keep it up super explanation", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "adbhuta information, kannada dalli tech videos nodoke thumba kushi aaguthe", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "super video anna, next part yavaga release madteera?", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "benki guru editing matthe background music, enjoyed a lot", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "tumba thanks sir, exam preparation ge tumba use aithu", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "namaskara, nimma channel inda tumba vishaya kalitukondidini", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "jai karnataka mathe, olle prayatna continue maadi", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "great presentation sir, clean and clear concepts", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "kushi aayithu e thara informative video nodi, thanks a lot", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "sound swalpa kammi ide, mic adjust maadi next video dalli please", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "subscribe madidini, daily upload madi videos please", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},

    # -------------------------------------------------------------------------
    # 4. NON-TOXIC: Negation of insults & negative terms (Contextual Non-Toxic)
    # -------------------------------------------------------------------------
    {"text": "ನೀನು ದಡ್ಡ ಅಲ್ಲ, ತುಂಬಾ ಚೆನ್ನಾಗಿ ಯೋಚನೆ ಮಾಡಿದ್ದೀಯ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಈ ವಿಡಿಯೋ ಕೆಟ್ಟದಲ್ಲ, ನನಗೆ ತುಂಬಾ ಇಷ್ಟವಾಯಿತು.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಅವರು ಮೂರ್ಖರಲ್ಲ, ಅವರಿಗೆ ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ಸ್ವಲ್ಪ ಸಮಯ ಬೇಕು.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ನೀವು ಹುಚ್ಚರಲ್ಲ, ನಿಮ್ಮ ಯೋಜನೆ ನಿಜಕ್ಕೂ ಅದ್ಭುತವಾಗಿದೆ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಇದು ಕೆಟ್ಟ ಕೆಲಸವಲ್ಲ, ಒಳ್ಳೆಯ ಕೆಲಸ ಮಾಡಿದ್ದೀರಿ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಯಾರೂ ಇಲ್ಲಿ ಮೂರ್ಖರಲ್ಲ, ಎಲ್ಲರೂ ಕಲಿಯುತ್ತಿದ್ದಾರೆ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "neenu dadda alla, tumba smart aagi solve madidya", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ee video kettadalla, super aagide nodi enjoy maadi", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "neenu huccha alla, confuse aagbeda calmly think madi", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "avaru moorkharalla, olle intentions ide avrige", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},

    # -------------------------------------------------------------------------
    # 5. NON-TOXIC: Self-deprecating or descriptive references
    # -------------------------------------------------------------------------
    {"text": "ನಾನು ದಡ್ಡ, ನನಗೆ ಈ ವಿಷಯ ಮೊದಲೇ ತಿಳಿದಿರಲಿಲ್ಲ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ನನ್ನದೇ ತಪ್ಪು, ಕ್ಷಮಿಸಿ ನಾನು ತಪ್ಪು ವಿಡಿಯೋ ನೋಡಿದೆ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "ಈ ಸಿನಿಮಾದಲ್ಲಿ ಕೊಲೆಗಾರನ ಪಾತ್ರ ತುಂಬಾ ಚೆನ್ನಾಗಿ ಮಾಡಿದ್ದಾರೆ.", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "naanu dadda nanage coding artha aagthilla, need more practice", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    {"text": "nanna thappu bro, sorry for the wrong comment earlier", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},

    # -------------------------------------------------------------------------
    # 6. TOXIC / INSULT: Personal attacks & insults (Kannada Script)
    # -------------------------------------------------------------------------
    {"text": "ನೀನು ಒಬ್ಬ ಮೂರ್ಖ ಮತ್ತು ನಿಷ್ಪ್ರಯೋಜಕ ಕತ್ತೆ.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ಬಾಯಿ ಮುಚ್ಚು ದಡ್ಡ, ನಿನಗೆ ಏನೂ ಗೊತ್ತಿಲ್ಲ ವಿಡಿಯೋ ಮಾಡೋಕೆ ಬರ್ತೀಯ.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ನೀನು ಬೇವರ್ಸಿ, ಕಿತ್ತೋದವನೆ ತೊಲಗು ಇಲ್ಲಿಂದ.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ಅಯೋಗ್ಯ ನೀನು, ನಿನ್ನ ಮುಖ ನೋಡಿದರೆ ವಾಂತಿ ಬರುತ್ತದೆ.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ಹುಚ್ಚನ ತರ ಮಾತಾಡ್ಬೇಡ ಲೋಫರ್, ನಿನಗೆ ಬುದ್ಧಿ ಇಲ್ಲ.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ದರಿದ್ರದವನೆ, ನಿನ್ನ ಚಾನೆಲ್ ಮುಚ್ಚು ಜನರನ್ನು ದಾರಿತಪ್ಪಿಸಬೇಡ.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ತಿಕ್ಕಲು ನಾಯಿ ತರ ಬೊಗಳಬೇಡ, ಮಾನಗೆಟ್ಟವನೆ.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ದಡ್ಡ ರಾಸ್ಕಲ್, ಇಂತಹ ಕೆಟ್ಟ ವಿಡಿಯೋಗಳನ್ನು ಯಾರು ನೋಡಲ್ಲ.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ನೀನು ಒಬ್ಬ ದೊಡ್ಡ ಕೋತಿ, ನಾಲಾಯಕ್ ಕೆಲಸಕ್ಕೆ ಬಾರದವನು.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ಗೂಬೆ ಮುಖದವನೆ, ಮೊದಲು ವಿಷಯ ತಿಳಿದುಕೊಂಡು ಬಂದು ಬೊಗಳು.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ಭಿಕಾರಿ ನಾಯಿ, ಹಣಕ್ಕಾಗಿ ಇಂತಹ ಸುಳ್ಳು ಸುದ್ದಿ ಹರಡಬೇಡ.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ನೀಚ ಮನುಷ್ಯ ನೀನು, ಸಮಾಜಕ್ಕೆ ಕಳಂಕ.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ಚಿಲ್ಲರೆ ಗಿರಾಕಿ, ನಿನ್ನ ಯೋಗ್ಯತೆಗೆ ಇಷ್ಟು ಸಾಕು ತೊಲಗು.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ಲಜ್ಜೆಗೆಟ್ಟವನೆ, ನಾಚಿಕೆ ಮಾನ ಮರ್ಯಾದೆ ಏನೂ ಇಲ್ವಾ ನಿನಗೆ?", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ನೀನು ಸೋಮಾರಿ ದಡ್ಡ, ಸತ್ತು ಹೋಗು ನಾಲಾಯಕ್.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ಬುದ್ಧಿಹೀನ ರಾಸ್ಕಲ್, ಜನರಿಗೆ ಮೋಸ ಮಾಡುವುದನ್ನು ನಿಲ್ಲಿಸು.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},

    # -------------------------------------------------------------------------
    # 7. TOXIC / INSULT: Personal attacks & insults in Kanglish
    # -------------------------------------------------------------------------
    {"text": "neenu obba huccha, kelsa nodkollo baddimaga", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "baayi mucchu dadda katte, enu gothilla ninge", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "bevarsi loafer, kettavne tholagu illinda", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ayogya rascal, stop making videos nobody wants your trash", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "kittodavne ninna mukha nodidre vaanthi baruthe", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "daridra naye thara bogalbeda thiklu", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "goobe face fellow, zero knowledge idiot", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "chillare giraki, ninna yoggyathege ishte saaku", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "shut up you moorkha, wasting our time with nonsense", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "nalayak fellow, get lost and delete channel", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},

    # -------------------------------------------------------------------------
    # 8. TOXIC / OBSCENE: Profanities, vulgar slurs (Kannada Script)
    # -------------------------------------------------------------------------
    {"text": "ಸೂಳೆಮಗನೆ ಬಾಯಿ ಮುಚ್ಚಿಕೊಂಡು ಇರು, ನಿನ್ನ ಯೋಗ್ಯತೆ ಗೊತ್ತು.", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ಬೋಳಿಮಗನೆ ನಿನ್ನ ತಿಕ ಮುಚ್ಕೊಂಡು ಹೋಗೋ ಇಲ್ಲಿಂದ.", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ಹಲ್ಕಟ್ ನಾಯಿ, ಹರಾಮ್ಖೋರ್ ಮಗನೆ ನೀನು.", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ತಿಕ ಮುಚ್ಕೋ ಸೂಳೆ, ಹೆಚ್ಚು ಮಾತಾಡಿದರೆ ಸರಿ ಇರಲ್ಲ.", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ರಾಂಡಿ ಮಗನೆ, ನಿನ್ನ ಜನ್ಮಕ್ಕೆ ಬೆಂಕಿ ಹಾಕ.", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ತುಣ್ಣೆ ಬಾಯ್ ಮುಚ್ಕೋ ಕತ್ತೆ, ಬೋಳಿಮಕ್ಕಳ.", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ಬೇವರ್ಸಿ ಹಲ್ಕಾ, ತಿಕದಲ್ಲಿ ದಮ್ಮಿದ್ರೆ ಎದುರಿಗೆ ಬಾ.", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "ಕಾಮಪಿಶಾಚಿ ಸೂಳೆಮಕ್ಕಳ, ನಿಮ್ಮ ಅಮ್ಮನ ತಿಕ ಮುಚ್ಚಿ.", "toxic": 1, "severe_toxic": 1, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},

    # -------------------------------------------------------------------------
    # 9. TOXIC / OBSCENE: Profanities in Kanglish
    # -------------------------------------------------------------------------
    {"text": "soolemaga thika muchu, dont talk too much", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "bolimaga thika muchkondu hogu illinda", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "halkat haramkhor rascal, nimma amman bolimaga", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "thika mucchu soolemagane, bloody idiot", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "randi maga, bevarsi halka fellow get lost", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
    {"text": "thunne baayi muchko katte bolimakla", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},

    # -------------------------------------------------------------------------
    # 10. TOXIC / THREAT: Violent death & physical threats (Kannada Script)
    # -------------------------------------------------------------------------
    {"text": "ನಿನ್ನನ್ನು ಕೊಲ್ಲುತ್ತೇನೆ ಹುಷಾರ್, ನನ್ನ ಕಣ್ಣಿಗೆ ಬೀಳಬೇಡ.", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
    {"text": "ನಿನ್ನ ಮನೆಗೆ ಬಂದು ಹೊಡೆದು ಸಾಯಿಸ್ತೀನಿ ಮಗನೆ.", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 1, "identity_hate": 0},
    {"text": "ನಿನ್ನ ಜೀವ ತೆಗೀತೀನಿ ಸರ್ವನಾಶ ಮಾಡ್ತೀನಿ ನಿನ್ನ ಕುಟುಂಬವನ್ನು.", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
    {"text": "ನಿನ್ನ ರಕ್ತ ಕುಡೀತೀನಿ, ಮುಗಿಸ್ತೀನಿ ನಿನ್ನ ಕಥೆ.", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
    {"text": "ಕೈಕಾಲು ಮುರೀತೀನಿ, ಜೀವಂತ ಸುಟ್ಟು ಹಾಕ್ತೀನಿ ಹುಷಾರ್.", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
    {"text": "ಸಾಯಿಸುತ್ತೇನೆ ನಿನ್ನನ್ನು, ಈ ಊರಿನಲ್ಲಿ ಜೀವಿಸೋಕೆ ಬಿಡಲ್ಲ.", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
    {"text": "ನಿನ್ನ ಕತ್ತು ಹಿಸುಕಿ ಸಾಯಿಸ್ತೀನಿ ನೀಚ ರಾಸ್ಕಲ್.", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 1, "identity_hate": 0},
    {"text": "ಹಾಳು ಮಾಡ್ತೀನಿ ನಿನ್ನ ಜೀವನ, ಕೊಂದು ಎಸೀತೀನಿ ನಾಯಿ ತರ.", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 1, "identity_hate": 0},

    # -------------------------------------------------------------------------
    # 11. TOXIC / THREAT: Threats in Kanglish
    # -------------------------------------------------------------------------
    {"text": "ninna kolluthene hushar, kan munde barbeda", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
    {"text": "ninna manege bandu hodedu sayistini magane", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 1, "identity_hate": 0},
    {"text": "ninna jeeva thegitheeni, mugisteeni ninna story", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
    {"text": "kaikaalu murithini suttu haaktini ninna", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
    {"text": "i will kill you ninna bidalla, sayisuttene", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},

    # -------------------------------------------------------------------------
    # 12. TOXIC / IDENTITY HATE: Targeted caste, religious or communal slurs
    # -------------------------------------------------------------------------
    {"text": "ನಿಮ್ಮ ಜಾತಿಯವರೇ ಹೀಗೆ, ಕೀಳು ಜಾತಿಯ ಕತ್ತೆಗಳು ನೀವು.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 1},
    {"text": "ಈ ಧರ್ಮದವರೆಲ್ಲ ಭಯೋತ್ಪಾದಕರು, ಈ ದೇಶದಿಂದ ಓಡಿಸಿ ಇವರನ್ನು.", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 1, "identity_hate": 1},
    {"text": "ಹೆಣ್ಣು ಮಕ್ಕಳಿಗೆ ಬುದ್ಧಿ ಇರಲ್ಲ, ಮನೆಯಲ್ಲಿ ಕುಳಿತು ಅಡುಗೆ ಮಾಡು.", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 1},
    {"text": "nimma jathiyavaru ella moorkharu, hate all of you", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 1},
    {"text": "ee dharmada janarannu naasha maadbeku", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 1}
]


def expand_dataset(base_data: list) -> pd.DataFrame:
    """Expand dataset with natural contextual variations and augmentations."""
    expanded = []
    
    # Prefix and suffix variations for natural conversational flow
    prefixes_clean = ["", "ಹಾಯ್ ", "ಹಲೋ ", "ಸರ್, ", "ನಮಸ್ಕಾರ, ", "ಗುರು, ", "hi bro ", "hello sir "]
    suffixes_clean = ["", "!", " ಧನ್ಯವಾದಗಳು.", " ಸೂಪರ್.", " keep it up.", " ಬೆಂಕಿ.", " jai karnataka."]
    
    prefixes_toxic = ["", "ಹೇಯ್ ", "ಲೋ ", "ಏಯ್ ", "hey ", "lo "]
    suffixes_toxic = ["", "!", " ತೊಲಗು.", " get lost.", " ಮುಚ್ಕೋ.", " ಮುಗ್ಸಿತೀನಿ."]

    for item in base_data:
        text = item["text"]
        is_toxic = item["toxic"] == 1
        
        # Original
        expanded.append(dict(item))
        
        # Add slight variations for training robustness
        if is_toxic:
            for p in prefixes_toxic[:3]:
                for s in suffixes_toxic[:2]:
                    if p or s:
                        new_item = dict(item)
                        new_item["text"] = f"{p}{text}{s}".strip()
                        expanded.append(new_item)
        else:
            for p in prefixes_clean[:3]:
                for s in suffixes_clean[:2]:
                    if p or s:
                        new_item = dict(item)
                        new_item["text"] = f"{p}{text}{s}".strip()
                        expanded.append(new_item)

    df = pd.DataFrame(expanded).drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


def train_kannada_model(out_dir: str = "models"):
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 65)
    print("Training Kannada Toxic Comment Classification Model")
    print("=" * 65)

    df = expand_dataset(KANNADA_TRAINING_DATA)
    print(f"Generated expanded Kannada training dataset: {len(df)} samples")
    
    # Save training dataset to data/
    os.makedirs("data", exist_ok=True)
    data_csv_path = os.path.join("data", "kannada_comments_dataset.csv")
    df.to_csv(data_csv_path, index=False, encoding="utf-8")
    print(f"Saved Kannada dataset to {data_csv_path}")

    # Label statistics
    for label in DEFAULT_LABELS:
        pos = int(df[label].sum())
        print(f"  {label:<15}: {pos:3d} positive samples ({pos/len(df)*100:.1f}%)")

    # Fit TF-IDF Vectorizer with sublinear TF, word (1-2) + char_wb (2-5) n-grams
    # Character n-grams are critical for Kannada morphological suffixes and transliteration variants
    print("\n[1/3] Fitting Kannada TF-IDF Vectorizer ...")
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        ngram_range=(1, 3),
        analyzer="char_wb",
        min_df=1,
        max_features=25000
    )
    X = vectorizer.fit_transform(df["text"].apply(clean_text))
    print(f"TF-IDF feature matrix shape: {X.shape}")

    # Train One-vs-Rest Logistic Regression classifiers for each label
    print("\n[2/3] Training binary classifiers for all 6 toxicity categories ...")
    classifiers = {}
    optimal_thresholds = {
        "toxic": 0.35,
        "severe_toxic": 0.25,
        "obscene": 0.30,
        "threat": 0.25,
        "insult": 0.32,
        "identity_hate": 0.25
    }

    for label in DEFAULT_LABELS:
        y = df[label].values
        # Use balanced class weighting to ensure rare toxic labels (threat, severe_toxic) are well detected
        clf = LogisticRegression(
            C=3.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        )
        clf.fit(X, y)
        classifiers[label] = clf
        
        # Self-eval F1
        preds = (clf.predict_proba(X)[:, 1] >= optimal_thresholds[label]).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        print(f"  Classifier '{label:<14}' trained | F1: {f1:.4f}")

    # Save artifacts
    print("\n[3/3] Saving model artifacts ...")
    vec_path = os.path.join(out_dir, "kannada_tfidf_vectorizer.pkl")
    model_path = os.path.join(out_dir, "kannada_multilabel_model.pkl")
    meta_path = os.path.join(out_dir, "kannada_model_metadata.json")

    joblib.dump(vectorizer, vec_path)
    joblib.dump(classifiers, model_path)

    metadata = {
        "model_name": "kannada_multilabel_toxicity_model",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "language": "Kannada (ಕನ್ನಡ) & Kanglish (Romanized)",
        "train_samples": len(df),
        "labels": DEFAULT_LABELS,
        "optimal_thresholds": optimal_thresholds,
        "feature_count": X.shape[1],
        "vectorizer_params": {
            "analyzer": "char_wb",
            "ngram_range": [1, 3],
            "sublinear_tf": True
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts successfully saved:")
    print(f"  - {vec_path}")
    print(f"  - {model_path}")
    print(f"  - {meta_path}")
    print("Kannada model training complete!")


if __name__ == "__main__":
    train_kannada_model()
