# YouTube Toxic Comment Analyzer

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.15-blue.svg)](https://www.python.org/)
[![Framework: Flask](https://img.shields.io/badge/Framework-Flask%203.1-black.svg)](https://flask.palletsprojects.com/)
[![ML Library: Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%201.9-orange.svg)](https://scikit-learn.org/)
[![API: YouTube Data v3](https://img.shields.io/badge/API-YouTube%20Data%20v3-red.svg)](https://developers.google.com/youtube/v3)
[![Tests: Pytest](https://img.shields.io/badge/Tests-89%20Passed-brightgreen.svg)](tests/)
[![Semantics: Emoji & Sarcasm](https://img.shields.io/badge/Semantics-Emoji%20%26%20Sarcasm%20Aware-gold.svg)](src/emoji_analyzer.py)
[![Languages: English & Kannada](https://img.shields.io/badge/Languages-English%20%7C%20ಕನ್ನಡ%20(Kannada)-purple.svg)](src/kannada_analyzer.py)

A production-grade, single-page web application that analyzes user comments on any public YouTube video to classify them as **Toxic** or **Non-Toxic**. Powered by offline-trained multi-label machine learning models supporting both **English and Kannada (ಕನ್ನಡ Script, Romanized Kanglish, and Code-Mixed)** comments, enhanced with a comprehensive **Emoji & Sarcasm Intelligence Engine**, calibrated decision thresholds, semantic sentence contextual NLP, real-time **Baseline vs. Upgraded Model Comparison**, and official YouTube Data API v3 integration.

---

## 1. Project Overview & Workflow

1. **User Enters Video Link**: Paste any valid YouTube video URL (standard, shortened `youtu.be`, mobile, embed, or shorts link) or evaluate using the built-in demo video dataset.
2. **Emoji Preservation & Preprocessing**: Unicode emojis are preserved through placeholder tokenization during text cleaning, preventing emoji-only comments from collapsing into empty strings.
3. **Language Detection & Processing**: Comments are automatically identified as **English**, **Kannada (ಕನ್ನಡ)**, **Kanglish (Romanized)**, or **Mixed (ಕನ್ನಡ / English)**.
4. **Emoji Semantics & Contextual Analysis**: Evaluates hostile gestures, violent threats, sarcasm inversion (superficial praise + mocking emojis / slow-clap on failure), compound multipliers (`🔪🩸`, `💀⚰️`), and cultural Kannada idiom alignment (`ಬೆಂಕಿ` + `🔥` -> praise, `ಧನ್ಯವಾದಗಳು` + `🙏` -> respect).
5. **Context-Aware Multi-Label Toxicity Classification**: Classifies comments across all 6 toxicity labels (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) and maps them into 4 calibrated severity levels (`Clean`, `Low`, `Medium`, `High`).
6. **Exact Metric Calculation & Summary**: Computes total comments analyzed, toxic count, non-toxic count, exact toxicity percentage, language distribution, and a dedicated **Emoji Signals Strip** highlighting emoji density, toxic emoji clusters, and sarcasm counts.
7. **Interactive Visual Feed**: Browse analyzed comments with color-coded classification badges, severity badges, and visual emoji chips (`[🔪 +90% Threat]`, `[🤡 +85% Insult]`, `[🔥 Praise]`). Filter comments by All, Toxic, Non-Toxic, ಕನ್ನಡ, or Emojis.

---

## 2. Key Features

- **Emoji & Sarcasm Intelligence Engine**: Detects toxicity expressed solely or jointly through emojis, distinguishing hostile gestures (`🖕`, `🖕🏻-🏿`), violent threats (`🔪`, `🩸`, `💣`, `💀`), mocking derision (`🤡`, `🤮`, `💩`, `🙄`), and neutral symbols (`📄`, `🕒`, `❓`).
- **Sarcasm Inversion & Mockery Detection**: Flags sarcastic comments where superficial praise words ("genius", "brilliant", "great job") are paired with mocking emojis (`🤡`, `🙃`, `🗑️`, `🤮`, `💩`) or sarcastic slow-claps on failure (`👏`, `😂`), accurately converting deceptive positive sentiment into `toxic` and `insult` flags.
- **Dual-Language & Cultural Integration (English & Kannada)**: Native Kannada Unicode script (ಕನ್ನಡ ಲಿಪಿ), Romanized Kanglish, and code-mixed comments are integrated with cultural emoji rules (e.g., `ಬೆಂಕಿ ಗುರು 🔥` treated as non-toxic high praise, `ನೀನು ದಡ್ಡ ಅಲ್ಲ 😊` verified as friendly negated insult).
- **All Six Toxicity Labels & Four Severity Levels Preserved**: Evaluates `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`, and groups toxicity into 4 standardized severity tiers: `Clean`, `Low`, `Medium`, and `High`.
- **Single-Page Focused Architecture**: Streamlined single-page UI dedicated entirely to YouTube video toxicity analysis with live preview, metrics dashboard, filterable feed, and real-time emoji intelligence.
- **Official YouTube Data API Integration**: Multi-page pagination up to 5,000 comments plus nested reply thread extraction.
- **Built-in Bilingual Demo Mode**: Includes 26 rich English and Kannada demo comments (with plain-text toxic, plain-text clean, emoji-only toxic, emoji-only clean, mixed toxic, and mixed clean) for instant testing without requiring an API key.

---

## 3. Technology Stack

- **Machine Learning & Preprocessing**: `scikit-learn 1.9`, `emoji 2.14`, `pandas 3.0`, `numpy 2.5`, `scipy 1.18`, `joblib 1.6`
- **Backend Web Server**: `Flask 3.1` (REST API & static file serving)
- **YouTube API Client**: `google-api-python-client 2.201`
- **Testing Suite**: `pytest 9.1` (89 automated unit & integration tests)
- **Frontend**: Vanilla HTML5, Vanilla CSS3 (custom dark glassmorphic design system), Vanilla ES6 JavaScript


---

## 4. Datasets & Model Training Pipeline

This project employs two specialized multi-label toxicity classification pipelines: one for **English** comments (trained on the benchmark Jigsaw challenge dataset) and one for **Kannada & Kanglish** comments.

### English Dataset: Jigsaw Toxic Comment Classification Challenge

- **Dataset Name**: Jigsaw Toxic Comment Classification Challenge Dataset (Wikipedia Talk Page Comments)
- **Source / Publisher**: Conversation AI team (founded by Google and Jigsaw) on Kaggle
- **Direct Dataset URLs**:
  - **Official Kaggle Competition & Dataset URL**:  
    [https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
  - **Direct Hugging Face Repository URL**:  
    [https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments](https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments)
  - **Direct Download URL for `train.csv.gz`** (159,571 comments):  
    [https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments/resolve/main/train.csv.gz](https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments/resolve/main/train.csv.gz)
  - **Direct Download URL for `test.csv.gz`** (153,164 comments):  
    [https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments/resolve/main/test.csv.gz](https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments/resolve/main/test.csv.gz)
  - **Direct Download URL for `test_labels.csv.gz`** (63,613 evaluated test labels):  
    [https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments/resolve/main/test_labels.csv.gz](https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments/resolve/main/test_labels.csv.gz)
  - **Google Research Documentation Reference URL**:  
    [https://huggingface.co/datasets/google/jigsaw_toxicity_pred](https://huggingface.co/datasets/google/jigsaw_toxicity_pred)

### Kannada Dataset: Kannada Multi-Label Toxic Comment Dataset

- **Dataset Name**: Kannada Multi-Label Toxic Comment Dataset
- **Source / Publisher**: Curated balanced dataset covering native Kannada Unicode script (ಕನ್ನಡ ಲಿಪಿ) and Romanized Kannada (Kanglish) across all 6 toxicity classes (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).
- **Dataset File Location**: [`data/kannada_comments_dataset.csv`](file:///c:/toxic_comment/data/kannada_comments_dataset.csv)
- **Dataset Generation & Training Script**: [`scripts/train_kannada_model.py`](file:///c:/toxic_comment/scripts/train_kannada_model.py)

### Reproducible Data & Model Commands

```powershell
# 1. Download verified Jigsaw challenge dataset (train.csv, test.csv, test_labels.csv)
python scripts/download_data.py

# 2. Preprocess, clean text, and generate 85%/15% train/val splits and test set
python scripts/prepare_data.py

# 3. Train English multi-label model with calibrated decision thresholds
python src/train.py

# 4. Evaluate on official held-out test set (63,613 comments)
python src/evaluate.py

# 5. Train Kannada & Kanglish multi-label model
python scripts/train_kannada_model.py
```

### Empirical Held-Out Test Set Performance (63,613 Comments)

| Toxicity Label | Decision Cutoff | Precision | Recall | $F_1$-Score | ROC-AUC | Test Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`toxic`** | 0.28 | 54.71% | 81.22% | **0.6538** | **0.9572** | 6,090 |
| **`severe_toxic`** | 0.12 | 24.00% | 65.40% | **0.3511** | **0.9805** | 367 |
| **`obscene`** | 0.20 | 60.73% | 77.30% | **0.6802** | **0.9707** | 3,691 |
| **`threat`** | 0.10 | 40.85% | 59.24% | **0.4836** | **0.9919** | 211 |
| **`insult`** | 0.24 | 59.94% | 70.29% | **0.6471** | **0.9663** | 3,427 |
| **`identity_hate`** | 0.16 | 51.59% | 57.02% | **0.5417** | **0.9766** | 712 |

- **Macro ROC-AUC**: **0.9739**
- **Micro $F_1$-Score**: **0.6393**
- **Exact Match Ratio**: **86.88%**
- **Hamming Loss**: **0.0325**

---

## 5. Emoji & Sarcasm Intelligence Architecture

Traditional NLP text pipelines frequently fail on modern social media comments because standard tokenizers (such as `scikit-learn`'s `CountVectorizer` and `TfidfVectorizer` with standard `token_pattern: (?u)\b\w\w+\b`) strip or ignore non-word Unicode characters. As a result:
- **Emoji-only insults and threats** produce all-zero feature vectors, yielding low baseline probabilities (~0.04) and severe **false negatives**.
- **Sarcastic comments** containing superficial praise paired with derisive emojis (e.g. *"You are a genius 🤡"*) fool text models into predicting high non-toxicity.
- **Cultural idioms** (such as Kannada *"ಬೆಂಕಿ ಗುರು 🔥"* meaning *"fire / absolute masterpiece"*) require contextual knowledge so that fire emojis are recognized as praise rather than hazard.

```
                           Raw YouTube Comment
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │ Language & Script Detection │
                     │  (English, Kannada, Mixed)  │
                     └──────────────┬──────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌───────────────────────────────┐              ┌───────────────────────────────┐
│     Text Preprocessing        │              │    Emoji Semantics Engine     │
│ - Placeholder Token Guard     │              │ - Demojization & Lexicon Map  │
│ - Negation & Scoping Tags     │              │ - Compound Pairs (🔪🩸, 💀⚰️) │
│ - Dual TF-IDF / Char n-grams  │              │ - Sarcasm & Mockery Inversion │
└──────────────┬────────────────┘              └───────────────┬───────────────┘
               │                                               │
               ▼                                               ▼
┌───────────────────────────────┐              ┌───────────────────────────────┐
│ Base Multi-Label ML Inference │              │   Cultural & Kannada Rules    │
│ (One-vs-Rest LogReg/SGD)      │              │ (ಬೆಂಕಿ+🔥, ಧನ್ಯವಾದಗಳು+🙏)     │
└──────────────┬────────────────┘              └───────────────┬───────────────┘
               │                                               │
               └───────────────────────┬───────────────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ Contextual Fusion & Overrides │
                       │ - 6 Toxicity Labels Preserved │
                       │ - 4 Calibrated Severity Tiers │
                       │ - Probability Deltas Computed │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │     Unified Output & UI       │
                       │ - Side-by-Side Model Compare  │
                       │ - Emoji Semantic Chips & Feed │
                       │ - Video-Level Emoji Metrics   │
                       └───────────────────────────────┘
```

### Preprocessing & Token Preservation
- **Placeholder Protection**: [`clean_text()`](file:///c:/toxic_comment/src/preprocessing.py) uses `preserve_emojis=True` by default, replacing Unicode emojis with collision-free token anchors (`__EMOJI_PH_n__`) before sanitizing HTML entities, URLs, and punctuation, then restoring them seamlessly.
- **Helper Utilities**: Provides [`extract_emojis()`](file:///c:/toxic_comment/src/preprocessing.py), [`has_emojis()`](file:///c:/toxic_comment/src/preprocessing.py), and [`is_emoji_only()`](file:///c:/toxic_comment/src/preprocessing.py).

### Multi-Label Emoji Lexicon & Compound Rules
[`src/emoji_analyzer.py`](file:///c:/toxic_comment/src/emoji_analyzer.py) maintains a verified multi-label taxonomy mapping emojis to calibrated probabilities across all 6 categories:
- **Hostile & Obscene Gestures**: `🖕`, `🖕🏻-🏿` (Middle Finger -> `obscene`: 0.95, `toxic`: 0.94, `insult`: 0.88), `👊`, `👎`.
- **Violent Threat Imagery**: `🔪` (Knife -> `threat`: 0.90, `toxic`: 0.92), `🩸` (Blood -> `threat`: 0.75), `💣` (Bomb -> `threat`: 0.85), `🔫` (Gun -> `threat`: 0.85).
- **Compound Threat Multipliers**: Sequences such as `🔪🩸` (knife + blood) or `💀⚰️` (skull + coffin) trigger elevated compound multipliers (`threat`: 0.95, `severe_toxic`: 0.88).
- **Sarcastic Mockery & Derision**: `🤡` (Clown -> `insult`: 0.85, `toxic`: 0.82), `🤮` (Vomit -> `toxic`: 0.78, `insult`: 0.72), `💩` (Poop -> `toxic`: 0.75), `🙄`, `🥱`, `🙃`.
- **Mitigating Praise & Gratitude**: `❤️`, `💖`, `🔥`, `👍`, `🙏`, `👏`, `✨`, `💯` act as positive goodwill markers that mitigate borderline ambiguity when no directed hostility exists.
- **Neutral Symbols**: Informational emojis (`📄`, `🕒`, `📅`, `❓`, `🔍`) are explicitly categorized as non-toxic neutral markers.

### Sarcasm Inversion Engine
The analyzer evaluates syntactic dissonance between praise tokens and derisive emojis:
1. **Superficial Praise + Mocking Emoji**: Words like *"genius"*, *"brilliant"*, *"masterpiece"*, *"great job"* paired with `🤡`, `🙃`, `🗑️`, `🤮`, `💩`, `🙄` trigger automatic sarcasm inversion, raising `toxic` and `insult` probabilities.
2. **Sarcastic Slow-Clap on Failure**: Praise phrases combined with negative outcomes (*"failing"*, *"broke"*, *"ruined"*, *"mess"*) and `👏`, `😂`, `🤣` trigger sarcastic mockery detection.

### Kannada & Kanglish Cultural Integration
- Native Kannada script (ಕನ್ನಡ) and Kanglish phrases are paired with contextual emoji rules:
  - `"ಬೆಂಕಿ"` / `"benki"` + `🔥` is recognized as high praise and kept strictly **Non-Toxic (Clean)**.
  - `"ಧನ್ಯವಾದಗಳು"` / `"namaskara"` + `🙏` is treated as deep respect and kept strictly **Non-Toxic (Clean)**.
  - Negated Kannada insults with friendly emojis (e.g., `"ನೀನು ದಡ್ಡ ಅಲ್ಲ 😊"` / `"neenu dadda alla 😊"`) are verified as non-hostile.
  - Insults in Kannada/Kanglish with derisive emojis (e.g., `"ಯಾವನ್ ಗುರು ಇವನು 🤡"` / `"Yeno clown tara madthya 🤡"`) are escalated to **Toxic (Medium/High)**.

### Preserved 6 Toxicity Labels & 4 Severity Levels
The upgrade maintains 100% architectural backward compatibility:
- **All 6 Labels**: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.
- **All 4 Severity Tiers**:
  - `Clean` (0 flagged categories, max probability < threshold)
  - `Low` (1 mild category flagged, e.g. borderline toxic)
  - `Medium` (2 categories flagged or moderate insults/sarcasm)
  - `High` (3+ categories flagged, or acute `threat` / `severe_toxic` / multiple hostile slurs)

---

## 6. Baseline vs. Upgraded Model Comparison Engine

To validate and demonstrate the upgrade, the engine includes an empirical model comparison framework ([`compare_predictions`](file:///c:/toxic_comment/src/emoji_analyzer.py#L790-L865)) evaluating the baseline text-only model against the upgraded text+emoji model across different comment categories.

### Empirical Comparison Matrix

The table below summarizes test results across emoji-only, text-plus-emoji, neutral, sarcastic, and Kannada comments:

| Test Comment | Baseline Text Model Verdict | Upgraded Model Verdict | Severity | Flags | Key Probability Deltas ($\Delta$) | Impact & Resolution |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| `🖕` | Non-Toxic (0.04) | **Toxic** | **High** | `toxic`, `obscene`, `insult` | `obscene`: **+91.0%**, `toxic`: **+90.0%** | **False Negative Recovered** (Hostile gesture) |
| `🔪🩸💀` | Non-Toxic (0.04) | **Toxic** | **High** | `toxic`, `severe_toxic`, `threat` | `threat`: **+94.8%**, `severe_toxic`: **+86.5%** | **False Negative Recovered** (Compound lethal threat) |
| `❤️✨🙏` | Non-Toxic (0.04) | **Non-Toxic** | **Clean** | *None* | `toxic`: 0.0%, `threat`: 0.0% | **Accurate Non-Toxic** (Positive praise) |
| `Watch your back 🔪🩸` | Non-Toxic (0.05) | **Toxic** | **High** | `toxic`, `severe_toxic`, `threat` | `threat`: **+94.8%**, `toxic`: **+87.5%** | **False Negative Recovered** (Text + weapon/blood emojis) |
| `You are an idiot 🤡` | Toxic (0.68) | **Toxic** | **High** | `toxic`, `severe_toxic`, `insult` | `insult`: **+25.0%**, `toxic`: **+20.0%** | **Severity Calibrated** (Text insult escalated by clown emoji) |
| `Please review page 42 📄` | Non-Toxic (0.02) | **Non-Toxic** | **Clean** | *None* | `toxic`: 0.0%, `insult`: 0.0% | **Consistent Neutral** (Informational symbol) |
| `Meeting starts at 3 PM 🕒` | Non-Toxic (0.02) | **Non-Toxic** | **Clean** | *None* | `toxic`: 0.0%, `insult`: 0.0% | **Consistent Neutral** (Time indicator) |
| `You are such a genius 🤡` | Non-Toxic (0.03) | **Toxic** | **Medium** | `toxic`, `insult` | `insult`: **+62.5%**, `toxic`: **+57.8%** | **False Negative Recovered** (Sarcastic praise inversion) |
| `Great job failing the test 👏😂` | Non-Toxic (0.04) | **Toxic** | **Medium** | `toxic`, `insult` | `insult`: **+60.0%**, `toxic`: **+55.0%** | **False Negative Recovered** (Mocking slow-clap on failure) |
| `Super video guru benki 🔥` | Non-Toxic (0.01) | **Non-Toxic** | **Clean** | *None* | `toxic`: 0.0%, `threat`: 0.0% | **Consistent Non-Toxic** (Kannada cultural praise preserved) |
| `ಧನ್ಯವಾದಗಳು ನಿಮ್ಮ ಸಹಾಯಕ್ಕೆ 🙏` | Non-Toxic (0.01) | **Non-Toxic** | **Clean** | *None* | `toxic`: 0.0%, `threat`: 0.0% | **Consistent Non-Toxic** (Kannada reverence preserved) |
| `Neenu dadda alla 😊` | Non-Toxic (0.05) | **Non-Toxic** | **Clean** | *None* | `toxic`: 0.0%, `insult`: 0.0% | **Consistent Non-Toxic** (Kanglish negated insult + smile) |
| `Yeno clown tara madthya 🤡` | Non-Toxic (0.12) | **Toxic** | **Medium** | `toxic`, `insult` | `insult`: **+68.0%**, `toxic`: **+64.0%** | **False Negative Recovered** (Kanglish clown derision) |

---

## 7. YouTube Data API Key Setup

To fetch live comments from any YouTube video:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (e.g., `YouTube-Toxic-Detection`).
3. Navigate to **APIs & Services** > **Library** and search for **YouTube Data API v3**. Click **Enable**.
4. Navigate to **APIs & Services** > **Credentials** and click **+ Create Credentials** > **API Key**.
5. Copy your API key.

### Configuring the API Key

You can configure your key in two ways:

**Option A (Recommended: Environment Variable):**
```powershell
# Set in PowerShell
$env:YOUTUBE_API_KEY="YOUR_ACTUAL_API_KEY"
```
Or create a `.env` file in the project root:
```ini
YOUTUBE_API_KEY=YOUR_ACTUAL_API_KEY
```

**Option B (In UI):**
Click the **Custom API Key** button in the web application search bar and paste your key.

---

## 8. Installation & Startup (Windows Environment)

### Step 1: Open PowerShell in Workspace
```powershell
cd c:\toxic_comment
```

### Step 2: Create & Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Requirements
```powershell
pip install -r requirements.txt
```

### Step 4: Run Automated Tests
```powershell
pytest tests/ -v
```
*(All 86 tests covering URL extraction, calculations, batch inference, Kannada analysis, emoji preservation, compound multipliers, sarcasm inversion, 6 labels, 4 severities, and model comparisons will run and pass).*

### Step 5: Launch the Web Application
```powershell
python web/app.py
```
Open your web browser and navigate to:  
👉 **`http://127.0.0.1:5000`**

---

## 9. API Quotas & Limitations

- **Daily Quota**: Google provides a free daily default quota of 10,000 units for YouTube Data API v3. Each `commentThreads.list` request consumes 1 unit.
- **Comment Volume & Pagination**: The application supports depth presets from 100 up to 2,000 comments (MAX / all accessible) with thread reply extraction, while keeping API quota usage low (1 quota unit per 100 comment threads).
- **Disabled Comments**: If a creator has disabled comments on their video, the API returns a `commentsDisabled` response, which is caught and displayed with a clear message.
- **Subsets Analyzed**: Comment percentages describe the sample retrieved and analyzed up to the configured limit, not every comment ever submitted on high-volume videos with millions of replies.

---

## 10. Project Structure

```
Toxic-Comment/
├── data/
│   ├── raw/                         # Downloaded Jigsaw datasets
│   ├── processed/                   # Cleaned splits and test data
│   └── kannada_comments_dataset.csv # Curated Kannada & Kanglish dataset
├── models/
│   ├── tfidf_vectorizer.pkl         # Serialized English TF-IDF feature extractor
│   ├── multilabel_toxicity_model.pkl # 6 trained One-vs-Rest English classifiers
│   ├── model_metadata.json          # Calibrated threshold config & parameters
│   ├── kannada_tfidf_vectorizer.pkl # Serialized Kannada char-ngram vectorizer
│   ├── kannada_multilabel_model.pkl # 6 trained One-vs-Rest Kannada classifiers
│   ├── kannada_model_metadata.json  # Kannada model thresholds & metadata
│   └── evaluation_metrics.json      # Official held-out test evaluation results
├── src/
│   ├── __init__.py
│   ├── preprocessing.py             # Emoji-preserving text sanitization & validation
│   ├── emoji_analyzer.py            # Emoji semantics, sarcasm inversion, & model comparison
│   ├── youtube_analyzer.py          # YouTube URL parsing, API fetch, & batch emoji metrics
│   ├── predict.py                   # Multi-label bilingual routing & emoji integration
│   ├── kannada_analyzer.py          # Kannada language detection & context NLP engine
│   ├── context_analyzer.py          # English sentence-level contextual NLP
│   ├── train.py                     # English model training & threshold tuning
│   └── evaluate.py                  # Held-out benchmark evaluation
├── scripts/
│   ├── download_data.py             # Automated Jigsaw dataset acquisition
│   ├── prepare_data.py              # Dataset cleaning & split preparation
│   └── train_kannada_model.py       # Kannada model training pipeline
├── web/
│   ├── app.py                       # Flask backend & REST API
│   ├── static/
│   │   ├── css/style.css            # Dark glassmorphic design with emoji chips & visual signals strip
│   │   └── js/app.js                # Client-side SPA controller
│   └── templates/
│       └── index.html               # Unified single-page YouTube toxic comment analyzer interface
├── tests/
│   ├── test_emoji_analysis.py       # Comprehensive unit & integration tests for emoji engine
│   ├── test_youtube_analyzer.py     # Unit & integration tests for YouTube analyzer
│   ├── test_preprocessing.py        # Unit tests for emoji-preserving text cleaning
│   ├── test_model.py                # Model inference tests
│   ├── test_context_toxicity.py     # Contextual linguistic tests
│   ├── test_kannada_analysis.py     # Kannada NLP and model tests
│   └── test_api.py                  # Web server endpoint tests
├── .env.example                     # Environment variable template
├── requirements.txt                 # Documented dependencies (including emoji>=2.10.0)
├── .gitignore                       # Git ignore rules
└── README.md                        # Comprehensive documentation
```

