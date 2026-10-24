# YouTube Toxic Comment Analyzer

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.15-blue.svg)](https://www.python.org/)
[![Framework: Flask](https://img.shields.io/badge/Framework-Flask%203.1-black.svg)](https://flask.palletsprojects.com/)
[![ML Library: Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%201.9-orange.svg)](https://scikit-learn.org/)
[![API: YouTube Data v3](https://img.shields.io/badge/API-YouTube%20Data%20v3-red.svg)](https://developers.google.com/youtube/v3)
[![Tests: Pytest](https://img.shields.io/badge/Tests-58%20Passed-brightgreen.svg)](tests/)
[![Languages: English & Kannada](https://img.shields.io/badge/Languages-English%20%7C%20ಕನ್ನಡ%20(Kannada)-purple.svg)](src/kannada_analyzer.py)

A focused, professional, single-page web application that analyzes user comments on any public YouTube video to classify them as **Toxic** or **Non-Toxic**. Powered by offline-trained multi-label machine learning models supporting both **English and Kannada (ಕನ್ನಡ Script, Romanized Kanglish, and Code-Mixed)** comments with calibrated decision thresholds, semantic sentence contextual NLP, and official YouTube Data API v3 integration.

---

## 1. Project Overview & Workflow

1. **User Enters Video Link**: Paste any valid YouTube video URL (standard, shortened `youtu.be`, mobile, embed, or shorts link).
2. **Language Detection & Processing**: Comments are automatically identified as **English**, **Kannada (ಕನ್ನಡ)**, **Kanglish (Romanized)**, or **Mixed (ಕನ್ನಡ / English)**.
3. **Official Comment Ingestion**: Accessible top-level comments are fetched via official YouTube Data API v3 (or built-in bilingual demo dataset).
4. **Context-Aware Toxicity Classification**: Evaluates comments across 6 multi-label categories (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) with negation scoping and idiom analysis for both English and Kannada.
5. **Exact Metric Calculation & Summary**: Computes total comments analyzed, toxic count, non-toxic count, exact toxicity percentage, non-toxicity percentage, language distribution, and visual split ratios.
6. **Interactive Feed**: Browse analyzed comments with color-coded classification badges, language pills (`[ಕನ್ನಡ]`, `[EN]`), and filter between **All**, **Toxic**, **Non-Toxic**, and **ಕನ್ನಡ (Kannada)**.

---

## 2. Key Features

- **Dual-Language Support (English & Kannada)**: Full toxicity analysis for both English and Kannada comments, covering native Kannada Unicode script (ಕನ್ನಡ ಲಿಪಿ), Romanized Kannada (Kanglish), and code-mixed comments.
- **Deep Semantic Contextual Engine**: Recognizes Kannada negation (e.g., 'ನೀನು ದಡ್ಡ ಅಲ್ಲ' / 'neenu dadda alla' -> Non-Toxic), colloquial praise (e.g., 'ಬೆಂಕಿ ಗುರು', 'ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ' -> Non-Toxic), self-deprecation, and targeted attacks/threats.
- **Single-Page Focused Architecture**: Responsive single-page application with video player preview, metrics dashboard, and filterable feed.
- **Official YouTube Data API Integration**: Fetches comments using Google's official API client with multi-page pagination up to 5,000 comments plus nested reply thread extraction.
- **Accurate Percentage Calculations**: Safely computes exact proportions with zero-division protection.
- **Built-in Bilingual Demo Mode**: Includes 16 realistic English and Kannada demo comments for immediate evaluation without requiring an API key.
- **Secure Key Management**: Credentials stay on the backend; never exposed in client bundles.
- **Graceful Error Handling**: Handles disabled comments, missing videos, quota limits, and invalid URLs.

---

## 3. Technology Stack

- **Machine Learning & Preprocessing**: `scikit-learn 1.9`, `pandas 3.0`, `numpy 2.5`, `scipy 1.18`, `joblib 1.6`
- **Backend Web Server**: `Flask 3.1` (REST API & static file serving)
- **YouTube API Client**: `google-api-python-client 2.201`
- **Testing Suite**: `pytest 9.1` (58 automated unit & integration tests)
- **Frontend**: Vanilla HTML5, Vanilla CSS3, Vanilla ES6 JavaScript

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

## 5. YouTube Data API Key Setup

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

## 6. Installation & Startup (Windows Environment)

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
*(All 58 tests covering URL extraction, calculations, batch inference, Kannada analysis, and API endpoints will run and pass).*

### Step 5: Launch the Web Application
```powershell
python web/app.py
```
Open your web browser and navigate to:  
👉 **`http://127.0.0.1:5000`**

---

## 7. API Quotas & Limitations

- **Daily Quota**: Google provides a free daily default quota of 10,000 units for YouTube Data API v3. Each `commentThreads.list` request consumes 1 unit.
- **Comment Volume & Pagination**: The application supports depth presets from 100 up to 2,000 comments (MAX / all accessible) with thread reply extraction, while keeping API quota usage low (1 quota unit per 100 comment threads).
- **Disabled Comments**: If a creator has disabled comments on their video, the API returns a `commentsDisabled` response, which is caught and displayed with a clear message.
- **Subsets Analyzed**: Comment percentages describe the sample retrieved and analyzed up to the configured limit, not every comment ever submitted on high-volume videos with millions of replies.

---

## 8. Project Structure

```
Toxic-Comment/
├── data/
│   ├── raw/                       # Downloaded Jigsaw datasets
│   ├── processed/                 # Cleaned splits and test data
│   └── kannada_comments_dataset.csv # Curated Kannada & Kanglish dataset
├── models/
│   ├── tfidf_vectorizer.pkl       # Serialized English TF-IDF feature extractor
│   ├── multilabel_toxicity_model.pkl # 6 trained One-vs-Rest English classifiers
│   ├── model_metadata.json        # Calibrated threshold config & parameters
│   ├── kannada_tfidf_vectorizer.pkl # Serialized Kannada char-ngram vectorizer
│   ├── kannada_multilabel_model.pkl # 6 trained One-vs-Rest Kannada classifiers
│   ├── kannada_model_metadata.json  # Kannada model thresholds & metadata
│   └── evaluation_metrics.json    # Official held-out test evaluation results
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Text sanitization & Unicode schema validation
│   ├── youtube_analyzer.py        # YouTube URL parsing, API fetch, & dual-batch analysis
│   ├── predict.py                 # Multi-label bilingual routing & ensembling
│   ├── kannada_analyzer.py        # Kannada language detection & context NLP engine
│   ├── context_analyzer.py        # English sentence-level contextual NLP
│   ├── train.py                   # English model training & threshold tuning
│   └── evaluate.py                # Held-out benchmark evaluation
├── scripts/
│   ├── download_data.py           # Automated Jigsaw dataset acquisition
│   ├── prepare_data.py            # Dataset cleaning & split preparation
│   └── train_kannada_model.py     # Kannada model training pipeline
├── web/
│   ├── app.py                     # Flask backend & REST API
│   ├── static/
│   │   ├── css/style.css          # Modern dark design system
│   │   └── js/app.js              # Client-side SPA controller
│   └── templates/
│       └── index.html             # Single-page user interface
├── tests/
│   ├── test_youtube_analyzer.py   # Unit & integration tests for YouTube analyzer
│   ├── test_preprocessing.py      # Unit tests for text cleaning
│   ├── test_model.py              # Model inference tests
│   ├── test_context_toxicity.py   # Contextual linguistic tests
│   ├── test_kannada_analysis.py   # Kannada NLP and model tests
│   └── test_api.py                # Web server endpoint tests
├── .env.example                   # Environment variable template
├── requirements.txt               # Documented dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Comprehensive documentation
```
