# SentiVerse AI (SocialPulse-AI)

SentiVerse AI is an end-to-end AI-powered social media data science platform built using Python 3.12+ and Streamlit. It enables users to collect, pre-process, analyze, and model social media posts to extract sentiment, detect multi-categorical emotions, cluster topics, perform explainable AI (XAI) attributions, forecast trends, and automatically generate professional business intelligence reports.

---

## Key Features

- **Multi-Source Collection**: Import CSV, Excel, and JSON files, or pull directly from Reddit, YouTube, and X APIs (with smart fallback mock data for testing).
- **Linguistic Preprocessing Pipeline**: Remove noise (URLs, HTML, punctuation) and resolve language barriers using a Google translation pipeline. Inspect tokens, POS tags, and Named Entity Recognition (NER) interactively.
- **Interactive EDA Dashboard**: Explore distributions, text length analytics, interactive N-grams, Word Clouds, timelines, and feature correlation heatmaps.
- **Model Workbench**: Train and tune 9 classical classifiers (Logistic Regression, Random Forest, XGBoost, SVM, LightGBM, etc.), PyTorch/Keras deep networks (LSTMs, CNNs), or load/fine-tune Transformer models (BERT, RoBERTa).
- **Explainable AI (XAI)**: Understand model decisions with LIME and SHAP attributions rendered as colored word-level confidence chips.
- **Topic Modeling & Trend Center**: Cluster posts using LDA, NMF, or Class-TF-IDF (BERTopic fallback) with PCA coordinate maps. Forecast future sentiment trends via Holt's Linear Exponential Smoothing.
- **Business Intelligence Reporting**: Calculate Brand Reputation Scores (with time-decay) and CSAT, run crisis alarms, and download executive summaries in Word or PDF formats.

---

## Directory Structure

```
SentiVerse AI/
│
├── config/
│   └── config.yaml               # Application config, defaults, hyperparameters
│
├── data/
│   └── sentiverse.db             # Local SQLite database (auto-created)
│
├── docker/
│   ├── Dockerfile                # Production-ready docker configuration
│   └── docker-compose.yml        # Multi-container orchestration setting
│
├── src/                          # Application source code
│   ├── data/
│   │   └── collector.py          # Data collectors (APIs, translation)
│   ├── preprocessing/
│   │   ├── cleaner.py            # Text cleaner pipeline & stats
│   │   └── nlp_pipeline.py       # POS, NER, tokenizer helpers
│   ├── features/
│   │   ├── text_stats.py         # Word/char counts, capital ratio stats
│   │   └── vectorizers.py        # TF-IDF, Count, and WordEmbeddings
│   ├── models/
│   │   ├── classical_ml.py       # scikit-learn/xgboost workbench
│   │   ├── deep_learning.py      # TensorFlow/Keras deep networks
│   │   └── transformers_model.py # Hugging Face transformers workbench
│   ├── explainability/
│   │   └── xai_explain.py        # SHAP and LIME text explainers
│   ├── business/
│   │   ├── emotion_detector.py   # 7-class emotion classifier
│   │   ├── advanced_detectors.py # Sarcasm, spam, toxicity filters
│   │   ├── topic_modeling.py     # LDA, NMF, BERTopic fallback
│   │   ├── trend_analyzer.py     # NER trends & timelines aggregator
│   │   ├── forecaster.py         # Holt time-series forecast engine
│   │   └── bi_engine.py          # Brand reputation, CSAT, crisis warnings
│   ├── utils/
│   │   ├── db.py                 # SQLAlchemy schemas & CRUD methods
│   │   └── report_gen.py         # Word/PDF document generators
│   └── dashboard/
│       └── pages/                # Individual Streamlit page modules
│
├── notebooks/
│   └── pipeline_walkthrough.ipynb # Step-by-step pipeline walkthrough notebook
│
├── tests/                        # Pytest unit testing suite
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
│
├── requirements.txt              # Project package dependencies
├── app.py                        # Streamlit dashboard router & styles
└── README.md                     # Documentation
```

---

## Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Verify that NLTK models are downloaded (they will also download automatically on app launch):

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### 2. Run the Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

### 3. Run Unit Tests

To run the full test suite and confirm everything is set up correctly:

```bash
python -m pytest tests/
```

---

## Container Deployment

To run SentiVerse AI within Docker:

### Build & Run via Docker Compose

From the root directory, run:

```bash
docker-compose -f docker/docker-compose.yml up --build
```

Access the dashboard at `http://localhost:8501`. The database will persist under the locally mounted `./data` directory.
