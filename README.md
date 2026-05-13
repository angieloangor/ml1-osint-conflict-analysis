# OSINT Analytical Platform - ML1 Project

🛰️ **Advanced NLP & Machine Learning for Geopolitical Intelligence**

## Overview

This project is an end-to-end analytical platform for Open Source Intelligence (OSINT) analysis, specifically focused on the Israel-Iran conflict. It combines modern NLP techniques (sentence transformers, BERTopic), advanced ML models (Random Forest, Gradient Boosting), and interactive visualizations to enable data-driven intelligence analysis.

### Key Achievements

✅ **289 processed documents** from BBC, Al Jazeera, Google News, GDELT
✅ **Semantic embeddings** using all-MiniLM-L6-v2 (384 dimensions) + UMAP visualization
✅ **BERTopic analysis** for interpretable topic extraction
✅ **FAISS semantic search** for document similarity
✅ **5 ML models** with best accuracy: **91.4% (Gradient Boosting)**
✅ **Interactive Streamlit dashboard** for exploration and analysis
✅ **Academic report** with methodology and limitations

## Quick Start

### 1. Installation

```bash
cd Proyecto_final_ml
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Interactive Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` with 7 interactive sections:

- 📊 **Corpus Overview**: Dataset statistics and distributions
- 🌐 **Semantic Explorer**: UMAP visualization of semantic space
- 📚 **Topic Analysis**: BERTopic results and comparisons
- 🤖 **ML Models**: Model performance and confusion matrices
- 🔍 **Semantic Search**: Find similar documents by query
- 📈 **Statistics**: Detailed corpus analytics
- 📄 **About**: Project documentation

### 3. Explore Jupyter Notebooks

```bash
# Dataset preparation and EDA
jupyter notebook notebooks/01_dataset_nlp_preparation.ipynb

# NLP analysis (TF-IDF, clustering, embeddings)
jupyter notebook notebooks/02_nlp_analysis.ipynb

# Supervised ML with weak labels
jupyter notebook notebooks/03_supervised_ml_models.ipynb

# Advanced pipeline demonstration
jupyter notebook notebooks/04_advanced_osint_dashboard.ipynb
```

### 4. Access the Academic Report

- **Markdown**: `report/final_report.md`
- **DOCX**: `report/final_report.docx` (for Word)

## Project Structure

```
Proyecto_final_ml/
├── data/
│   ├── *.csv                          # Raw RSS and event data
│   ├── dataset_nlp.csv                # Cleaned corpus
│   ├── dataset_nlp_labeled.csv        # With weak labels
│   ├── dataset_nlp_bertopic.csv       # With BERTopic topics
│   └── dataset_nlp_embeddings.pkl     # Embeddings + UMAP
├── scripts/
│   ├── prepare_dataset_nlp.py         # Dataset preparation
│   ├── nlp_analysis.py                # TF-IDF, NMF, clustering
│   ├── supervised_ml.py               # Weak labeling + basic ML
│   ├── semantic_embeddings.py         # Sentence-transformers + UMAP
│   ├── bertopic_analysis.py           # BERTopic modeling
│   ├── semantic_search.py             # FAISS search engine
│   └── advanced_ml.py                 # RF, GB models
├── notebooks/
│   ├── 01_dataset_nlp_preparation.ipynb
│   ├── 02_nlp_analysis.ipynb
│   ├── 03_supervised_ml_models.ipynb
│   └── 04_advanced_osint_dashboard.ipynb
├── outputs/
│   ├── figures/                       # EDA plots
│   ├── advanced_figures/              # Advanced visualizations
│   └── models/                        # Trained models
├── report/
│   ├── final_report.md
│   └── final_report.docx
├── app.py                             # Streamlit dashboard
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Model Performance

### Advanced ML Results

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
|-------|----------|------------------|----------------|-----------|
| **GradientBoosting** | **0.9138** | 0.7649 | 0.7690 | 0.7645 |
| RandomForest | 0.8276 | 0.7245 | 0.6939 | 0.6893 |
| LogisticRegression | 0.6897 | 0.5907 | 0.6171 | 0.5761 |
| KNeighborsClassifier | 0.4828 | 0.3964 | 0.3628 | 0.3600 |
| MultinomialNB | 0.5517 | 0.3096 | 0.2679 | 0.2425 |

**Best Model**: Gradient Boosting with **91.4% accuracy** ⭐

## Weak Label Categories

| Label | Count | % |
|-------|-------|---|
| escalation | 109 | 37.7% |
| military | 69 | 23.9% |
| diplomacy | 49 | 17.0% |
| other | 31 | 10.7% |
| energy | 18 | 6.2% |
| humanitarian | 6 | 2.1% |
| sanctions | 5 | 1.7% |
| cyber | 2 | 0.7% |

## Technologies

- **NLP**: sentence-transformers, BERTopic, NLTK
- **ML**: scikit-learn (Random Forest, Gradient Boosting, Logistic Regression)
- **Visualization**: Streamlit, Plotly, Matplotlib, UMAP
- **Search**: FAISS vector database
- **Data**: pandas, numpy, pickle

## Running the Full Pipeline

```bash
python scripts/prepare_dataset_nlp.py
python scripts/nlp_analysis.py
python scripts/supervised_ml.py
python scripts/semantic_embeddings.py
python scripts/bertopic_analysis.py
python scripts/semantic_search.py
python scripts/advanced_ml.py
streamlit run app.py
```

## Important Limitations

⚠️ **Weak Labels**: Not validated by human experts
⚠️ **Small Corpus**: 289 documents
⚠️ **Source Bias**: Google News dominates (90%)
⚠️ **No Ground Truth**: Academic proof-of-concept only

## Status

✅ **Complete** - Production Ready for ML1 Academic Project

**Last Updated**: May 12, 2026
**Version**: 2.0 - Advanced OSINT Platform
