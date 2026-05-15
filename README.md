# OSINT Intelligence Center - ML1 Project

Advanced NLP, geospatial context and machine-learning dashboard for OSINT-style analysis of the Iran-Israel-US conflict narrative.

## What This Project Includes

- Professional Streamlit dashboard in `app.py`
- Overview with KPI cards and automatic executive summary
- Geospatial Intelligence map with NASA FIRMS and OpenSky points
- Timeline analysis with daily/weekly filters and spike detection
- UMAP Semantic Explorer powered by sentence-transformer embeddings
- BERTopic topic analysis with top words and embedded HTML view
- ML model comparison, accuracy/macro-F1 charts and confusion matrices
- FAISS semantic search over the document corpus
- Filterable News Explorer with clickable URLs
- About / Methodology page with sources, pipeline, weak labels and limitations

## Project Structure

```text
Proyecto_final_ml/
├── app.py
├── dashboard/
│   └── assets/
│       └── styles.css
├── data/
│   ├── dataset_nlp_labeled.csv
│   ├── dataset_nlp_bertopic.csv
│   ├── dataset_nlp_embeddings.pkl
│   ├── nasa_firms.csv
│   └── opensky.csv
├── outputs/
│   ├── model_comparison_advanced.csv
│   ├── figures/
│   ├── advanced_figures/
│   │   └── bertopic_topics.html
│   └── models/
│       ├── bertopic_topics.csv
│       └── faiss_index.pkl
├── scripts/
├── notebooks/
├── report/
├── requirements.txt
└── README.md
```

## Local Setup

Recommended environment: macOS, Python 3.14, virtual environment.

```bash
cd Proyecto_final_ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Dashboard Sections

1. **Overview**: total documents, sources, temporal coverage, best ML model, topic count, conflict-related documents and executive summary.
2. **Geospatial Intelligence**: Plotly map with NASA FIRMS and OpenSky points, source/date filters, tooltips and optional heatmap.
3. **Timeline Analysis**: daily or weekly activity, source/label filters and automatic spike register.
4. **Semantic Explorer**: interactive UMAP projection colored by label, source, topic or cluster.
5. **Topic Analysis**: BERTopic topic table, top words, document distribution and embedded `bertopic_topics.html` when present.
6. **ML Models**: model leaderboard, accuracy/macro-F1 bars and confusion matrices.
7. **Semantic Search**: FAISS similarity search with score, source, date, label, title and URL. If the sentence-transformer model cannot be loaded, the dashboard falls back to a local TF-IDF similarity search.
8. **News Explorer**: searchable and filterable article table with clickable links.
9. **About / Methodology**: data sources, pipeline, weak labels, limitations and geospatial context explanation.

## Required Dependencies

Main dashboard dependencies are already listed in `requirements.txt`:

- `streamlit`
- `pandas`, `numpy`
- `plotly`
- `scikit-learn`
- `sentence-transformers`
- `umap-learn`
- `bertopic`
- `faiss-cpu`
- `folium`
- `matplotlib`, `seaborn`
- `nltk`
- `xgboost`

For semantic search, the first query may take longer because the sentence-transformer model is loaded into memory. If the model is unavailable offline, the dashboard still returns local TF-IDF fallback results instead of breaking.

## Regenerate Pipeline Outputs

If any dashboard section shows a missing-file warning, regenerate the pipeline artifacts:

```bash
python scripts/prepare_dataset_nlp.py
python scripts/nlp_analysis.py
python scripts/supervised_ml.py
python scripts/semantic_embeddings.py
python scripts/bertopic_analysis.py
python scripts/semantic_search.py
python scripts/advanced_ml.py
```

The dashboard is robust to missing files, but the richest experience uses the full `data/`, `outputs/models/` and `outputs/advanced_figures/` artifacts.

## Deploy to Streamlit Community Cloud

Official Streamlit deployment docs: https://docs.streamlit.io/deploy/streamlit-community-cloud

1. Push the repository to GitHub.
2. Make sure `app.py`, `requirements.txt`, `data/`, `outputs/models/` and `outputs/advanced_figures/` are committed if the app should run without rebuilding artifacts.
3. Go to Streamlit Community Cloud and create a new app from the GitHub repository.
4. Select the branch that contains this project.
5. Set the main file path to:

```text
app.py
```

6. Deploy the app. Streamlit Cloud will install packages from `requirements.txt`.
7. If dependency installation is too heavy for the free runtime, deploy a lighter branch that keeps generated artifacts and removes unused training-only packages.

Notes:

- Do not put `python` itself inside `requirements.txt`.
- If you need API keys or private config, use Streamlit secrets instead of committing `.env`.
- This project is designed for Python 3.14 locally. On Streamlit Cloud, use the newest supported Python runtime available for your workspace if 3.14 is not offered.

## Expected Screenshots

Capture these views for the final report or submission:

- `screenshots/overview.png`: KPI cards, executive summary and quick analytics.
- `screenshots/geospatial_intelligence.png`: map with NASA FIRMS/OpenSky filters and visible point tooltips.
- `screenshots/timeline_spikes.png`: daily or weekly timeline with spike markers.
- `screenshots/semantic_explorer.png`: UMAP scatter colored by `weak_label`.
- `screenshots/topic_analysis.png`: BERTopic table, top-word chips and embedded topic visualization.
- `screenshots/ml_models.png`: model comparison bars and one confusion matrix.
- `screenshots/semantic_search.png`: query results with similarity scores and URLs.
- `screenshots/news_explorer.png`: filtered document table with clickable links.

## Current Results Snapshot

- 289 processed documents
- Sources: Google News RSS, GDELT, Al Jazeera RSS, BBC RSS
- Weak labels: escalation, military, diplomacy, other, energy, humanitarian, sanctions, cyber
- Best current model: Gradient Boosting / GradientBoosting, about 91% accuracy on the advanced comparison output
- Geospatial context: NASA FIRMS hotspots and OpenSky aircraft points are treated as contextual overlays, not as the primary NLP corpus

## Important Limitations

- Weak labels are heuristic and not expert-validated ground truth.
- The corpus is small and source distribution is imbalanced.
- Geospatial observations do not prove causal relationships with news events.
- The platform is intended for academic OSINT analysis and exploratory intelligence workflows.
