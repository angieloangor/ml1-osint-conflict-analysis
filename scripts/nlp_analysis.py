"""
NLP analysis module for Proyecto_final_ml.

Este módulo carga `data/dataset_nlp.csv`, aplica procesamiento de texto, extrae
features con TF-IDF, identifica temas y clusters, genera embeddings semánticos y
salva artefactos en `outputs/models` y visualizaciones en `outputs/figures`.

Diseñado para ser reproducible con Python 3.14 y usar scikit-learn como base.
"""

from __future__ import annotations

import json
import re
import unicodedata
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError as exc:
    raise ImportError(
        "nltk no está instalado. Instala nltk y descarga wordnet, stopwords y omw-1.4. "
        "Por ejemplo: python -m pip install nltk && python -m nltk.downloader wordnet stopwords omw-1.4"
    ) from exc

sns.set(style="whitegrid")

DATA_DIR = Path("data")
OUTPUT_MODELS_DIR = Path("outputs/models")
OUTPUT_FIGURES_DIR = Path("outputs/figures")
DATASET_PATH = DATA_DIR / "dataset_nlp.csv"
ENRICHED_DATASET_PATH = OUTPUT_MODELS_DIR / "dataset_nlp_enriched.csv"
SIMILAR_DOCS_PATH = OUTPUT_MODELS_DIR / "document_similarity.csv"
TOPICS_PATH = OUTPUT_MODELS_DIR / "topics.csv"

CONFLICT_KEYWORDS = {
    "iran",
    "israel",
    "tehran",
    "jerusalem",
    "hamas",
    "hezbollah",
    "missile",
    "strike",
    "attack",
    "airstrike",
    "diplomatic",
    "sanctions",
    "nuclear",
    "drone",
    "peace",
    "ceasefire",
    "escalation",
    "conflict",
    "war",
    "troops",
    "plane",
    "ship",
    "navy",
    "tank",
    "rocket",
}

STOPWORDS: set[str] = set()

NGRAM_RANGE = (1, 2)
MAX_FEATURES = 6000
NMF_COMPONENTS = 6
KMEANS_CLUSTERS = 8
SVD_COMPONENTS = 50
SVD_PLOT_COMPONENTS = 2
TOP_N_SIMILAR = 3


@dataclass
class NlpMetrics:
    silhouette: float
    explained_variance_ratio: float
    n_documents: int
    n_clusters: int
    n_topics: int


def ensure_nltk_resources() -> None:
    """Descarga recursos NLTK necesarios si no están disponibles."""
    for resource in ("wordnet", "stopwords", "omw-1.4"):
        try:
            if resource == "stopwords":
                stopwords.words("english")
            else:
                nltk.corpus.wordnet.ensure_loaded()
            continue
        except LookupError:
            nltk.download(resource)


def normalize_text(text: str) -> str:
    """Normaliza Unicode y limpia el texto base."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.[^\s]+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def tokenize(text: str) -> list[str]:
    """Tokenización simple basada en expresiones regulares."""
    return re.findall(r"\b[a-z']{2,}\b", text)


def lemmatize_tokens(tokens: Iterable[str]) -> list[str]:
    """Lematiza tokens usando NLTK WordNet."""
    lemmatizer = WordNetLemmatizer()
    normalized_tokens: list[str] = []
    for token in tokens:
        if token.isdigit():
            continue
        lemma = lemmatizer.lemmatize(token, pos="v")
        lemma = lemmatizer.lemmatize(lemma, pos="n")
        if len(lemma) > 1:
            normalized_tokens.append(lemma)
    return normalized_tokens


def get_stopwords() -> set[str]:
    """Carga las stopwords de NLTK tras asegurar los recursos."""
    global STOPWORDS
    if not STOPWORDS:
        ensure_nltk_resources()
        STOPWORDS = set(stopwords.words("english"))
    return STOPWORDS


def remove_stopwords(tokens: Iterable[str]) -> list[str]:
    """Remueve stopwords comunes en inglés."""
    stopword_set = get_stopwords()
    return [token for token in tokens if token not in stopword_set]


def preprocess_text(text: str) -> str:
    """Pipeline ligero de NLP: normalización, tokenización, stopwords y lematización."""
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


def detect_conflict_related(tokens: Iterable[str]) -> bool:
    """Detecta si el texto está relacionado con el conflicto Iran-Israel mediante palabras clave."""
    tokens_lower = set(token.lower() for token in tokens)
    return bool(tokens_lower & CONFLICT_KEYWORDS)


def label_conflict_theme(tokens: Iterable[str]) -> str:
    """Clasifica temas de conflicto con etiquetas heurísticas."""
    token_set = set(tokens)
    if token_set & {"nuclear", "uranium", "enrichment", "reactor"}:
        return "nuclear"
    if token_set & {"sanction", "diplomatic", "peace", "ceasefire", "talk"}:
        return "diplomacy"
    if token_set & {"missile", "strike", "attack", "drone", "tank", "troop"}:
        return "kinetic"
    if token_set & {"protest", "refugee", "civilian", "siege"}:
        return "humanitarian"
    return "general_conflict"


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Carga el dataset principal de texto y asegura columnas base."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    expected = {"timestamp", "source", "title", "text", "url"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en dataset_nlp.csv: {sorted(missing)}")
    return df


def build_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Construye columnas procesadas necesarias para el análisis NLP."""
    source_text = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    df = df.copy()
    df["raw_text"] = source_text
    df["processed_text"] = df["raw_text"].map(preprocess_text)
    df["token_count"] = df["processed_text"].str.split().apply(len)
    df["conflict_related"] = df["processed_text"].str.split().apply(detect_conflict_related)
    df["conflict_theme"] = df["processed_text"].str.split().apply(label_conflict_theme)
    df = df[df["processed_text"].str.len() > 0].reset_index(drop=True)
    return df


def build_tfidf_matrix(documents: Iterable[str]) -> tuple[TfidfVectorizer, np.ndarray]:
    """Genera la matriz TF-IDF para el corpus procesado."""
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        norm="l2",
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix


def extract_topics(tfidf_matrix: np.ndarray, vectorizer: TfidfVectorizer, n_topics: int = NMF_COMPONENTS) -> tuple[NMF, list[str], np.ndarray]:
    """Extrae tópicos usando NMF sobre TF-IDF."""
    model = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=400)
    topic_weights = model.fit_transform(tfidf_matrix)
    topic_labels = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        top_indexes = topic.argsort()[::-1][:10]
        topic_terms = [feature_names[i] for i in top_indexes]
        topic_labels.append(" ".join(topic_terms[:6]))
    return model, topic_labels, topic_weights


def semantic_embeddings(tfidf_matrix: np.ndarray, n_components: int = SVD_COMPONENTS) -> tuple[TruncatedSVD, np.ndarray]:
    """Construye embeddings semánticos densos a partir de TF-IDF."""
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)
    return svd, embeddings


def cluster_documents(embeddings: np.ndarray, n_clusters: int = KMEANS_CLUSTERS) -> tuple[KMeans, np.ndarray]:
    """Agrupa noticias usando KMeans sobre embeddings semánticos."""
    if embeddings.shape[0] < n_clusters:
        n_clusters = max(2, embeddings.shape[0] // 2)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(embeddings)
    return model, labels


def compute_similarity(embeddings: np.ndarray, top_n: int = TOP_N_SIMILAR) -> pd.DataFrame:
    """Genera las noticias más similares por similitud de coseno."""
    matrix = cosine_similarity(embeddings)
    rows = []
    for index in range(matrix.shape[0]):
        similarity_scores = list(enumerate(matrix[index]))
        similarity_scores = [(i, score) for i, score in similarity_scores if i != index]
        similarity_scores.sort(key=lambda item: item[1], reverse=True)
        for neighbor_idx, score in similarity_scores[:top_n]:
            rows.append({"document_id": index, "similar_document_id": neighbor_idx, "score": float(score)})
    return pd.DataFrame(rows)


def temporal_cluster_summary(df: pd.DataFrame, cluster_col: str = "cluster_id") -> pd.DataFrame:
    """Detección de agrupamientos temporales por cluster de eventos."""
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    summary = df.groupby(["date", cluster_col]).size().reset_index(name="count")
    return summary


def plot_topic_distribution(df: pd.DataFrame) -> None:
    counts = df["topic_id"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.astype(str), counts.values, color=sns.color_palette("muted", len(counts)))
    ax.set_title("Distribución de tópicos en el corpus")
    ax.set_xlabel("ID de tópico")
    ax.set_ylabel("Número de noticias")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES_DIR / "topic_distribution.png", dpi=200)
    plt.close(fig)


def plot_cluster_timeline(summary: pd.DataFrame) -> None:
    pivot = summary.pivot(index="date", columns="cluster_id", values="count").fillna(0)
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="line", marker="o", ax=ax)
    ax.set_title("Comportamiento temporal de clusters de noticias")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Número de noticias")
    ax.legend(title="Cluster")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES_DIR / "cluster_timeline.png", dpi=200)
    plt.close(fig)


def plot_embedding_space(embeddings: np.ndarray, labels: np.ndarray) -> None:
    if embeddings.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="tab10", alpha=0.75)
    ax.set_title("Proyección semántica de noticias")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES_DIR / "semantic_embedding_scatter.png", dpi=200)
    plt.close(fig)


def plot_conflict_vs_date(df: pd.DataFrame) -> None:
    timeline = df.groupby([df["timestamp"].dt.date, "conflict_related"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    timeline.plot(kind="line", marker="o", ax=ax)
    ax.set_title("Noticias relacionadas con conflicto por fecha")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Número de noticias")
    ax.legend(title="Conflicto")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES_DIR / "conflict_timeline.png", dpi=200)
    plt.close(fig)


def build_report(df: pd.DataFrame, metrics: NlpMetrics) -> dict[str, str]:
    """Genera un resumen metodológico y métricas clave para el informe."""
    n_conflict = int(df["conflict_related"].sum())
    n_general = len(df) - n_conflict
    return {
        "documents": str(metrics.n_documents),
        "clusters": str(metrics.n_clusters),
        "topics": str(metrics.n_topics),
        "silhouette_score": f"{metrics.silhouette:.4f}",
        "explained_variance_ratio": f"{metrics.explained_variance_ratio:.4f}",
        "conflict_related_documents": str(n_conflict),
        "general_documents": str(n_general),
    }


def save_models(
    vectorizer: TfidfVectorizer,
    nmf_model: NMF,
    kmeans_model: KMeans,
    svd_model: TruncatedSVD,
) -> None:
    OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, OUTPUT_MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(nmf_model, OUTPUT_MODELS_DIR / "nmf_model.joblib")
    joblib.dump(kmeans_model, OUTPUT_MODELS_DIR / "kmeans_model.joblib")
    joblib.dump(svd_model, OUTPUT_MODELS_DIR / "svd_model.joblib")


def save_topic_terms(topic_labels: list[str]) -> None:
    rows = [{"topic_id": idx, "topic_terms": label} for idx, label in enumerate(topic_labels)]
    pd.DataFrame(rows).to_csv(TOPICS_PATH, index=False)


def save_similarity(similarity_df: pd.DataFrame) -> None:
    similarity_df.to_csv(SIMILAR_DOCS_PATH, index=False)


def generate_nlp_outputs() -> dict[str, str]:
    ensure_nltk_resources()
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    df = build_corpus(df)

    vectorizer, tfidf_matrix = build_tfidf_matrix(df["processed_text"])
    nmf_model, topic_labels, topic_weights = extract_topics(tfidf_matrix, vectorizer)
    df["topic_id"] = topic_weights.argmax(axis=1)
    df["topic_label"] = [topic_labels[i] for i in df["topic_id"]]
    df["topic_weight"] = topic_weights.max(axis=1)

    svd_model, embeddings = semantic_embeddings(tfidf_matrix)
    kmeans_model, cluster_labels = cluster_documents(embeddings)
    df["cluster_id"] = cluster_labels

    similarity_df = compute_similarity(embeddings)
    temporal_summary = temporal_cluster_summary(df)

    save_models(vectorizer, nmf_model, kmeans_model, svd_model)
    save_topic_terms(topic_labels)
    save_similarity(similarity_df)
    df.to_csv(ENRICHED_DATASET_PATH, index=False)
    temporal_summary.to_csv(OUTPUT_MODELS_DIR / "temporal_clusters.csv", index=False)

    plot_topic_distribution(df)
    plot_cluster_timeline(temporal_summary)
    plot_embedding_space(TruncatedSVD(n_components=SVD_PLOT_COMPONENTS, random_state=42).fit_transform(tfidf_matrix), cluster_labels)
    plot_conflict_vs_date(df)

    metrics = NlpMetrics(
        silhouette=float(silhouette_score(embeddings, cluster_labels)) if len(set(cluster_labels)) > 1 else 0.0,
        explained_variance_ratio=float(svd_model.explained_variance_ratio_.sum()),
        n_documents=len(df),
        n_clusters=len(set(cluster_labels)),
        n_topics=len(topic_labels),
    )

    report = build_report(df, metrics)
    Path(OUTPUT_MODELS_DIR / "nlp_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report


def main() -> None:
    report = generate_nlp_outputs()
    print("NLP pipeline completado")
    for key, value in report.items():
        print(f"- {key}: {value}")
    print(f"Dataset enriquecido guardado en: {ENRICHED_DATASET_PATH}")
    print(f"Modelos guardados en: {OUTPUT_MODELS_DIR}")
    print(f"Figuras guardadas en: {OUTPUT_FIGURES_DIR}")


if __name__ == "__main__":
    main()
