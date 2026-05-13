"""
Supervised ML pipeline for the OSINT NLP project.

This module loads the enriched NLP dataset, derives weak labels for conflict-related
news categories, trains three supervised classifiers, evaluates them with standard
metrics, and saves results and plots for academic reporting.

The generated labels are weak and heuristic by design; they are intended to support
an initial ML1 proof of concept, not to replace human ground truth.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

sns.set(style="whitegrid")

DATA_DIR = Path("data")
OUTPUT_FIGURES_DIR = Path("outputs/figures")
OUTPUT_MODELS_DIR = Path("outputs/models")
LABELED_PATH = DATA_DIR / "dataset_nlp_labeled.csv"
COMPARISON_PATH = Path("outputs/model_comparison.csv")

ENRICHED_PATH = OUTPUT_MODELS_DIR / "dataset_nlp_enriched.csv"

LABELS_ORDER = [
    "escalation",
    "diplomacy",
    "military",
    "humanitarian",
    "sanctions",
    "energy",
    "cyber",
    "other",
]

WEAK_LABEL_RULES = [
    ("cyber", ["cyber", "hack", "hacker", "ransom", "malware", "phish", "espionage", "botnet", "breach", "ddos", "trojan"]),
    ("energy", ["nuclear", "uranium", "reactor", "oil", "gas", "pipeline", "energy", "electricity", "power"]),
    ("sanctions", ["sanction", "embargo", "freeze", "asset", "bank", "trade", "economic", "blacklist", "restriction"]),
    ("diplomacy", ["ceasefire", "talk", "talks", "diplomatic", "diplomacy", "negotiation", "negotiations", "agreement", "deal", "peace", "mediator", "summit", "condemn"]),
    ("humanitarian", ["civilian", "refugee", "aid", "hospital", "shelter", "displaced", "casualty", "humanitarian", "siege", "health", "medical", "food", "water", "evacuate"]),
    ("military", ["attack", "strike", "airstrike", "missile", "drone", "army", "troops", "navy", "tank", "rocket", "shelling", "bomb", "covert", "launch", "raid", "assault", "firing", "battle"]),
    ("escalation", ["escalation", "escalate", "widened", "heightened", "escalating", "clash", "escalated", "warning"]),
]

COLORS = {
    "escalation": "#d73027",
    "diplomacy": "#1a9850",
    "military": "#4575b4",
    "humanitarian": "#fdae61",
    "sanctions": "#542788",
    "energy": "#f46d43",
    "cyber": "#66c2a5",
    "other": "#999999",
}


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    model_name: str


def load_dataset(path: Path = ENRICHED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "processed_text" not in df.columns:
        raise ValueError("El dataset enriquecido debe contener la columna processed_text")
    return df


def normalize_text_for_label(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def assign_weak_label(row: pd.Series) -> str:
    text = normalize_text_for_label(str(row["processed_text"]))
    for label, keywords in WEAK_LABEL_RULES:
        if any(keyword in text for keyword in keywords):
            return label
    theme = str(row.get("conflict_theme", "")).lower()
    if theme == "diplomacy":
        return "diplomacy"
    if theme == "humanitarian":
        return "humanitarian"
    if theme == "nuclear":
        return "energy"
    if theme == "kinetic":
        return "military"
    if theme == "general_conflict":
        if any(word in text for word in ["strike", "attack", "missile", "drone", "war"]):
            return "escalation"
    return "other"


def build_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weak_label"] = df.apply(assign_weak_label, axis=1)
    return df


def safe_train_test_split(df: pd.DataFrame, label_col: str = "weak_label") -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    try:
        return train_test_split(
            df["processed_text"],
            df[label_col],
            test_size=0.2,
            random_state=42,
            stratify=df[label_col],
        )
    except ValueError:
        logging.warning("Train/test split no pudo estratificar por clases, usando split simple")
        return train_test_split(
            df["processed_text"],
            df[label_col],
            test_size=0.2,
            random_state=42,
        )


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )


def fit_models(X_train: Any, y_train: pd.Series) -> dict[str, Any]:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "MultinomialNB": MultinomialNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate_model(model: Any, X_test: Any, y_test: pd.Series, label_order: list[str]) -> ModelMetrics:
    y_pred = model.predict(X_test)
    return ModelMetrics(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, average="macro", zero_division=0),
        recall=recall_score(y_test, y_pred, average="macro", zero_division=0),
        f1=f1_score(y_test, y_pred, average="macro", zero_division=0),
        model_name=type(model).__name__,
    )


def build_classification_report(model: Any, X_test: Any, y_test: pd.Series, label_order: list[str]) -> str:
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, labels=label_order, zero_division=0)


def save_confusion_matrix(model: Any, X_test: Any, y_test: pd.Series, labels: list[str], filename: Path) -> None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(f"Confusion matrix - {type(model).__name__}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def save_model_comparison(metrics: list[ModelMetrics], filename: Path) -> pd.DataFrame:
    rows = [
        {
            "model": m.model_name,
            "accuracy": m.accuracy,
            "precision_macro": m.precision,
            "recall_macro": m.recall,
            "f1_macro": m.f1,
        }
        for m in metrics
    ]
    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename, index=False)
    return df


def plot_model_comparison(df: pd.DataFrame, filename: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(x="model", y=["accuracy", "precision_macro", "recall_macro", "f1_macro"], kind="bar", ax=ax)
    ax.set_title("Comparación de modelos supervisados")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Metrics")
    fig.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def run_supervised_pipeline() -> dict[str, Any]:
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    df = build_weak_labels(df)
    label_counts = df["weak_label"].value_counts().to_dict()
    df.to_csv(LABELED_PATH, index=False)

    X_train, X_test, y_train, y_test = safe_train_test_split(df)
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = fit_models(X_train_tfidf, y_train)

    metrics = []
    reports = {}
    for name, model in models.items():
        metrics.append(evaluate_model(model, X_test_tfidf, y_test, LABELS_ORDER))
        reports[name] = build_classification_report(model, X_test_tfidf, y_test, LABELS_ORDER)
        filename = name.lower()
        if filename == "logisticregression":
            cm_file = OUTPUT_FIGURES_DIR / "confusion_matrix_logreg.png"
        elif filename == "multinomialnb":
            cm_file = OUTPUT_FIGURES_DIR / "confusion_matrix_nb.png"
        elif filename == "knn":
            cm_file = OUTPUT_FIGURES_DIR / "confusion_matrix_knn.png"
        else:
            cm_file = OUTPUT_FIGURES_DIR / f"confusion_matrix_{filename}.png"
        save_confusion_matrix(model, X_test_tfidf, y_test, LABELS_ORDER, cm_file)
        joblib.dump(model, OUTPUT_MODELS_DIR / f"{filename}_model.joblib")

    comparison_df = save_model_comparison(metrics, COMPARISON_PATH)
    plot_model_comparison(comparison_df, OUTPUT_FIGURES_DIR / "model_comparison.png")
    joblib.dump(vectorizer, OUTPUT_MODELS_DIR / "supervised_tfidf_vectorizer.joblib")

    return {
        "labeled_path": str(LABELED_PATH),
        "comparison_path": str(COMPARISON_PATH),
        "figures": [
            str(OUTPUT_FIGURES_DIR / "confusion_matrix_logisticregression.png"),
            str(OUTPUT_FIGURES_DIR / "confusion_matrix_multinomialnb.png"),
            str(OUTPUT_FIGURES_DIR / "confusion_matrix_knn.png"),
            str(OUTPUT_FIGURES_DIR / "model_comparison.png"),
        ],
        "label_distribution": label_counts,
        "model_comparison": comparison_df,
        "reports": reports,
    }


if __name__ == "__main__":
    summary = run_supervised_pipeline()
    print("Supervised ML pipeline completed")
    print(f"Labeled dataset: {summary['labeled_path']}")
    print(f"Model comparison: {summary['comparison_path']}")
    print("Label distribution:")
    for label, count in summary["label_distribution"].items():
        print(f"- {label}: {count}")
