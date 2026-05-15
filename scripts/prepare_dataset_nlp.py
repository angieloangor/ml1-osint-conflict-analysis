from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/figures")
INPUT_FILES = [
    DATA_DIR / "bbc_rss.csv",
    DATA_DIR / "aljazeera_rss.csv",
    DATA_DIR / "google_news_rss.csv",
    DATA_DIR / "gdelt.csv",
    DATA_DIR / "acled.csv",
    DATA_DIR / "ukmto.csv",
    DATA_DIR / "bluesky_posts.csv",
    DATA_DIR / "youtube_metadata.csv",
]
OUTPUT_CSV = DATA_DIR / "dataset_nlp.csv"
REQUIRED_COLUMNS = ["timestamp", "source", "title", "text", "url"]
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "has",
    "are",
    "was",
    "were",
    "been",
    "their",
    "they",
    "but",
    "not",
    "will",
    "its",
    "his",
    "her",
    "she",
    "he",
    "you",
    "your",
    "his",
    "them",
    "about",
    "than",
    "who",
    "which",
    "when",
    "where",
    "what",
    "can",
    "all",
    "any",
    "one",
    "our",
    "there",
    "been",
    "more",
    "also",
    "into",
    "after",
    "over",
    "such",
    "their",
    "were",
    "had",
    "its",
    "may",
    "many",
    "most",
    "other",
    "than",
    "some",
    "even",
}


def normalize_string(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_source_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column {col} in {path}")
    return df.copy()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[REQUIRED_COLUMNS]
    for col in ["source", "title", "text", "url"]:
        df[col] = df[col].map(normalize_string)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    before_rows = len(df)
    df = df.dropna(subset=["timestamp", "source", "title", "text", "url"])
    df["full_text"] = (df["title"] + " ").str.strip() + " " + df["text"].str.strip()
    df["full_text"] = df["full_text"].map(normalize_string)
    df = df[df["full_text"] != ""]
    df = df.drop_duplicates(subset=["timestamp", "source", "title", "text", "url", "full_text"]).reset_index(drop=True)
    after_rows = len(df)
    removed = before_rows - after_rows
    if removed > 0:
        print(f"Removed {removed} rows during cleaning for source {df['source'].iloc[0] if not df.empty else 'unknown'}")
    return df


def count_missing_by_column(df: pd.DataFrame) -> pd.Series:
    return df[REQUIRED_COLUMNS].isna().sum()


def build_corpus() -> tuple[pd.DataFrame, pd.Series, int, int]:
    datasets = []
    missing_by_source = []
    empty_texts_removed = 0
    duplicates_removed = 0
    for path in INPUT_FILES:
        if not path.exists():
            print(f"Skipping missing optional NLP source: {path}")
            continue
        source_name = path.stem
        df = load_source_csv(path)
        missing_by_source.append(df[REQUIRED_COLUMNS].isna().sum())
        before = len(df)
        cleaned = clean_dataframe(df)
        after = len(cleaned)
        empty_texts_removed += max(0, before - after)
        if not cleaned.empty:
            datasets.append(cleaned)
    if not datasets:
        empty = pd.DataFrame(columns=REQUIRED_COLUMNS + ["full_text"])
        missing = pd.DataFrame(0, index=REQUIRED_COLUMNS, columns=["missing"])
        return empty, missing, duplicates_removed, empty_texts_removed
    combined = pd.concat(datasets, ignore_index=True)
    before_dups = len(combined)
    combined = combined.drop_duplicates(subset=["timestamp", "source", "title", "text", "url", "full_text"]).reset_index(drop=True)
    duplicates_removed = before_dups - len(combined)
    missing = pd.concat(missing_by_source, axis=1) if missing_by_source else pd.DataFrame(0, index=REQUIRED_COLUMNS, columns=["missing"])
    return combined, missing, duplicates_removed, empty_texts_removed


def plot_distribution_by_source(df: pd.DataFrame) -> None:
    counts = df["source"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(counts.index, counts.values, color=sns.color_palette("muted", len(counts)))
    ax.set_title("Distribución de filas por fuente")
    ax.set_xlabel("Cantidad de registros")
    ax.set_ylabel("Fuente")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "distribution_by_source.png", dpi=200)
    plt.close(fig)


def plot_text_length_distribution(df: pd.DataFrame) -> None:
    df = df.copy()
    df["text_length"] = df["full_text"].str.len()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["text_length"], bins=40, kde=True, color="#2c7fb8", ax=ax)
    ax.set_title("Distribución de longitud de textos (full_text)")
    ax.set_xlabel("Longitud de texto (caracteres)")
    ax.set_ylabel("Número de registros")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "text_length_distribution.png", dpi=200)
    plt.close(fig)


def plot_timeline(df: pd.DataFrame) -> None:
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    timeline = df.groupby(["date", "source"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    timeline.plot(kind="line", marker="o", ax=ax)
    ax.set_title("Timeline temporal por fuente")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Número de registros")
    ax.legend(title="Fuente")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "timeline_per_source.png", dpi=200)
    plt.close(fig)


def plot_missing_values(original_missing: pd.Series) -> None:
    summary = original_missing.groupby(original_missing.index).sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    summary.plot(kind="bar", color="#4c72b0", ax=ax)
    ax.set_title("Valores faltantes por columna antes de limpieza")
    ax.set_xlabel("Columna")
    ax.set_ylabel("Cantidad de valores faltantes")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "missing_values_before_cleaning.png", dpi=200)
    plt.close(fig)


def top_frequent_words(df: pd.DataFrame, top_n: int = 25) -> Counter[str]:
    all_text = " ".join(df["full_text"].astype(str).tolist()).lower()
    tokens = re.findall(r"\b[\w']{2,}\b", all_text)
    tokens = [token for token in tokens if token not in STOPWORDS]
    return Counter(tokens).most_common(top_n)


def plot_top_words(df: pd.DataFrame) -> None:
    top_words = top_frequent_words(df, top_n=20)
    words, counts = zip(*top_words)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(words, counts, color=sns.color_palette("viridis", len(words)))
    ax.set_title("Palabras más frecuentes en el corpus NLP")
    ax.set_xlabel("Frecuencia")
    ax.set_ylabel("Palabra")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_words.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined, missing_before, duplicates_removed, empty_removed = build_corpus()
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined.to_csv(OUTPUT_CSV, index=False)
    summary = {
        "total_rows": len(combined),
        "rows_per_source": combined["source"].value_counts().to_dict(),
        "duplicates_removed": int(duplicates_removed),
        "empty_texts_removed": int(empty_removed),
        "avg_text_length": float(combined["full_text"].str.len().mean()),
    }
    print("Resumen del corpus NLP:")
    for key, value in summary.items():
        print(f"- {key}: {value}")

    plot_distribution_by_source(combined)
    plot_text_length_distribution(combined)
    plot_timeline(combined)
    plot_missing_values(missing_before.sum(axis=1))
    plot_top_words(combined)
    print(f"Dataset guardado en: {OUTPUT_CSV}")
    print(f"Figuras guardadas en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
