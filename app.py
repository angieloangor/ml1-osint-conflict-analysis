#!/usr/bin/env python3
"""
OSINT Intelligence Center - Streamlit Dashboard
===============================================

Professional Streamlit interface for OSINT-style analysis of the
Iran-Israel-US conflict corpus. The app is intentionally defensive:
missing files are reported as analyst-friendly warnings instead of
breaking the dashboard.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import ast
import html as html_lib
import pickle
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
ADVANCED_FIGURES_DIR = OUTPUTS_DIR / "advanced_figures"
FIGURES_DIR = OUTPUTS_DIR / "figures"
ASSETS_DIR = BASE_DIR / "dashboard" / "assets"

NAV_ITEMS = [
    "Overview",
    "Geospatial Intelligence",
    "Timeline Analysis",
    "Semantic Explorer",
    "Topic Analysis",
    "ML Models",
    "Semantic Search",
    "News Explorer",
    "About / Methodology",
]

SOURCE_COLORS = {
    "google_news_rss": "#2563eb",
    "gdelt": "#7c3aed",
    "bbc_rss": "#0f766e",
    "aljazeera_rss": "#d97706",
    "NASA FIRMS": "#dc2626",
    "OpenSky": "#0284c7",
}

LABEL_COLORS = {
    "escalation": "#dc2626",
    "military": "#b91c1c",
    "diplomacy": "#0f766e",
    "energy": "#d97706",
    "humanitarian": "#16a34a",
    "sanctions": "#7c3aed",
    "cyber": "#0891b2",
    "other": "#64748b",
}

st.set_page_config(
    page_title="OSINT Intelligence Center",
    page_icon=":satellite:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "OSINT Intelligence Center for ML1 conflict analysis.",
    },
)


def load_css() -> None:
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    else:
        st.warning("Custom CSS was not found at dashboard/assets/styles.css. The app will continue with default styling.")


load_css()


# ============================================================================
# File and data loading helpers
# ============================================================================


def first_existing(candidates: Iterable[Path | str]) -> Path | None:
    for candidate in candidates:
        path = candidate if isinstance(candidate, Path) else BASE_DIR / candidate
        if path.exists():
            return path
    return None


def relative_path(path: Path | None) -> str:
    if path is None:
        return "missing"
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def elegant_warning(title: str, detail: str = "") -> None:
    detail_html = f"<p>{html_lib.escape(detail)}</p>" if detail else ""
    st.markdown(
        f"""
        <div class="notice warning">
            <strong>{html_lib.escape(title)}</strong>
            {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def prepare_dates(df: pd.DataFrame, candidates: Iterable[str] = ("timestamp", "date", "published", "acq_date")) -> pd.DataFrame:
    df = df.copy()
    for column in candidates:
        if column in df.columns:
            df[column] = normalize_datetime(df[column])
    return df


@st.cache_data(show_spinner=False)
def read_csv_cached(candidates: tuple[str, ...], date_columns: tuple[str, ...] = ("timestamp",)) -> pd.DataFrame | None:
    path = first_existing(candidates)
    if path is None:
        return None

    try:
        df = pd.read_csv(path)
        df = prepare_dates(df, date_columns)
        df.attrs["source_path"] = relative_path(path)
        return df
    except Exception as exc:
        elegant_warning(f"Could not load {relative_path(path)}", str(exc))
        return None


@st.cache_data(show_spinner=False)
def load_labeled_data() -> pd.DataFrame | None:
    df = read_csv_cached(
        (
            "data/dataset_nlp_labeled.csv",
            "outputs/models/dataset_nlp_enriched.csv",
            "data/dataset_nlp.csv",
        ),
        ("timestamp",),
    )
    if df is None:
        return None

    df = df.copy()
    if "conflict_related" in df.columns and not pd.api.types.is_bool_dtype(df["conflict_related"]):
        df["conflict_related"] = df["conflict_related"].astype(str).str.lower().isin(("true", "1", "yes", "y"))
    return df


@st.cache_data(show_spinner=False)
def load_bertopic_data() -> pd.DataFrame | None:
    return read_csv_cached(("data/dataset_nlp_bertopic.csv", "outputs/models/dataset_nlp_enriched.csv"), ("timestamp",))


@st.cache_data(show_spinner=False)
def load_document_data() -> pd.DataFrame | None:
    bertopic_df = load_bertopic_data()
    if bertopic_df is not None:
        return bertopic_df
    return load_labeled_data()


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame | None:
    return read_csv_cached(
        (
            "outputs/model_comparison_advanced.csv",
            "outputs/model_comparison.csv",
            "outputs/models/model_comparison.csv",
        ),
        (),
    )


@st.cache_data(show_spinner=False)
def load_bertopic_topics() -> pd.DataFrame | None:
    return read_csv_cached(
        (
            "outputs/models/bertopic_topics.csv",
            "outputs/models/topics.csv",
        ),
        (),
    )


@st.cache_data(show_spinner=False)
def load_embeddings() -> dict | None:
    path = first_existing(("data/dataset_nlp_embeddings.pkl", "outputs/models/dataset_nlp_embeddings.pkl"))
    if path is None:
        return None

    try:
        with path.open("rb") as file:
            data = pickle.load(file)
        if not isinstance(data, dict) or "df" not in data or "umap" not in data:
            elegant_warning("Embeddings file has an unexpected structure.", relative_path(path))
            return None
        return data
    except Exception as exc:
        elegant_warning(f"Could not load {relative_path(path)}", str(exc))
        return None


def normalize_geospatial_frame(df: pd.DataFrame | None, source_name: str) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    lat_col = next((column for column in ("lat", "latitude") if column in df.columns), None)
    lon_col = next((column for column in ("lon", "longitude", "long") if column in df.columns), None)
    if lat_col is None or lon_col is None:
        return None

    date_col = next((column for column in ("timestamp", "acq_date", "date", "time_position", "last_contact") if column in df.columns), None)
    value_col = next((column for column in ("value", "frp", "brightness", "velocity", "velocity_mps", "baro_altitude_m") if column in df.columns), None)
    confidence_col = next((column for column in ("confidence", "confidence_score", "probability") if column in df.columns), None)

    out = df.copy()
    out["_source_display"] = source_name
    out["_lat"] = pd.to_numeric(out[lat_col], errors="coerce")
    out["_lon"] = pd.to_numeric(out[lon_col], errors="coerce")
    out["_date"] = normalize_datetime(out[date_col]) if date_col else pd.NaT
    out["_value"] = pd.to_numeric(out[value_col], errors="coerce") if value_col else np.nan
    out["_confidence"] = out[confidence_col].astype(str) if confidence_col else "N/A"
    out["_title"] = out["title"].astype(str) if "title" in out.columns else source_name
    out["_url"] = out["url"].astype(str) if "url" in out.columns else ""
    out = out.dropna(subset=["_lat", "_lon"])
    return out


@st.cache_data(show_spinner=False)
def load_nasa_firms() -> pd.DataFrame | None:
    df = read_csv_cached(("data/nasa_firms.csv", "outputs/nasa_firms.csv"), ("timestamp", "acq_date"))
    return normalize_geospatial_frame(df, "NASA FIRMS")


@st.cache_data(show_spinner=False)
def load_opensky() -> pd.DataFrame | None:
    df = read_csv_cached(("data/opensky.csv", "outputs/opensky.csv"), ("timestamp", "time_position", "last_contact"))
    return normalize_geospatial_frame(df, "OpenSky")


@st.cache_data(show_spinner=False)
def load_geospatial_data() -> pd.DataFrame | None:
    frames = [frame for frame in (load_nasa_firms(), load_opensky()) if frame is not None and not frame.empty]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


@st.cache_resource(show_spinner=False)
def load_semantic_engine():
    path = first_existing(("outputs/models/faiss_index.pkl",))
    if path is None:
        return None, "outputs/models/faiss_index.pkl is missing."

    try:
        from scripts.semantic_search import SemanticSearchEngine

        engine = SemanticSearchEngine.load(path)
        return engine, None
    except Exception as exc:
        return None, str(exc)


@st.cache_resource(show_spinner=False)
def load_tfidf_fallback_search():
    df = load_document_data()
    if df is None or df.empty:
        return None, None, None, "No document corpus available for fallback search."

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        work = df.reset_index(drop=True).copy()
        text_parts = []
        for column in ("title", "text", "processed_text", "full_text"):
            if column in work.columns:
                text_parts.append(work[column].fillna("").astype(str))
        if not text_parts:
            return None, None, None, "No searchable text columns were found."

        corpus = text_parts[0]
        for part in text_parts[1:]:
            corpus = corpus + " " + part

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000, min_df=1)
        matrix = vectorizer.fit_transform(corpus)
        return work, vectorizer, matrix, None
    except Exception as exc:
        return None, None, None, str(exc)


def fallback_search(query: str, k: int) -> tuple[list[dict], str | None]:
    df, vectorizer, matrix, error = load_tfidf_fallback_search()
    if error or df is None or vectorizer is None or matrix is None:
        return [], error

    query_vector = vectorizer.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()
    if not np.any(scores):
        return [], None

    top_indices = np.argsort(scores)[::-1][:k]
    results = []
    for rank, idx in enumerate(top_indices, 1):
        row = df.iloc[int(idx)]
        results.append(
            {
                "rank": rank,
                "similarity": float(scores[idx]),
                "source": row.get("source", "N/A"),
                "title": row.get("title", "N/A"),
                "text": str(row.get("text", row.get("processed_text", "")))[:300],
                "timestamp": row.get("timestamp", "N/A"),
                "weak_label": row.get("weak_label", "N/A"),
                "url": row.get("url", ""),
            }
        )
    return results, None


# ============================================================================
# UI helpers
# ============================================================================


def apply_plot_style(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0f172a", family="Inter, system-ui, sans-serif"),
        margin=dict(l=20, r=20, t=62, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if height:
        fig.update_layout(height=height)
    return fig


def format_int(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


def format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def format_date(value) -> str:
    if pd.isna(value):
        return "N/A"
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def nice_model_name(name: str) -> str:
    text = re.sub(r"(?<!^)([A-Z])", r" \1", str(name)).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def kpi_card(title: str, value: str, subtitle: str = "", icon: str = "●", status: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="kpi-card kpi-{html_lib.escape(status)}">
            <div class="kpi-icon">{html_lib.escape(icon)}</div>
            <div>
                <div class="kpi-title">{html_lib.escape(title)}</div>
                <div class="kpi-value">{html_lib.escape(value)}</div>
                <div class="kpi-subtitle">{html_lib.escape(subtitle)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_title(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-title">
            <span class="eyebrow">OSINT Intelligence Center</span>
            <h1>{html_lib.escape(title)}</h1>
            <p>{html_lib.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(label: str, value: str, status: str = "neutral") -> str:
    return (
        f'<span class="status-pill status-{html_lib.escape(status)}">'
        f"<strong>{html_lib.escape(label)}</strong> {html_lib.escape(value)}</span>"
    )


def safe_unique(df: pd.DataFrame, column: str) -> list:
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().unique().tolist(), key=lambda item: str(item))


def parse_topic_words(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []

    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass

    return [part.strip(" '[]\"") for part in text.split(",") if part.strip(" '[]\"")]


def dataframe_with_links(df: pd.DataFrame, height: int = 460) -> None:
    column_config = {}
    if "url" in df.columns:
        column_config["url"] = st.column_config.LinkColumn("URL", display_text="open")
    st.dataframe(df, width="stretch", hide_index=True, height=height, column_config=column_config)


def build_executive_summary(df: pd.DataFrame, model_df: pd.DataFrame | None, topics_df: pd.DataFrame | None) -> list[str]:
    summary: list[str] = []
    docs = len(df)
    source_count = df["source"].nunique() if "source" in df.columns else 0
    top_source = df["source"].value_counts().idxmax() if "source" in df.columns and not df.empty else "N/A"
    top_label = df["weak_label"].value_counts().idxmax() if "weak_label" in df.columns and not df.empty else "N/A"

    summary.append(
        f"The current corpus contains {format_int(docs)} documents from {source_count} sources. "
        f"The dominant feed is {top_source}, so source concentration should be considered when interpreting trends."
    )

    if "timestamp" in df.columns:
        dates = df["timestamp"].dropna()
        if not dates.empty:
            summary.append(
                f"Temporal coverage runs from {format_date(dates.min())} to {format_date(dates.max())}, "
                f"with the strongest weak-label signal concentrated around '{top_label}'."
            )

    if "conflict_related" in df.columns:
        conflict_docs = int(df["conflict_related"].fillna(False).sum())
        summary.append(
            f"{format_int(conflict_docs)} documents are marked conflict-related by the pipeline, "
            "which makes the dashboard suitable for thematic monitoring but not for ground-truth event attribution."
        )

    if model_df is not None and {"model", "accuracy", "f1_macro"}.issubset(model_df.columns):
        best = model_df.sort_values(["accuracy", "f1_macro"], ascending=False).iloc[0]
        summary.append(
            f"The strongest supervised model is {nice_model_name(best['model'])} "
            f"with {format_pct(best['accuracy'])} accuracy and {format_pct(best['f1_macro'])} macro-F1."
        )

    if topics_df is not None and not topics_df.empty:
        topic_col = "Topic" if "Topic" in topics_df.columns else topics_df.columns[0]
        topic_count = int((topics_df[topic_col] != -1).sum()) if topic_col in topics_df.columns else len(topics_df)
        summary.append(
            f"BERTopic identifies {topic_count} interpretable topic groups plus any outlier bucket, "
            "useful for analyst review and narrative clustering."
        )

    return summary


# ============================================================================
# Shell and navigation
# ============================================================================


def render_header() -> None:
    st.markdown(
        """
        <div class="main-header">
            <div>
                <span class="system-tag">OSINT Intelligence Center</span>
                <h1>Iran-Israel-US Conflict Monitor</h1>
                <p>Semantic intelligence, geospatial context, weak-label classification and model diagnostics.</p>
            </div>
            <div class="header-badge">
                <span>Live local analysis</span>
                <strong>ML1 Project</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <span class="brand-mark">OSINT</span>
            <div>
                <strong>Intelligence Center</strong>
                <small>Conflict analytics workspace</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    requested_page = st.query_params.get("page", NAV_ITEMS[0])
    if isinstance(requested_page, list):
        requested_page = requested_page[0] if requested_page else NAV_ITEMS[0]
    page_index = NAV_ITEMS.index(requested_page) if requested_page in NAV_ITEMS else 0

    page = st.sidebar.radio("Navigation", NAV_ITEMS, index=page_index, label_visibility="collapsed")
    if st.query_params.get("page") != page:
        st.query_params["page"] = page

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Status")
    watched_files = {
        "Corpus": first_existing(("data/dataset_nlp_labeled.csv", "data/dataset_nlp.csv")),
        "BERTopic": first_existing(("data/dataset_nlp_bertopic.csv", "outputs/models/bertopic_topics.csv")),
        "Embeddings": first_existing(("data/dataset_nlp_embeddings.pkl",)),
        "Models": first_existing(("outputs/model_comparison_advanced.csv", "outputs/model_comparison.csv")),
        "Geo": first_existing(("data/nasa_firms.csv", "data/opensky.csv")),
    }
    for label, path in watched_files.items():
        status = "ready" if path else "missing"
        icon = "OK" if path else "NA"
        st.sidebar.markdown(
            f"<div class='file-status {status}'><span>{icon}</span><strong>{label}</strong><small>{relative_path(path)}</small></div>",
            unsafe_allow_html=True,
        )

    df = load_document_data()
    model_df = load_model_comparison()
    if df is not None:
        st.sidebar.markdown("---")
        st.sidebar.metric("Documents", format_int(len(df)))
        if "source" in df.columns:
            st.sidebar.metric("Sources", df["source"].nunique())
        if "weak_label" in df.columns:
            st.sidebar.metric("Weak labels", df["weak_label"].nunique())
    if model_df is not None and "accuracy" in model_df.columns:
        st.sidebar.metric("Best accuracy", format_pct(float(model_df["accuracy"].max())))

    return page


# ============================================================================
# Pages
# ============================================================================


def render_overview() -> None:
    page_title("Overview", "Executive view of corpus coverage, model readiness and conflict signals.")

    df = load_document_data()
    labeled_df = load_labeled_data()
    model_df = load_model_comparison()
    topics_df = load_bertopic_topics()

    if df is None:
        elegant_warning("No corpus file found.", "Expected data/dataset_nlp_labeled.csv or data/dataset_nlp.csv.")
        return

    source_count = df["source"].nunique() if "source" in df.columns else 0
    source_subtitle = ", ".join(safe_unique(df, "source")[:3])
    if source_count > 3:
        source_subtitle += f" +{source_count - 3}"

    if "timestamp" in df.columns and df["timestamp"].notna().any():
        min_date = df["timestamp"].min()
        max_date = df["timestamp"].max()
        period_value = f"{(max_date - min_date).days} days"
        period_subtitle = f"{format_date(min_date)} to {format_date(max_date)}"
    else:
        period_value = "N/A"
        period_subtitle = "No timestamp column"

    if model_df is not None and {"model", "accuracy", "f1_macro"}.issubset(model_df.columns):
        best_model = model_df.sort_values(["accuracy", "f1_macro"], ascending=False).iloc[0]
        best_model_value = nice_model_name(best_model["model"])
        best_model_subtitle = f"{format_pct(best_model['accuracy'])} acc | {format_pct(best_model['f1_macro'])} macro-F1"
    else:
        best_model_value = "Missing"
        best_model_subtitle = "Run scripts/advanced_ml.py"

    if topics_df is not None and not topics_df.empty:
        topic_col = "Topic" if "Topic" in topics_df.columns else topics_df.columns[0]
        topic_value = format_int(int((topics_df[topic_col] != -1).sum()))
        topic_subtitle = "BERTopic topics, excluding outlier bucket"
    elif "topic_id" in df.columns:
        topic_value = format_int(df["topic_id"].nunique())
        topic_subtitle = "Pipeline topic_id groups"
    else:
        topic_value = "Missing"
        topic_subtitle = "Run topic pipeline"

    conflict_count = int(df["conflict_related"].fillna(False).sum()) if "conflict_related" in df.columns else len(df)

    top_row = st.columns(3)
    with top_row[0]:
        kpi_card("Total documents", format_int(len(df)), "Processed intelligence corpus", "DOC", "neutral")
    with top_row[1]:
        kpi_card("Sources used", format_int(source_count), source_subtitle or "No source column", "SRC", "neutral")
    with top_row[2]:
        kpi_card("Temporal period", period_value, period_subtitle, "TIME", "neutral")

    second_row = st.columns(3)
    with second_row[0]:
        kpi_card("Best model", best_model_value, best_model_subtitle, "ML", "success" if model_df is not None else "alert")
    with second_row[1]:
        kpi_card("Topic groups", topic_value, topic_subtitle, "TOPIC", "neutral")
    with second_row[2]:
        kpi_card("Conflict-related docs", format_int(conflict_count), f"{format_pct(conflict_count / max(len(df), 1))} of corpus", "ALERT", "alert")

    st.markdown("### Executive Summary")
    summary_items = build_executive_summary(labeled_df if labeled_df is not None else df, model_df, topics_df)
    st.markdown(
        "<div class='intel-brief'>" + "".join(f"<p>{html_lib.escape(item)}</p>" for item in summary_items) + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Intelligence Snapshot")
    chart_cols = st.columns([1.1, 1, 1.2])

    if "source" in df.columns:
        source_counts = df["source"].value_counts().reset_index()
        source_counts.columns = ["source", "documents"]
        fig = px.bar(
            source_counts,
            x="documents",
            y="source",
            orientation="h",
            color="source",
            color_discrete_map=SOURCE_COLORS,
            title="Documents by source",
        )
        fig.update_layout(showlegend=False)
        with chart_cols[0]:
            st.plotly_chart(apply_plot_style(fig, 360), width="stretch")

    if "weak_label" in df.columns:
        label_counts = df["weak_label"].value_counts().reset_index()
        label_counts.columns = ["weak_label", "documents"]
        fig = px.bar(
            label_counts,
            x="weak_label",
            y="documents",
            color="weak_label",
            color_discrete_map=LABEL_COLORS,
            title="Weak-label distribution",
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-35)
        with chart_cols[1]:
            st.plotly_chart(apply_plot_style(fig, 360), width="stretch")

    if "timestamp" in df.columns:
        timeline = df.dropna(subset=["timestamp"]).copy()
        timeline["date"] = timeline["timestamp"].dt.tz_convert(None).dt.floor("D")
        timeline_counts = timeline.groupby("date").size().reset_index(name="documents")
        fig = px.line(timeline_counts, x="date", y="documents", markers=True, title="Daily document activity")
        fig.update_traces(line_color="#0f766e", marker_color="#0f766e")
        with chart_cols[2]:
            st.plotly_chart(apply_plot_style(fig, 360), width="stretch")

    with st.expander("Corpus health and caveats", expanded=False):
        corpus_path = df.attrs.get("source_path", "data source")
        st.markdown(
            f"""
            {status_pill("Loaded", corpus_path, "ready")}
            {status_pill("Weak labels", "heuristic", "warning")}
            {status_pill("Geo context", "separate from text corpus", "neutral")}
            """,
            unsafe_allow_html=True,
        )
        st.write("The dashboard is optimized for exploration and academic analysis. Weak labels are not human-validated ground truth.")


def render_geospatial() -> None:
    page_title("Geospatial Intelligence", "NASA FIRMS hotspots and OpenSky aircraft positions as contextual overlays.")

    geo_df = load_geospatial_data()
    if geo_df is None or geo_df.empty:
        elegant_warning(
            "No geospatial points available.",
            "Expected data/nasa_firms.csv and/or data/opensky.csv with lat/lon or latitude/longitude columns.",
        )
        return

    filter_cols = st.columns([1, 1.25, 1, 1])
    with filter_cols[0]:
        selected_sources = st.multiselect(
            "Source",
            safe_unique(geo_df, "_source_display"),
            default=safe_unique(geo_df, "_source_display"),
        )

    dated = geo_df.dropna(subset=["_date"])
    date_range = None
    with filter_cols[1]:
        if not dated.empty:
            min_date = dated["_date"].min().date()
            max_date = dated["_date"].max().date()
            date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        else:
            st.caption("No date column detected for geospatial filtering.")

    with filter_cols[2]:
        map_style = st.selectbox("Map style", ["carto-darkmatter", "open-street-map", "carto-positron"], index=0)
    with filter_cols[3]:
        show_heatmap = st.checkbox("Heatmap layer", value=len(geo_df) >= 120, help="Adds a density layer when many points are visible.")

    filtered = geo_df[geo_df["_source_display"].isin(selected_sources)].copy()
    if date_range and len(date_range) == 2 and filtered["_date"].notna().any():
        start, end = date_range
        filtered = filtered[(filtered["_date"].dt.date >= start) & (filtered["_date"].dt.date <= end)]

    if filtered.empty:
        elegant_warning("No geospatial observations match the current filters.")
        return

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Visible points", format_int(len(filtered)))
    with metric_cols[1]:
        st.metric("NASA FIRMS", format_int((filtered["_source_display"] == "NASA FIRMS").sum()))
    with metric_cols[2]:
        st.metric("OpenSky", format_int((filtered["_source_display"] == "OpenSky").sum()))
    with metric_cols[3]:
        value_label = "Avg value"
        st.metric(value_label, f"{filtered['_value'].mean():.2f}" if filtered["_value"].notna().any() else "N/A")

    fig = go.Figure()

    if show_heatmap and len(filtered) >= 20:
        fig.add_trace(
            go.Densitymapbox(
                lat=filtered["_lat"],
                lon=filtered["_lon"],
                z=filtered["_value"].fillna(1),
                radius=22,
                colorscale=[[0, "rgba(37, 99, 235, 0.05)"], [0.5, "rgba(217, 119, 6, 0.45)"], [1, "rgba(220, 38, 38, 0.8)"]],
                showscale=False,
                name="Density",
                hoverinfo="skip",
            )
        )

    for source, source_df in filtered.groupby("_source_display"):
        values = source_df["_value"].fillna(source_df["_value"].median() if source_df["_value"].notna().any() else 1)
        marker_size = np.clip(np.sqrt(np.maximum(values, 0) + 1) * 4.5, 7, 20)
        customdata = np.stack(
            [
                source_df["_source_display"].astype(str),
                source_df["_date"].dt.strftime("%Y-%m-%d %H:%M").fillna("N/A"),
                source_df["_value"].round(3).astype(str).replace("nan", "N/A"),
                source_df["_confidence"].astype(str),
                source_df["_title"].astype(str).str.slice(0, 110),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scattermapbox(
                lat=source_df["_lat"],
                lon=source_df["_lon"],
                mode="markers",
                marker=dict(size=marker_size, color=SOURCE_COLORS.get(source, "#64748b"), opacity=0.78),
                name=source,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Date: %{customdata[1]}<br>"
                    "Value/confidence: %{customdata[2]} / %{customdata[3]}<br>"
                    "%{customdata[4]}<extra></extra>"
                ),
            )
        )

    center = {"lat": float(filtered["_lat"].median()), "lon": float(filtered["_lon"].median())}
    fig.update_layout(
        mapbox=dict(style=map_style, center=center, zoom=4.1),
        height=650,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", y=0.99, x=0.01, bgcolor="rgba(255,255,255,0.72)"),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Geospatial Activity Timeline")
    if filtered["_date"].notna().any():
        geo_timeline = filtered.dropna(subset=["_date"]).copy()
        geo_timeline["date"] = geo_timeline["_date"].dt.tz_convert(None).dt.floor("D")
        counts = geo_timeline.groupby(["date", "_source_display"]).size().reset_index(name="observations")
        fig = px.bar(
            counts,
            x="date",
            y="observations",
            color="_source_display",
            color_discrete_map=SOURCE_COLORS,
            title="Daily geospatial observations",
        )
        st.plotly_chart(apply_plot_style(fig, 360), width="stretch")
    else:
        st.info("No usable date field was found for geospatial timeline aggregation.")


def render_timeline() -> None:
    page_title("Timeline Analysis", "Interactive source and weak-label activity with spike detection.")

    df = load_document_data()
    if df is None or "timestamp" not in df.columns:
        elegant_warning("Timeline requires a corpus with a timestamp column.")
        return

    work = df.dropna(subset=["timestamp"]).copy()
    if work.empty:
        elegant_warning("No valid timestamps were found in the corpus.")
        return

    filters = st.columns([1, 1, 1, 1])
    with filters[0]:
        selected_sources = st.multiselect("Sources", safe_unique(work, "source"), default=safe_unique(work, "source"))
    with filters[1]:
        selected_labels = st.multiselect("Weak labels", safe_unique(work, "weak_label"), default=safe_unique(work, "weak_label"))
    with filters[2]:
        min_date = work["timestamp"].min().date()
        max_date = work["timestamp"].max().date()
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    with filters[3]:
        granularity = st.radio("Granularity", ["Daily", "Weekly"], horizontal=True)

    if selected_sources and "source" in work.columns:
        work = work[work["source"].isin(selected_sources)]
    if selected_labels and "weak_label" in work.columns:
        work = work[work["weak_label"].isin(selected_labels)]
    if len(date_range) == 2:
        work = work[(work["timestamp"].dt.date >= date_range[0]) & (work["timestamp"].dt.date <= date_range[1])]

    if work.empty:
        elegant_warning("No documents match the selected timeline filters.")
        return

    naive_time = work["timestamp"].dt.tz_convert(None)
    if granularity == "Weekly":
        work["_period"] = naive_time.dt.to_period("W").dt.start_time
    else:
        work["_period"] = naive_time.dt.floor("D")

    source_counts = work.groupby(["_period", "source"]).size().reset_index(name="documents")
    fig = px.line(
        source_counts,
        x="_period",
        y="documents",
        color="source",
        markers=True,
        color_discrete_map=SOURCE_COLORS,
        title=f"{granularity} document activity by source",
    )

    total_counts = work.groupby("_period").size().reset_index(name="documents")
    threshold = float(total_counts["documents"].mean() + max(total_counts["documents"].std(ddof=0) * 1.35, 2))
    spikes = total_counts[total_counts["documents"] >= threshold]
    if not spikes.empty:
        fig.add_trace(
            go.Scatter(
                x=spikes["_period"],
                y=spikes["documents"],
                mode="markers+text",
                marker=dict(size=13, color="#dc2626", symbol="diamond"),
                text=["spike"] * len(spikes),
                textposition="top center",
                name="Activity spike",
                hovertemplate="Spike<br>%{x}<br>%{y} documents<extra></extra>",
            )
        )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(apply_plot_style(fig, 520), width="stretch")

    lower_cols = st.columns([1.25, 1])
    with lower_cols[0]:
        if "weak_label" in work.columns:
            label_counts = work.groupby(["_period", "weak_label"]).size().reset_index(name="documents")
            fig = px.area(
                label_counts,
                x="_period",
                y="documents",
                color="weak_label",
                color_discrete_map=LABEL_COLORS,
                title="Weak-label composition over time",
            )
            st.plotly_chart(apply_plot_style(fig, 420), width="stretch")

    with lower_cols[1]:
        st.markdown("### Spike Register")
        if spikes.empty:
            st.info("No spike crossed the current threshold.")
        else:
            spike_table = spikes.sort_values("documents", ascending=False).copy()
            spike_table["_period"] = spike_table["_period"].dt.strftime("%Y-%m-%d")
            st.dataframe(spike_table.rename(columns={"_period": "period"}), width="stretch", hide_index=True)
        st.metric("Spike threshold", f"{threshold:.1f} docs/{granularity.lower()[:-2] if granularity == 'Daily' else 'week'}")
        st.metric("Average activity", f"{total_counts['documents'].mean():.1f}")


def render_semantic_explorer() -> None:
    page_title("Semantic Explorer", "Interactive UMAP projection of document embeddings.")

    embeddings_data = load_embeddings()
    if embeddings_data is None:
        elegant_warning("Embeddings are not available.", "Expected data/dataset_nlp_embeddings.pkl. Run scripts/semantic_embeddings.py.")
        return

    df = embeddings_data["df"].copy()
    umap_coords = np.asarray(embeddings_data["umap"])
    if len(df) != len(umap_coords):
        elegant_warning("Embedding metadata and UMAP coordinates have different lengths.")
        return

    color_options = [column for column in ("weak_label", "source", "topic_id", "cluster_id", "bertopic_id") if column in df.columns]
    if not color_options:
        elegant_warning("No categorical columns were found for coloring the semantic projection.")
        return

    controls = st.columns([1, 1.2, 1])
    with controls[0]:
        color_by = st.selectbox("Color by", color_options, index=0)
    with controls[1]:
        selected_values = st.multiselect(f"Filter {color_by}", safe_unique(df, color_by), default=safe_unique(df, color_by))
    with controls[2]:
        keyword = st.text_input("Keyword", placeholder="Filter title or text")

    mask = df[color_by].isin(selected_values)
    if keyword:
        keyword_mask = pd.Series(False, index=df.index)
        for column in ("title", "text", "processed_text"):
            if column in df.columns:
                keyword_mask = keyword_mask | df[column].astype(str).str.contains(keyword, case=False, na=False)
        mask = mask & keyword_mask

    plot_df = df[mask].copy()
    plot_coords = umap_coords[mask.to_numpy()]
    if plot_df.empty:
        elegant_warning("No semantic points match the current filters.")
        return

    plot_df["UMAP 1"] = plot_coords[:, 0]
    plot_df["UMAP 2"] = plot_coords[:, 1]
    size_col = "topic_weight" if "topic_weight" in plot_df.columns else None
    if size_col:
        plot_df["topic_weight_size"] = pd.to_numeric(plot_df[size_col], errors="coerce").fillna(0.05).clip(lower=0.03)

    hover_cols = [column for column in ("title", "source", "weak_label", "topic_id", "cluster_id") if column in plot_df.columns]
    fig = px.scatter(
        plot_df,
        x="UMAP 1",
        y="UMAP 2",
        color=color_by,
        size="topic_weight_size" if size_col else None,
        size_max=18,
        hover_data=hover_cols,
        color_discrete_map={**SOURCE_COLORS, **LABEL_COLORS},
        title=f"Semantic space: {format_int(len(plot_df))} visible documents",
    )
    fig.update_traces(marker=dict(opacity=0.76, line=dict(width=0.4, color="white")))
    st.plotly_chart(apply_plot_style(fig, 650), width="stretch")

    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("Visible documents", format_int(len(plot_df)))
    with stat_cols[1]:
        st.metric("Labels", plot_df["weak_label"].nunique() if "weak_label" in plot_df.columns else "N/A")
    with stat_cols[2]:
        st.metric("Clusters", plot_df["cluster_id"].nunique() if "cluster_id" in plot_df.columns else "N/A")
    with stat_cols[3]:
        st.metric("Size field", "topic_weight" if size_col else "fixed")


def render_topic_analysis() -> None:
    page_title("Topic Analysis", "BERTopic themes, top words and document distribution.")

    topics_df = load_bertopic_topics()
    bertopic_df = load_bertopic_data()
    if topics_df is None or topics_df.empty:
        elegant_warning("BERTopic topic table is not available.", "Expected outputs/models/bertopic_topics.csv.")
        return

    topics = topics_df.copy()
    topic_col = "Topic" if "Topic" in topics.columns else topics.columns[0]
    count_col = "Count" if "Count" in topics.columns else "doc_count" if "doc_count" in topics.columns else None

    metrics = st.columns(4)
    with metrics[0]:
        topic_count = int((topics[topic_col] != -1).sum()) if topic_col in topics.columns else len(topics)
        st.metric("Interpretable topics", topic_count)
    with metrics[1]:
        if count_col:
            st.metric("Largest topic", format_int(topics[count_col].max()))
    with metrics[2]:
        if bertopic_df is not None:
            st.metric("Documents with topics", format_int(len(bertopic_df)))
    with metrics[3]:
        outliers = int((topics[topic_col] == -1).sum()) if topic_col in topics.columns else 0
        st.metric("Outlier buckets", outliers)

    display_cols = [column for column in (topic_col, count_col, "Name", "Representation") if column and column in topics.columns]
    topic_table = topics[display_cols].copy()
    if "Representation" in topic_table.columns:
        topic_table["Top words"] = topic_table["Representation"].apply(lambda value: ", ".join(parse_topic_words(value)[:8]))
        topic_table = topic_table.drop(columns=["Representation"])
    st.dataframe(topic_table, width="stretch", hide_index=True, height=340)

    detail_cols = st.columns([1, 1.2])
    with detail_cols[0]:
        selected_topic = st.selectbox("Inspect topic", topics[topic_col].tolist())
        selected_row = topics[topics[topic_col] == selected_topic].iloc[0]
        st.markdown("### Top Words")
        words = parse_topic_words(selected_row.get("Representation", ""))
        if words:
            chips = "".join(f"<span class='word-chip'>{html_lib.escape(word)}</span>" for word in words[:12])
            st.markdown(f"<div class='word-chip-row'>{chips}</div>", unsafe_allow_html=True)
        else:
            st.info("No representation words found for this topic.")

        if "Representative_Docs" in selected_row.index:
            docs = parse_topic_words(selected_row["Representative_Docs"])[:3]
            with st.expander("Representative documents", expanded=False):
                for doc in docs:
                    st.write(doc)

    with detail_cols[1]:
        if bertopic_df is not None:
            dist_col = "bertopic_id" if "bertopic_id" in bertopic_df.columns else "topic_id" if "topic_id" in bertopic_df.columns else None
            if dist_col:
                dist = bertopic_df[dist_col].value_counts().sort_index().reset_index()
                dist.columns = ["topic", "documents"]
                fig = px.bar(dist, x="topic", y="documents", color="documents", color_continuous_scale="Tealgrn", title="Documents per topic")
                st.plotly_chart(apply_plot_style(fig, 410), width="stretch")
            else:
                st.info("No topic id column was found in the enriched corpus.")

    html_path = ADVANCED_FIGURES_DIR / "bertopic_topics.html"
    if html_path.exists():
        st.markdown("### Interactive BERTopic View")
        html_content = html_path.read_text(encoding="utf-8", errors="ignore")
        st.iframe(html_path, height=650)
        st.download_button("Download BERTopic HTML", html_content, file_name="bertopic_topics.html", mime="text/html")
    else:
        elegant_warning("Interactive BERTopic HTML was not found.", relative_path(html_path))


def confusion_matrix_path(model_name: str) -> Path | None:
    raw = str(model_name)
    slugs = {
        raw.lower(),
        raw.lower().replace(" ", "_"),
        re.sub(r"[^a-z0-9]+", "", raw.lower()),
        re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_"),
    }
    aliases = {
        "logisticregression": ["logreg"],
        "multinomialnb": ["nb"],
        "kneighborsclassifier": ["knn"],
    }
    for slug in list(slugs):
        slugs.update(aliases.get(slug, []))
    candidates = []
    for directory in (ADVANCED_FIGURES_DIR, FIGURES_DIR):
        for slug in slugs:
            candidates.append(directory / f"confusion_matrix_{slug}.png")
    return first_existing(candidates)


def render_ml_models() -> None:
    page_title("ML Models", "Comparative model performance and confusion-matrix diagnostics.")

    model_df = load_model_comparison()
    if model_df is None or model_df.empty:
        elegant_warning("Model comparison table is not available.", "Expected outputs/model_comparison_advanced.csv.")
        return

    mc = model_df.copy()
    st.markdown("### Model Leaderboard")
    display = mc.copy()
    if "model" in display.columns:
        display["model"] = display["model"].map(nice_model_name)
    st.dataframe(display.round(4), width="stretch", hide_index=True)

    metric_cols = [column for column in ("accuracy", "f1_macro") if column in mc.columns]
    if "model" in mc.columns and metric_cols:
        long = mc.melt(id_vars="model", value_vars=metric_cols, var_name="metric", value_name="score")
        long["model"] = long["model"].map(nice_model_name)
        fig = px.bar(
            long,
            x="model",
            y="score",
            color="metric",
            barmode="group",
            text=long["score"].map(lambda value: f"{value:.2f}"),
            color_discrete_map={"accuracy": "#2563eb", "f1_macro": "#0f766e"},
            title="Accuracy and macro-F1 by model",
        )
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(apply_plot_style(fig, 470), width="stretch")

    st.markdown("### Confusion Matrices")
    tabs = st.tabs([nice_model_name(name) for name in mc["model"].tolist()]) if "model" in mc.columns else []
    for tab, model_name in zip(tabs, mc["model"].tolist()):
        with tab:
            cm_path = confusion_matrix_path(model_name)
            if cm_path:
                st.image(str(cm_path), caption=f"Confusion matrix: {nice_model_name(model_name)}", width="stretch")
            else:
                elegant_warning("Confusion matrix not found.", f"No PNG found for {model_name} in outputs/advanced_figures or outputs/figures.")

    st.markdown("### Automatic Interpretation")
    if {"model", "accuracy", "f1_macro"}.issubset(mc.columns):
        best_acc = mc.sort_values("accuracy", ascending=False).iloc[0]
        best_f1 = mc.sort_values("f1_macro", ascending=False).iloc[0]
        label_df = load_document_data()
        imbalance_note = ""
        if label_df is not None and "weak_label" in label_df.columns:
            top_share = label_df["weak_label"].value_counts(normalize=True).max()
            imbalance_note = f"The largest weak-label class represents {format_pct(top_share)} of documents, so macro-F1 is the key fairness-oriented metric."

        st.markdown(
            f"""
            <div class="intel-brief">
                <p><strong>{html_lib.escape(nice_model_name(best_acc['model']))}</strong> leads by accuracy at {format_pct(best_acc['accuracy'])}.</p>
                <p><strong>{html_lib.escape(nice_model_name(best_f1['model']))}</strong> leads by macro-F1 at {format_pct(best_f1['f1_macro'])}, which is important for minority labels.</p>
                <p>{html_lib.escape(imbalance_note or "Macro-F1 should be read alongside accuracy because the labels are weak and imbalanced.")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_semantic_search() -> None:
    page_title("Semantic Search", "Retrieve documents by embedding similarity instead of keyword overlap.")

    index_path = first_existing(("outputs/models/faiss_index.pkl",))
    if index_path is None:
        elegant_warning("Semantic search engine is not available.", "Expected outputs/models/faiss_index.pkl. Run scripts/semantic_search.py.")
        return

    controls = st.columns([2.5, 1, 1])
    with controls[0]:
        default_query = st.query_params.get("q", "")
        if isinstance(default_query, list):
            default_query = default_query[0] if default_query else ""
        query = st.text_input(
            "Search query",
            value=default_query,
            placeholder="Example: Israeli strikes against Iranian nuclear facilities",
        )
    with controls[1]:
        k = st.slider("Top results", 3, 20, 8)
    with controls[2]:
        st.caption("Semantic search uses the saved FAISS index and sentence-transformer query embeddings.")

    if not query:
        st.info("Enter a query to retrieve the most semantically similar documents.")
        return

    search_mode = "FAISS semantic search"
    engine, error = load_semantic_engine()
    if engine is not None:
        try:
            with st.spinner("Searching semantic index..."):
                results = engine.search(query, k=k)
        except Exception as exc:
            results = []
            error = str(exc)
    else:
        results = []

    if not results:
        search_mode = "TF-IDF fallback search"
        if error:
            elegant_warning("Semantic model unavailable; using local TF-IDF fallback.", error)
        with st.spinner("Searching local TF-IDF index..."):
            results, fallback_error = fallback_search(query, k=k)
        if fallback_error:
            elegant_warning("Fallback search failed.", fallback_error)
            return

    if not results:
        st.info("No semantic results returned.")
        return

    st.caption(f"Search mode: {search_mode}")
    result_df = pd.DataFrame(results)
    if "timestamp" in result_df.columns:
        result_df["timestamp"] = pd.to_datetime(result_df["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d")
    table_cols = [column for column in ("rank", "similarity", "source", "timestamp", "weak_label", "title", "url") if column in result_df.columns]
    result_table = result_df[table_cols].copy()
    if "similarity" in result_table.columns:
        result_table["similarity"] = result_table["similarity"].round(4)
    dataframe_with_links(result_table, height=330)

    st.markdown("### Result Details")
    for result in results:
        with st.expander(f"#{result.get('rank')} | {result.get('similarity', 0):.3f} | {result.get('title', 'Untitled')}"):
            cols = st.columns([1, 1, 1])
            cols[0].markdown(f"**Source:** {result.get('source', 'N/A')}")
            cols[1].markdown(f"**Date:** {format_date(result.get('timestamp'))}")
            cols[2].markdown(f"**Label:** {result.get('weak_label', 'N/A')}")
            url = result.get("url")
            if url and str(url).startswith("http"):
                st.markdown(f"**URL:** [{url}]({url})")
            st.write(result.get("text", ""))


def render_news_explorer() -> None:
    page_title("News Explorer", "Filterable document table with source, label, topic and cluster controls.")

    df = load_document_data()
    if df is None or df.empty:
        elegant_warning("No document corpus found.")
        return

    work = df.copy()
    filters = st.columns([1, 1, 1, 1])
    with filters[0]:
        selected_sources = st.multiselect("Source", safe_unique(work, "source"), default=safe_unique(work, "source"))
    with filters[1]:
        selected_labels = st.multiselect("Weak label", safe_unique(work, "weak_label"), default=safe_unique(work, "weak_label"))
    with filters[2]:
        topic_column = "bertopic_id" if "bertopic_id" in work.columns else "topic_id" if "topic_id" in work.columns else None
        selected_topics = st.multiselect("Topic ID", safe_unique(work, topic_column), default=safe_unique(work, topic_column)) if topic_column else []
    with filters[3]:
        selected_clusters = st.multiselect("Cluster ID", safe_unique(work, "cluster_id"), default=safe_unique(work, "cluster_id")) if "cluster_id" in work.columns else []

    search_cols = st.columns([1.6, 1])
    with search_cols[0]:
        keyword = st.text_input("Keyword search", placeholder="Search in title, text or processed_text")
    with search_cols[1]:
        if "timestamp" in work.columns and work["timestamp"].notna().any():
            min_date = work["timestamp"].min().date()
            max_date = work["timestamp"].max().date()
            date_range = st.date_input("Publication window", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        else:
            date_range = None

    if selected_sources and "source" in work.columns:
        work = work[work["source"].isin(selected_sources)]
    if selected_labels and "weak_label" in work.columns:
        work = work[work["weak_label"].isin(selected_labels)]
    if topic_column and selected_topics:
        work = work[work[topic_column].isin(selected_topics)]
    if selected_clusters:
        work = work[work["cluster_id"].isin(selected_clusters)]
    if date_range and len(date_range) == 2 and "timestamp" in work.columns:
        work = work[(work["timestamp"].dt.date >= date_range[0]) & (work["timestamp"].dt.date <= date_range[1])]
    if keyword:
        keyword_mask = pd.Series(False, index=work.index)
        for column in ("title", "text", "processed_text", "full_text"):
            if column in work.columns:
                keyword_mask = keyword_mask | work[column].astype(str).str.contains(keyword, case=False, na=False)
        work = work[keyword_mask]

    summary_cols = st.columns(4)
    summary_cols[0].metric("Filtered docs", format_int(len(work)))
    summary_cols[1].metric("Sources", work["source"].nunique() if "source" in work.columns and not work.empty else 0)
    summary_cols[2].metric("Labels", work["weak_label"].nunique() if "weak_label" in work.columns and not work.empty else 0)
    summary_cols[3].metric("Topics", work[topic_column].nunique() if topic_column and not work.empty else "N/A")

    if work.empty:
        elegant_warning("No documents match the selected filters.")
        return

    table_cols = [column for column in ("timestamp", "source", "weak_label", topic_column, "cluster_id", "title", "url") if column and column in work.columns]
    table = work[table_cols].copy().sort_values("timestamp" if "timestamp" in table_cols else table_cols[0], ascending=False)
    if "timestamp" in table.columns:
        table["timestamp"] = table["timestamp"].dt.strftime("%Y-%m-%d")
    dataframe_with_links(table.head(300), height=520)

    if len(table) > 300:
        st.info(f"Showing the first 300 of {len(table)} filtered documents. Tighten filters for a narrower view.")

    selected_index = st.selectbox("Inspect document", work.index.tolist(), format_func=lambda idx: str(work.loc[idx].get("title", "Untitled"))[:120])
    row = work.loc[selected_index]
    with st.expander("Document detail", expanded=True):
        st.markdown(f"**Title:** {row.get('title', 'N/A')}")
        st.markdown(f"**Source:** {row.get('source', 'N/A')} | **Label:** {row.get('weak_label', 'N/A')} | **Topic:** {row.get(topic_column, 'N/A') if topic_column else 'N/A'}")
        if str(row.get("url", "")).startswith("http"):
            st.markdown(f"**URL:** [{row.get('url')}]({row.get('url')})")
        st.write(row.get("text", row.get("full_text", "")))

    st.download_button(
        "Download filtered CSV",
        work.to_csv(index=False),
        file_name="osint_filtered_documents.csv",
        mime="text/csv",
    )


def render_about() -> None:
    page_title("About / Methodology", "Sources, pipeline, assumptions and analytical limitations.")

    df = load_document_data()
    source_text = "BBC RSS, Al Jazeera RSS, Google News RSS, GDELT"
    if df is not None and "source" in df.columns:
        source_text = ", ".join(safe_unique(df, "source"))

    st.markdown(
        f"""
        <div class="method-grid">
            <section>
                <h3>Sources</h3>
                <p>{html_lib.escape(source_text)} feed the text corpus. NASA FIRMS and OpenSky are loaded separately as geospatial context.</p>
            </section>
            <section>
                <h3>Pipeline</h3>
                <p>Collection, cleaning, weak labeling, TF-IDF features, semantic embeddings, BERTopic, supervised ML and FAISS search.</p>
            </section>
            <section>
                <h3>Weak Labels</h3>
                <p>Labels such as escalation, military, diplomacy, sanctions, humanitarian, energy and cyber are generated with heuristic rules.</p>
            </section>
            <section>
                <h3>Geospatial Context</h3>
                <p>NASA FIRMS hotspots and OpenSky aircraft points provide situational awareness. They are not the primary NLP corpus and are not used as ground truth.</p>
            </section>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Analytical Workflow")
    st.markdown(
        """
        1. **Data ingestion:** RSS feeds and structured event files are collected into `data/`.
        2. **Text preparation:** HTML/noise removal, normalization, tokenization and processed text generation.
        3. **Weak supervision:** Keyword and theme rules assign provisional labels for ML training.
        4. **Semantic modeling:** Sentence-transformer embeddings and UMAP expose semantic neighborhoods.
        5. **Topic modeling:** BERTopic extracts interpretable groups and representative terms.
        6. **Classification:** Multiple supervised models are compared using accuracy and macro-F1.
        7. **Search and dashboarding:** FAISS search and Streamlit views support analyst exploration.
        """
    )

    st.markdown("### Limitations")
    st.warning(
        "This is an academic OSINT analytics platform. Weak labels are not human-validated ground truth, "
        "the corpus is small, source distribution is imbalanced, and geospatial observations are contextual rather than causal evidence."
    )

    st.markdown("### File Locations")
    st.code(
        "\n".join(
            [
                "data/                      # corpus and raw context data",
                "outputs/                   # metrics, reports and generated artifacts",
                "outputs/models/            # trained models, topics and FAISS index",
                "outputs/advanced_figures/  # BERTopic HTML, UMAP and confusion matrices",
                "dashboard/assets/styles.css # custom dashboard styling",
            ]
        ),
        language="text",
    )


# ============================================================================
# Main entry point
# ============================================================================


def main() -> None:
    render_header()
    page = render_sidebar()

    if page == "Overview":
        render_overview()
    elif page == "Geospatial Intelligence":
        render_geospatial()
    elif page == "Timeline Analysis":
        render_timeline()
    elif page == "Semantic Explorer":
        render_semantic_explorer()
    elif page == "Topic Analysis":
        render_topic_analysis()
    elif page == "ML Models":
        render_ml_models()
    elif page == "Semantic Search":
        render_semantic_search()
    elif page == "News Explorer":
        render_news_explorer()
    elif page == "About / Methodology":
        render_about()

    st.markdown(
        """
        <div class="footer">
            OSINT Intelligence Center | Streamlit, Plotly, BERTopic, FAISS and scikit-learn
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
