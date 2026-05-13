#!/usr/bin/env python3
"""
OSINT Analytical Platform - Streamlit Dashboard
================================================

Dashboard interactivo para análisis OSINT con NLP, ML y visualizaciones avanzadas.

Características:
- Corpus overview y estadísticas
- Semantic explorer con UMAP
- Topic analysis (BERTopic + NMF comparison)
- ML models comparison
- Semantic search
- Export utilities

Ejecutar con: streamlit run app.py
"""

from __future__ import annotations

import warnings
from pathlib import Path
import pickle

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

st.set_page_config(
    page_title="OSINT Analytical Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        padding-top: 0rem;
    }
    h1, h2 { color: #1f77b4; }
    .stMetric { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Data Loading
# ============================================================================

@st.cache_data
def load_labeled_data() -> pd.DataFrame:
    """Load labeled dataset."""
    path = Path('data/dataset_nlp_labeled.csv')
    if path.exists():
        return pd.read_csv(path, parse_dates=['timestamp'])
    return None

@st.cache_data
def load_bertopic_data() -> pd.DataFrame:
    """Load BERTopic enriched dataset."""
    path = Path('data/dataset_nlp_bertopic.csv')
    if path.exists():
        return pd.read_csv(path, parse_dates=['timestamp'])
    return None

@st.cache_data
def load_embeddings() -> dict:
    """Load embeddings and UMAP coordinates."""
    path = Path('data/dataset_nlp_embeddings.pkl')
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_search_engine():
    """Load semantic search engine."""
    path = Path('outputs/models/faiss_index.pkl')
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    """Load advanced model comparison."""
    path = Path('outputs/model_comparison_advanced.csv')
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_bertopic_topics() -> pd.DataFrame:
    """Load BERTopic topic info."""
    path = Path('outputs/models/bertopic_topics.csv')
    if path.exists():
        return pd.read_csv(path)
    return None

# ============================================================================
# Header
# ============================================================================

st.markdown("# 🛰️ OSINT Analytical Platform")
st.markdown("**Advanced NLP & ML Analysis for Geopolitical Intelligence**")
st.divider()

# ============================================================================
# Sidebar Navigation
# ============================================================================

page = st.sidebar.radio(
    "Navigate to:",
    [
        "📊 Corpus Overview",
        "🌐 Semantic Explorer",
        "📚 Topic Analysis",
        "🤖 ML Models",
        "🔍 Semantic Search",
        "📈 Statistics",
        "📄 About"
    ]
)

# ============================================================================
# PAGE 1: Corpus Overview
# ============================================================================

if page == "📊 Corpus Overview":
    st.markdown("## Dataset Overview")
    
    df = load_labeled_data()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", len(df))
        with col2:
            st.metric("Date Range", f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        with col3:
            st.metric("Unique Sources", df['source'].nunique())
        with col4:
            st.metric("Unique Labels", df['weak_label'].nunique())
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Documents by Source")
            source_counts = df['source'].value_counts()
            fig = px.bar(
                x=source_counts.index, y=source_counts.values,
                title="Distribution by Source",
                labels={'x': 'Source', 'y': 'Count'},
                color=source_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Documents by Weak Label")
            label_counts = df['weak_label'].value_counts()
            fig = px.pie(
                values=label_counts.values, names=label_counts.index,
                title="Distribution by Weak Label"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Timeline")
        timeline = df.groupby(df['timestamp'].dt.date).size()
        fig = px.line(
            x=timeline.index, y=timeline.values,
            title="Documents Over Time",
            labels={'x': 'Date', 'y': 'Count'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: Semantic Explorer
# ============================================================================

elif page == "🌐 Semantic Explorer":
    st.markdown("## Semantic Space Visualization")
    
    embeddings_data = load_embeddings()
    if embeddings_data is not None:
        df = embeddings_data['df']
        umap_coords = embeddings_data['umap']
        
        col1, col2 = st.columns(2)
        
        with col1:
            color_by = st.selectbox("Color by:", ['weak_label', 'source', 'cluster_id'])
        
        with col2:
            selected_label = st.selectbox("Filter (optional):", [None] + sorted(df[color_by].unique().tolist()))
        
        # Filter data if needed
        if selected_label:
            mask = df[color_by] == selected_label
            plot_coords = umap_coords[mask]
            plot_df = df[mask].copy()
            plot_df['UMAP1'] = plot_coords[:, 0]
            plot_df['UMAP2'] = plot_coords[:, 1]
        else:
            plot_df = df.copy()
            plot_df['UMAP1'] = umap_coords[:, 0]
            plot_df['UMAP2'] = umap_coords[:, 1]
        
        fig = px.scatter(
            plot_df,
            x='UMAP1', y='UMAP2',
            color=color_by,
            hover_data=['title', 'source', 'timestamp'],
            title=f"Semantic Space (UMAP) - Colored by {color_by}",
            height=600
        )
        fig.update_traces(marker=dict(size=6, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: Topic Analysis
# ============================================================================

elif page == "📚 Topic Analysis":
    st.markdown("## Topic Analysis")
    
    tab1, tab2, tab3 = st.tabs(["BERTopic", "NMF Comparison", "Topic Distribution"])
    
    with tab1:
        st.markdown("### BERTopic Topics")
        topics_df = load_bertopic_topics()
        if topics_df is not None:
            st.dataframe(topics_df.head(10), use_container_width=True)
        
        # Interactive visualization
        bertopic_html_path = Path('outputs/advanced_figures/bertopic_topics.html')
        if bertopic_html_path.exists():
            st.markdown("### Interactive Topic Visualization")
            with open(bertopic_html_path, 'r', encoding='utf-8') as f:
                html = f.read()
            st.components.v1.html(html, height=800, scrolling=True)
    
    with tab2:
        st.markdown("### Topic Method Comparison")
        comparison_path = Path('outputs/models/topic_comparison.csv')
        if comparison_path.exists():
            comparison = pd.read_csv(comparison_path)
            st.dataframe(comparison, use_container_width=True)
    
    with tab3:
        st.markdown("### Topic Distribution")
        bertopic_df = load_bertopic_data()
        if bertopic_df is not None and 'bertopic_id' in bertopic_df.columns:
            topic_dist = bertopic_df['bertopic_id'].value_counts().sort_index()
            fig = px.bar(
                x=topic_dist.index, y=topic_dist.values,
                title="Documents per BERTopic",
                labels={'x': 'Topic ID', 'y': 'Count'},
                color=topic_dist.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: ML Models
# ============================================================================

elif page == "🤖 ML Models":
    st.markdown("## Machine Learning Models Comparison")
    
    comparison_df = load_model_comparison()
    if comparison_df is not None:
        st.markdown("### Model Performance Metrics")
        st.dataframe(comparison_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                comparison_df.sort_values('accuracy', ascending=False),
                x='model', y='accuracy',
                title="Accuracy Comparison",
                color='accuracy',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                comparison_df.sort_values('f1_macro', ascending=False),
                x='model', y='f1_macro',
                title="F1 Macro Comparison",
                color='f1_macro',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrices
        st.markdown("### Confusion Matrices")
        models = comparison_df['model'].tolist()
        selected_model = st.selectbox("Select model:", models)
        
        cm_path = Path(f"outputs/advanced_figures/confusion_matrix_{selected_model.lower()}.png")
        if cm_path.exists():
            img = Image.open(cm_path)
            st.image(img, caption=f"Confusion Matrix: {selected_model}")

# ============================================================================
# PAGE 5: Semantic Search
# ============================================================================

elif page == "🔍 Semantic Search":
    st.markdown("## Semantic Similarity Search")
    
    search_data = load_search_engine()
    df = load_labeled_data()
    
    if search_data is not None and df is not None:
        from scripts.semantic_search import SemanticSearchEngine
        
        # Load the search engine properly
        embeddings_data = load_embeddings()
        if embeddings_data is not None:
            engine = SemanticSearchEngine(embeddings_data['embeddings'], df)
            
            st.markdown("### Find Similar Documents")
            query = st.text_input("Enter your query:", "Israel Iran military conflict")
            k = st.slider("Number of results:", 1, 20, 5)
            
            if query:
                results = engine.search(query, k=k)
                
                st.markdown(f"#### Results for: **{query}**")
                
                for result in results:
                    with st.expander(f"[{result['rank']}] {result['title']} (Similarity: {result['similarity']:.3f})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Source:** {result['source']}")
                            st.markdown(f"**Date:** {result['timestamp']}")
                            st.markdown(f"**Label:** {result['weak_label']}")
                        
                        with col2:
                            st.markdown(f"**Similarity Score:** {result['similarity']:.4f}")
                            st.markdown(f"**URL:** [{result['url'][:50]}...]({result['url']})")
                        
                        st.markdown(f"**Preview:** {result['text']}")

# ============================================================================
# PAGE 6: Statistics
# ============================================================================

elif page == "📈 Statistics":
    st.markdown("## Detailed Statistics")
    
    df = load_labeled_data()
    if df is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_text_length = df['processed_text'].str.len().mean()
            st.metric("Avg Text Length", f"{avg_text_length:.0f} chars")
        
        with col2:
            avg_tokens = df['processed_text'].str.split().apply(len).mean()
            st.metric("Avg Tokens", f"{avg_tokens:.0f}")
        
        with col3:
            date_range = (df['timestamp'].max() - df['timestamp'].min()).days
            st.metric("Date Range", f"{date_range} days")
        
        st.divider()
        
        st.markdown("### Label Distribution Statistics")
        label_stats = df['weak_label'].value_counts().to_frame('count')
        label_stats['percentage'] = (label_stats['count'] / len(df) * 100).round(2)
        st.dataframe(label_stats, use_container_width=True)
        
        st.markdown("### Source Statistics")
        source_stats = df.groupby('source').agg({
            'timestamp': ['count', 'min', 'max'],
            'processed_text': lambda x: x.str.len().mean()
        }).round(2)
        st.dataframe(source_stats, use_container_width=True)

# ============================================================================
# PAGE 7: About
# ============================================================================

elif page == "📄 About":
    st.markdown("""
    ## About OSINT Analytical Platform
    
    ### Project Overview
    This platform provides advanced NLP and ML analysis for Open Source Intelligence (OSINT) 
    focused on geopolitical events, particularly the Israel-Iran conflict.
    
    ### Key Features
    
    ✅ **Semantic Embeddings**: Using sentence-transformers (all-MiniLM-L6-v2) for 384-dimensional embeddings
    
    ✅ **Advanced Topic Modeling**: BERTopic for interpretable topic extraction
    
    ✅ **ML Classification**: Multiple models (LogReg, RF, GB) with class weighting and stratified validation
    
    ✅ **Semantic Search**: FAISS-based similarity search for finding related documents
    
    ✅ **Interactive Visualizations**: UMAP, Plotly for exploratory data analysis
    
    ### Dataset
    - **Sources**: BBC RSS, Al Jazeera RSS, Google News RSS, GDELT
    - **Documents**: 289 processed articles
    - **Period**: 2025-12-29 to 2026-05-13
    - **Languages**: English
    
    ### Methodology
    
    1. **Data Preparation**: RSS aggregation, text cleaning, normalization
    2. **Feature Engineering**: TF-IDF vectors and semantic embeddings
    3. **Topic Extraction**: NMF and BERTopic for latent semantic analysis
    4. **Weak Labeling**: Heuristic categorization (escalation, military, diplomacy, etc.)
    5. **ML Classification**: Supervised learning with weak labels
    6. **Visualization**: Interactive dashboards for exploration
    
    ### Model Performance (Advanced)
    
    | Model | Accuracy | F1 Macro |
    |-------|----------|----------|
    | GradientBoosting | 91.4% | 0.765 |
    | RandomForest | 82.8% | 0.689 |
    | LogisticRegression | 69.0% | 0.576 |
    | KNeighbors | 48.3% | 0.360 |
    | MultinomialNB | 55.2% | 0.243 |
    
    ### Limitations
    
    - Weak labels are heuristic-based, not ground truth
    - Small corpus size (289 documents)
    - Google News RSS dominates source distribution
    - No validation with human expert review
    
    ### Technical Stack
    
    - **Python 3.14** | scikit-learn | pandas
    - **NLP**: sentence-transformers, NLTK, BERTopic
    - **ML**: sklearn ensemble methods
    - **Visualization**: Plotly, UMAP, Streamlit
    - **Semantic Search**: FAISS
    
    ### Use Cases
    
    1. **Intelligence Monitoring**: Track news trends and escalation indicators
    2. **Document Similarity**: Find related articles automatically
    3. **Topic Tracking**: Monitor emerging topics and themes
    4. **Prediction**: Classify new articles based on trained models
    5. **Exploratory Analysis**: Understand corpus structure and patterns
    
    ---
    
    **Last Updated**: May 12, 2026
    
    **Project**: ML1 - OSINT Analytical Platform
    """)

# ============================================================================
# Footer
# ============================================================================

st.divider()
st.markdown("Built with ❤️ using Streamlit, scikit-learn, and sentence-transformers")
