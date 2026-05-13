#!/usr/bin/env python3
"""
BERTopic Analysis Module
=========================

Implementa topic modeling avanzado con BERTopic y compara con NMF.
Genera tópicos interpretables y visualizaciones interactivas.

Outputs:
- outputs/models/bertopic_model/: BERTopic model saved
- outputs/models/bertopic_topics.csv: Tópicos interpretables
- outputs/advanced_figures/bertopic_topics.html: Visualización interactiva
- outputs/models/topic_comparison.csv: Comparación BERTopic vs NMF
"""

from __future__ import annotations

import warnings
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl

warnings.filterwarnings('ignore')

def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load data and embeddings."""
    csv_path = Path('data/dataset_nlp_labeled.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f'{csv_path} not found.')
    
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    
    # Load embeddings if available
    embeddings_path = Path('data/dataset_nlp_embeddings.pkl')
    if embeddings_path.exists():
        with embeddings_path.open('rb') as f:
            data = pickle.load(f)
            embeddings = data['embeddings']
        print(f'Loaded pre-computed embeddings: {embeddings.shape}')
    else:
        embeddings = None
        print('No pre-computed embeddings found; BERTopic will generate them.')
    
    return df, embeddings

def train_bertopic(
    texts: list[str],
    embeddings: np.ndarray | None = None,
    nr_topics: int = 10,
    min_topic_size: int = 5
) -> BERTopic:
    """
    Train BERTopic model.
    
    Args:
        texts: list of documents
        embeddings: pre-computed sentence embeddings (optional)
        nr_topics: desired number of topics
        min_topic_size: minimum cluster size
    
    Returns:
        model: trained BERTopic model
    """
    print(f'Training BERTopic with {nr_topics} topics, min_topic_size={min_topic_size}...')
    
    vectorizer_model = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=2,
        stop_words='english'
    )
    
    model = BERTopic(
        vectorizer_model=vectorizer_model,
        nr_topics=nr_topics,
        min_topic_size=min_topic_size,
        calculate_probabilities=True,
        verbose=True
    )
    
    if embeddings is not None:
        topics, probs = model.fit_transform(texts, embeddings)
    else:
        topics, probs = model.fit_transform(texts)
    
    print(f'BERTopic training completed')
    return model, topics, probs

def extract_topic_info(model: BERTopic, df: pd.DataFrame) -> pd.DataFrame:
    """Extract readable topic information."""
    topic_info = model.get_topic_info()
    
    # Enrich with document counts
    topics_count = df['bertopic_id'].value_counts() if 'bertopic_id' in df.columns else pd.Series()
    topic_info['doc_count'] = topic_info['Topic'].map(topics_count).fillna(0).astype(int)
    
    return topic_info

def visualize_bertopic(model: BERTopic, output_path: Path | str = 'outputs/advanced_figures/bertopic_topics.html') -> None:
    """Create interactive BERTopic visualization."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f'Generating BERTopic visualization...')
    try:
        fig = model.visualize_topics()
        fig.write_html(str(output_path))
        print(f'Saved BERTopic interactive visualization: {output_path}')
    except Exception as e:
        print(f'Warning: Could not generate interactive visualization: {e}')

def save_bertopic_model(model: BERTopic, output_dir: Path | str = 'outputs/models/bertopic_model') -> None:
    """Save BERTopic model."""
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir / 'model'))
    print(f'Saved BERTopic model: {output_dir}')

def compare_with_nmf(df: pd.DataFrame) -> pd.DataFrame:
    """Compare BERTopic with existing NMF topics."""
    if 'topic_label' not in df.columns:
        print('Warning: No NMF topic_label found for comparison.')
        return None
    
    comparison = pd.DataFrame({
        'method': ['BERTopic', 'NMF'],
        'unique_topics': [
            len(df['bertopic_id'].unique()) if 'bertopic_id' in df.columns else 0,
            len(df['topic_label'].unique())
        ],
        'model_type': ['Transformer-based', 'Matrix Factorization'],
        'interpretability': ['High (topic words)', 'Medium (factor loadings)']
    })
    
    return comparison

def run_bertopic_pipeline() -> dict:
    """Run complete BERTopic pipeline."""
    print('=' * 70)
    print('BERTOPIC ANALYSIS PIPELINE')
    print('=' * 70)
    
    df, embeddings = load_data()
    texts = df['processed_text'].fillna('').tolist()
    
    model, topics, probs = train_bertopic(texts, embeddings=embeddings, nr_topics=10, min_topic_size=5)
    df['bertopic_id'] = topics
    df['bertopic_prob'] = probs.max(axis=1) if isinstance(probs, np.ndarray) else 0.0
    
    topic_info = extract_topic_info(model, df)
    print('\nTop BERTopic topics:')
    print(topic_info.head(10).to_string())
    
    # Save topic info
    topic_info_path = Path('outputs/models/bertopic_topics.csv')
    topic_info_path.parent.mkdir(parents=True, exist_ok=True)
    topic_info.to_csv(topic_info_path, index=False)
    print(f'\nSaved topic info: {topic_info_path}')
    
    # Visualizations
    visualize_bertopic(model)
    
    # Save model
    save_bertopic_model(model)
    
    # Save enriched dataset
    df.to_csv('data/dataset_nlp_bertopic.csv', index=False)
    print(f'Saved enriched dataset: data/dataset_nlp_bertopic.csv')
    
    # Comparison with NMF
    comparison = compare_with_nmf(df)
    if comparison is not None:
        comparison.to_csv('outputs/models/topic_comparison.csv', index=False)
        print('\nTopic method comparison:')
        print(comparison.to_string(index=False))
    
    summary = {
        'n_documents': len(df),
        'n_topics': len(df['bertopic_id'].unique()),
        'model_path': 'outputs/models/bertopic_model',
        'topic_info_path': 'outputs/models/bertopic_topics.csv',
        'visualization_path': 'outputs/advanced_figures/bertopic_topics.html',
        'enriched_dataset': 'data/dataset_nlp_bertopic.csv'
    }
    
    print('\nBERTopic pipeline completed')
    for key, value in summary.items():
        print(f'  {key}: {value}')
    
    return summary

if __name__ == '__main__':
    run_bertopic_pipeline()
