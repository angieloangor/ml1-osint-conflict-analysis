#!/usr/bin/env python3
"""
Semantic Embeddings Module
===========================

Genera embeddings semánticos usando sentence-transformers (all-MiniLM-L6-v2).
Proporciona visualización UMAP y búsqueda por similitud.

Outputs:
- data/dataset_nlp_embeddings.pkl: Dataset con embeddings
- outputs/advanced_figures/semantic_space_umap.png: Visualización 2D
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sentence_transformers import SentenceTransformer
import pickle

warnings.filterwarnings('ignore')

def load_data() -> pd.DataFrame:
    """Load preprocessed NLP dataset."""
    csv_path = Path('data/dataset_nlp_labeled.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f'{csv_path} not found. Run supervised_ml.py first.')
    return pd.read_csv(csv_path, parse_dates=['timestamp'])

def generate_embeddings(df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Generate sentence embeddings using transformers.
    
    Args:
        df: DataFrame with processed_text column
        model_name: HuggingFace model identifier
    
    Returns:
        embeddings: (n_docs, embedding_dim) array
    """
    print(f'Loading {model_name}...')
    model = SentenceTransformer(model_name)
    
    print(f'Generating embeddings for {len(df)} documents...')
    texts = df['processed_text'].fillna('').tolist()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    return embeddings

def reduce_embeddings_umap(embeddings: np.ndarray, n_components: int = 2, seed: int = 42) -> np.ndarray:
    """
    Reduce embeddings to 2D using UMAP for visualization.
    
    Args:
        embeddings: (n_docs, embedding_dim) array
        n_components: target dimensions
        seed: random seed
    
    Returns:
        umap_embeddings: (n_docs, 2) array
    """
    print(f'Reducing to {n_components}D with UMAP...')
    reducer = umap.UMAP(n_components=n_components, random_state=seed, verbose=True)
    umap_embeddings = reducer.fit_transform(embeddings)
    return umap_embeddings

def visualize_semantic_space(
    df: pd.DataFrame,
    umap_embeddings: np.ndarray,
    output_path: Path | str = 'outputs/advanced_figures/semantic_space_umap.png'
) -> None:
    """
    Create interactive visualization of semantic space.
    
    Args:
        df: DataFrame with metadata
        umap_embeddings: (n_docs, 2) UMAP embeddings
        output_path: where to save the figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color by weak_label if available
    if 'weak_label' in df.columns:
        labels = df['weak_label'].values
        unique_labels = sorted(df['weak_label'].unique())
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        for label in unique_labels:
            mask = labels == label
            ax.scatter(
                umap_embeddings[mask, 0],
                umap_embeddings[mask, 1],
                c=[label_to_color[label]],
                label=label,
                s=50,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
    ax.set_title('Semantic Space Visualization (UMAP)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved semantic visualization: {output_path}')
    plt.close()

def save_embeddings(df: pd.DataFrame, embeddings: np.ndarray, umap_embeddings: np.ndarray, output_path: str = 'data/dataset_nlp_embeddings.pkl') -> None:
    """Save dataset with embeddings and UMAP coordinates."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df['embedding'] = [embeddings[i] for i in range(len(df))]
    df['umap_x'] = umap_embeddings[:, 0]
    df['umap_y'] = umap_embeddings[:, 1]
    
    with output_path.open('wb') as f:
        pickle.dump({'df': df, 'embeddings': embeddings, 'umap': umap_embeddings}, f)
    
    print(f'Saved embeddings: {output_path}')
    print(f'  - Embedding dimension: {embeddings.shape[1]}')
    print(f'  - UMAP coordinates: {umap_embeddings.shape}')

def run_semantic_embeddings_pipeline() -> dict:
    """Run the complete embeddings pipeline."""
    print('=' * 70)
    print('SEMANTIC EMBEDDINGS PIPELINE')
    print('=' * 70)
    
    df = load_data()
    print(f'Loaded {len(df)} documents')
    
    embeddings = generate_embeddings(df)
    print(f'Generated embeddings: {embeddings.shape}')
    
    umap_embeddings = reduce_embeddings_umap(embeddings)
    print(f'UMAP reduced to: {umap_embeddings.shape}')
    
    visualize_semantic_space(df, umap_embeddings)
    
    save_embeddings(df, embeddings, umap_embeddings)
    
    summary = {
        'n_documents': len(df),
        'embedding_dim': embeddings.shape[1],
        'umap_shape': umap_embeddings.shape,
        'embeddings_path': 'data/dataset_nlp_embeddings.pkl',
        'visualization_path': 'outputs/advanced_figures/semantic_space_umap.png'
    }
    
    print('\nSemantic embeddings pipeline completed')
    for key, value in summary.items():
        print(f'  {key}: {value}')
    
    return summary

if __name__ == '__main__':
    run_semantic_embeddings_pipeline()
