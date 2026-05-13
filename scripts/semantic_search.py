#!/usr/bin/env python3
"""
Semantic Search Module
======================

Implementa búsqueda semántica usando embeddings y FAISS.
Permite encontrar documentos similares por similitud coseno.

Outputs:
- outputs/models/faiss_index.pkl: FAISS index + metadata
- Capacidad de búsqueda interactiva en el dashboard
"""

from __future__ import annotations

import warnings
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

class SemanticSearchEngine:
    """FAISS-based semantic search engine."""
    
    def __init__(self, embeddings: np.ndarray, df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize search engine.
        
        Args:
            embeddings: (n_docs, embedding_dim) array
            df: DataFrame with document metadata
            model_name: HuggingFace model for query encoding
        """
        self.embeddings = embeddings
        self.df = df.reset_index(drop=True)
        self.model = SentenceTransformer(model_name)
        
        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype(np.float32))
        print(f'Built FAISS index: {self.index.ntotal} documents, {dim} dimensions')
    
    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Search for similar documents.
        
        Args:
            query: text query
            k: number of results
        
        Returns:
            list of dicts with rank, score, document info
        """
        # Encode query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), 1):
            row = self.df.iloc[idx]
            similarity = 1.0 / (1.0 + distance)  # Convert L2 distance to similarity
            
            results.append({
                'rank': rank,
                'similarity': float(similarity),
                'distance': float(distance),
                'source': row.get('source', 'N/A'),
                'title': row.get('title', 'N/A'),
                'text': row.get('text', 'N/A')[:200],  # First 200 chars
                'timestamp': row.get('timestamp', 'N/A'),
                'weak_label': row.get('weak_label', 'N/A'),
                'url': row.get('url', 'N/A')
            })
        
        return results
    
    def batch_search(self, queries: list[str], k: int = 5) -> dict:
        """Search multiple queries."""
        results = {}
        for query in queries:
            results[query] = self.search(query, k)
        return results
    
    def save(self, output_path: Path | str = 'outputs/models/faiss_index.pkl') -> None:
        """Save the search engine."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open('wb') as f:
            pickle.dump({
                'index': self.index,
                'df': self.df,
                'model_name': self.model.__class__.__name__,
                'embeddings_shape': self.embeddings.shape
            }, f)
        
        print(f'Saved search engine: {output_path}')
    
    @staticmethod
    def load(path: Path | str = 'outputs/models/faiss_index.pkl') -> SemanticSearchEngine:
        """Load a saved search engine."""
        path = Path(path)
        with path.open('rb') as f:
            data = pickle.load(f)
        
        # Reconstruct
        engine = SemanticSearchEngine.__new__(SemanticSearchEngine)
        engine.index = data['index']
        engine.df = data['df']
        engine.model = SentenceTransformer('all-MiniLM-L6-v2')
        engine.embeddings = None  # Not needed after loading
        
        print(f'Loaded search engine: {path}')
        return engine

def load_embeddings_data() -> tuple[np.ndarray, pd.DataFrame]:
    """Load pre-computed embeddings and dataset."""
    embeddings_path = Path('data/dataset_nlp_embeddings.pkl')
    csv_path = Path('data/dataset_nlp_labeled.csv')
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f'{embeddings_path} not found. Run semantic_embeddings.py first.')
    
    with embeddings_path.open('rb') as f:
        data = pickle.load(f)
        embeddings = data['embeddings']
    
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    
    return embeddings, df

def run_semantic_search_pipeline() -> dict:
    """Run semantic search setup."""
    print('=' * 70)
    print('SEMANTIC SEARCH PIPELINE')
    print('=' * 70)
    
    embeddings, df = load_embeddings_data()
    print(f'Loaded {len(df)} documents with embeddings {embeddings.shape}')
    
    # Build search engine
    engine = SemanticSearchEngine(embeddings, df)
    
    # Example searches
    example_queries = [
        'Israel Iran military conflict',
        'diplomatic negotiations ceasefire',
        'sanctions embargo energy',
        'humanitarian crisis refugees'
    ]
    
    print('\nExample searches:')
    for query in example_queries:
        print(f'\nQuery: "{query}"')
        results = engine.search(query, k=3)
        for result in results:
            print(f'  [{result["rank"]}] {result["similarity"]:.3f} - {result["title"][:60]}...')
    
    # Save engine
    engine.save()
    
    summary = {
        'n_documents': len(df),
        'embeddings_shape': embeddings.shape,
        'index_path': 'outputs/models/faiss_index.pkl',
        'example_queries': len(example_queries)
    }
    
    print('\nSemantic search pipeline completed')
    for key, value in summary.items():
        print(f'  {key}: {value}')
    
    return summary

if __name__ == '__main__':
    run_semantic_search_pipeline()
