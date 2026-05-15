# Academic Methodology

## Objective

The objective is to build a reproducible OSINT-style NLP platform for analyzing public news narratives around the Iran-Israel-US conflict. The project focuses on document exploration, weak-label classification, topic modeling and contextual geospatial awareness.

## NLP Pipeline

1. Data collection from RSS and structured sources.
2. Text normalization and cleaning.
3. Classic NLP with TF-IDF and clustering.
4. Weak-label generation through heuristic rules.
5. Supervised classification using weak labels.
6. Semantic embeddings using sentence-transformers.
7. UMAP projection for semantic exploration.
8. BERTopic topic modeling.
9. FAISS semantic search and dashboard integration.

## Classic NLP vs Modern NLP

Classic NLP methods such as TF-IDF are transparent, fast and useful for baselines. Modern embeddings capture semantic similarity even when documents do not share exact vocabulary. The project uses both to compare interpretability and semantic power.

## Weak Labels

Weak labels are generated automatically from thematic cues. They enable supervised experiments without manual annotation, but they introduce noise. Model metrics must therefore be interpreted as pipeline performance over weak labels, not as verified real-world truth.

## BERTopic

BERTopic provides topic representations using transformer embeddings and class-based TF-IDF. It helps summarize dominant narratives and inspect representative documents.

## Geospatial Context

NASA FIRMS and OpenSky are treated as contextual signals. They are not merged as text documents and are not used as ground truth. Their purpose is situational awareness and temporal/spatial comparison.

## Limitations

- Small corpus.
- Source imbalance.
- Heuristic labels.
- English-only analysis.
- No expert validation.
- No causal claim between geospatial observations and news documents.

