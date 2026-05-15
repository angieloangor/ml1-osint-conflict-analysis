#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate
# shellcheck source=/dev/null
source scripts/setup_runtime_env.sh

python -m pip install -r requirements.txt

echo "[1/9] OSINT API/data pipeline"
python osint_pipeline.py

echo "[2/9] NLP dataset preparation"
python scripts/prepare_dataset_nlp.py

echo "[3/9] Classic NLP analysis"
python scripts/nlp_analysis.py

echo "[4/9] Supervised ML baselines"
python scripts/supervised_ml.py

echo "[5/9] Semantic embeddings"
python scripts/semantic_embeddings.py

echo "[6/9] BERTopic topic modeling"
python scripts/bertopic_analysis.py

echo "[7/9] Semantic search index"
python scripts/semantic_search.py

echo "[8/9] Advanced ML comparison"
python scripts/advanced_ml.py

echo "[9/9] Humanitarian GeoAI risk model"
python scripts/humanitarian_risk_model.py

echo "All scripts completed."

