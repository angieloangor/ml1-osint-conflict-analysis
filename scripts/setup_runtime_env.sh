#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
cd "$PROJECT_DIR"

if [ -f ".env" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ""|\#*) continue ;;
      *=*) export "$line" ;;
    esac
  done < ".env"
fi

export PYTHONPATH="${PYTHONPATH:-.}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-.cache/matplotlib}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export NLTK_DATA="${NLTK_DATA:-.cache/nltk}"
export HF_HOME="${HF_HOME:-.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-.cache/huggingface/transformers}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-.cache/sentence_transformers}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-8}"

mkdir -p \
  "$MPLCONFIGDIR" \
  "$NLTK_DATA" \
  "$HF_HOME" \
  "$TRANSFORMERS_CACHE" \
  "$SENTENCE_TRANSFORMERS_HOME" \
  data \
  outputs/figures \
  outputs/advanced_figures \
  outputs/models \
  outputs/humanitarian_risk \
  dashboard/assets
