#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "OSINT Intelligence Center - Local Launcher"
echo "Project directory: $PROJECT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 was not found."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "Python: $(python --version)"

if ! python -c "import streamlit" >/dev/null 2>&1; then
  echo "Installing dependencies..."
  python -m pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "Dependencies appear to be installed."
fi

critical_files=(
  "app.py"
  "README.md"
  "requirements.txt"
  "data/dataset_nlp_labeled.csv"
  "outputs/model_comparison_advanced.csv"
  "dashboard/assets/styles.css"
)

echo "Validating critical files..."
for file in "${critical_files[@]}"; do
  if [ ! -f "$file" ]; then
    echo "WARNING: Missing $file"
  else
    echo "OK: $file"
  fi
done

echo ""
echo "Starting Streamlit..."
echo "Local URL: http://localhost:8501"
echo "Network URL will be printed by Streamlit if available."
echo ""

streamlit run app.py
