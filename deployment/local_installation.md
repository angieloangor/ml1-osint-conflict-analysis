# Local Installation Guide

This guide runs the OSINT Intelligence Center locally on macOS with Python 3.14.

## 1. Requirements

- macOS
- Python 3.14
- Terminal access
- Internet access for first dependency/model installation

## 2. Create Virtual Environment

```bash
cd Proyecto_final_ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Validate Critical Files

```bash
test -f app.py
test -f requirements.txt
test -f data/dataset_nlp_labeled.csv
test -f outputs/model_comparison_advanced.csv
test -f dashboard/assets/styles.css
```

## 5. Run Dashboard

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## 6. One-Command Launch

```bash
chmod +x run_project.sh
./run_project.sh
```

The script creates or activates `.venv`, installs dependencies when needed, validates critical files and starts Streamlit.

