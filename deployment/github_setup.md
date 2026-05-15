# GitHub Setup Guide

## Recommended Repository Contents

Commit:

```text
app.py
README.md
requirements.txt
.env.example
run_project.sh
data/
notebooks/
scripts/
outputs/
report/
dashboard/
assets/
docs/
deployment/
screenshots/
```

Do not commit:

```text
.env
.venv/
__pycache__/
.DS_Store
.ipynb_checkpoints/
release/
*.zip
```

## Initialize And Push

```bash
git init
git add .
git commit -m "Final OSINT Intelligence Center release"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

## Recommended Tags

```bash
git tag -a v1.0.0 -m "University delivery release"
git push origin v1.0.0
```

## Academic Submission Tip

Use GitHub for traceability and the generated ZIP for LMS/university submission.

