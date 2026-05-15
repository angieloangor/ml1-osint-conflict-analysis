# Streamlit Community Cloud Deployment

Official documentation: https://docs.streamlit.io/deploy/streamlit-community-cloud

## Deployment Steps

1. Push the repository to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app from the GitHub repository.
4. Select the branch that contains this project.
5. Set the main file path to:

```text
app.py
```

6. Confirm that `requirements.txt` exists in the repository root.
7. Deploy the app.

## Files Required In Repository

```text
app.py
requirements.txt
README.md
.env.example
data/
outputs/
scripts/
dashboard/
report/
```

## Secrets

Do not commit `.env`. If a deployment requires API keys, configure them in Streamlit secrets.

## Notes For Large Dependencies

This project includes NLP and ML libraries such as `sentence-transformers`, `bertopic`, `faiss-cpu` and `xgboost`. If the cloud runtime is slow or memory constrained:

- Keep generated artifacts committed in `data/` and `outputs/`.
- Avoid retraining models during app startup.
- Use the built-in TF-IDF fallback for semantic search when transformer loading is unavailable.
- Consider a lighter deployment branch if the instructor only needs the dashboard demo.

## Main URL

After deployment, Streamlit provides a public URL similar to:

```text
https://your-app-name.streamlit.app
```

