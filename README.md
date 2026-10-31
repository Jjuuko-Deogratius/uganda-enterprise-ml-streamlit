# Uganda Household Enterprise ML Predictor

A Streamlit deployment of two UICT Developing Data Products capstone models:

1. **Regression:** Random Forest predicts monthly gross revenue (UGX).
2. **Classification:** Multinomial Logistic Regression predicts whether revenue increased, decreased, or remained the same.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Project files

```text
app.py
requirements.txt
models/
  regression_random_forest.joblib
  classification_logistic.joblib
  metadata.json
train_models.py        # optional model-regeneration script; requires gsec12_2.csv
```

The raw UNPS dataset is **not required at runtime**. The app loads already-fitted pipelines from `models/`.

## Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and the `models/` directory.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Create a new app and select your repository, branch, and `app.py` entrypoint.
4. Deploy.

The deployed app is for academic demonstration and decision support, not guaranteed business forecasting.
