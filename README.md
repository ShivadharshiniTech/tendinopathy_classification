# Tendinopathy classification demo


This repository contains a minimal pipeline and Streamlit demo to classify tendinopathy using each row as an independent observation (each row = one measurement/observation).

Files in this repo:
- `utils.py` — helpers to read CSVs and extract pain features when needed.
- `train_and_save_model.py` — trains a Random Forest baseline using a stratified train/test split (rows treated independently) and saves `model.joblib` and test predictions.
- `app.py` — Streamlit demo: upload CSVs, predict, and simulate CSV replay.
- `requirements.txt` — Python dependencies.

Quick start

1. Create a virtual environment and install requirements:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Train a model using a CSV where each row is one observation (must contain `Condition` column):

```bash
python train_and_save_model.py --summary "dataset/PainModel_Summary_AllSubjects_2 (1).csv" --out model.joblib --test-out test_results.csv --test-size 0.2
```

3. Run the Streamlit demo:

```bash
streamlit run app.py
```

Notes
- Now each row is treated as one independent observation. The training script uses a stratified random train/test split (rows are split, not subjects), and preprocessing (scaler) is fit on the training rows only.
- If your CSV contains multiple measurements per person and you intend to predict per-person status, you should aggregate rows to a person-level label before training.

Next steps
- Add SHAP explanation panel in the app.
- Improve feature engineering and nested cross-validation.
- Add model calibration and exportable PDF reports.
