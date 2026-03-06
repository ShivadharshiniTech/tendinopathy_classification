import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
)

# Config (match training defaults)
TEST_SIZE = 0.25
RANDOM_STATE = 42

# Load dataset
df = pd.read_csv('dataset/temporal_features.csv')
exclude_cols = ['Subject', 'Condition', 'Task', 'Speed', 'true_label']
FEATURES = [c for c in df.columns if c not in exclude_cols]
X = df[FEATURES].values
y = df['true_label'].values
groups = df['Subject'].values

# Recreate group-aware split
splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Load artifacts
model_bundle = joblib.load('model_temporal.joblib') if 'model_temporal.joblib' else None
model = model_bundle['model'] if model_bundle else None
scaler = joblib.load('scaler_temporal.joblib') if 'scaler_temporal.joblib' else None

# If scaler saved separately, use it
if scaler is None and model_bundle is not None and 'scaler' in model_bundle:
    scaler = model_bundle['scaler']

# Scale
if scaler is not None:
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
else:
    X_train_s, X_test_s = X_train, X_test

# Predict
proba_train = model.predict_proba(X_train_s)[:,1]
pred_train = (proba_train >= 0.5).astype(int)
proba_test = model.predict_proba(X_test_s)[:,1]
pred_test = (proba_test >= 0.5).astype(int)

# Metrics function
def metrics(y_true, pred, proba=None):
    m = {}
    m['n'] = int(len(y_true))
    m['accuracy'] = float(accuracy_score(y_true, pred))
    m['precision'] = float(precision_score(y_true, pred, zero_division=0))
    m['recall'] = float(recall_score(y_true, pred, zero_division=0))
    m['f1'] = float(f1_score(y_true, pred, zero_division=0))
    m['confusion_matrix'] = confusion_matrix(y_true, pred).tolist()
    if proba is not None and len(np.unique(y_true))>=2:
        try:
            m['roc_auc'] = float(roc_auc_score(y_true, proba))
        except Exception:
            m['roc_auc'] = None
        try:
            m['brier'] = float(brier_score_loss(y_true, proba))
        except Exception:
            m['brier'] = None
    else:
        m['roc_auc'] = None
        m['brier'] = None
    return m

m_train = metrics(y_train, pred_train, proba_train)
m_test = metrics(y_test, pred_test, proba_test)

import json
from pathlib import Path

# Create results directory
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

out = {'train': m_train, 'test': m_test, 'features': FEATURES}
with open(results_dir / 'metrics_temporal.json','w') as f:
    json.dump(out, f, indent=2)

print('Train metrics:')
print(json.dumps(m_train, indent=2))
print('\nTest metrics:')
print(json.dumps(m_test, indent=2))
print('\nArtifacts saved to results/metrics_temporal.json')
