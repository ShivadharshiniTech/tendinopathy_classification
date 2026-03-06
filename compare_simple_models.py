import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, brier_score_loss
)
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = Path('results')
IMAGES_DIR = Path('images')
RESULTS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
TEST_SIZE = 0.25
RANDOM_STATE = 42

# Load temporal features
df = pd.read_csv('dataset/temporal_features.csv')
exclude_cols = ['Subject', 'Condition', 'Task', 'Speed', 'true_label']
FEATURES = [c for c in df.columns if c not in exclude_cols]
X = df[FEATURES].values
y = df['true_label'].values
groups = df['Subject'].values

# Group-aware split
splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Define models (use pipelines where appropriate)
models = {
    'knn': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=3, weights='distance'))]),
    'decision_tree': DecisionTreeClassifier(max_depth=5, min_samples_split=4, class_weight='balanced', random_state=RANDOM_STATE),
    'naive_bayes': Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())]),
    'lda': Pipeline([('scaler', StandardScaler()), ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))]),
    'logistic': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE))])
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    # fit
    model.fit(X_train, y_train)

    # predict train
    try:
        proba_train = model.predict_proba(X_train)[:,1]
    except Exception:
        proba_train = None
    pred_train = model.predict(X_train)

    # predict test
    try:
        proba_test = model.predict_proba(X_test)[:,1]
    except Exception:
        proba_test = None
    pred_test = model.predict(X_test)

    def compute_metrics(y_true, y_pred, y_proba=None):
        m = {}
        m['n'] = int(len(y_true))
        m['accuracy'] = float(accuracy_score(y_true, y_pred))
        m['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        m['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        m['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
        m['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        if y_proba is not None and len(np.unique(y_true))>=2:
            try:
                m['roc_auc'] = float(roc_auc_score(y_true, y_proba))
            except Exception:
                m['roc_auc'] = None
            try:
                m['brier'] = float(brier_score_loss(y_true, y_proba))
            except Exception:
                m['brier'] = None
        else:
            m['roc_auc'] = None
            m['brier'] = None
        return m

    m_train = compute_metrics(y_train, pred_train, proba_train)
    m_test = compute_metrics(y_test, pred_test, proba_test)

    results[name] = {'train': m_train, 'test': m_test}

    # save confusion matrix images
    def save_cm(cm, title, fname):
        labels = ['Normal', 'Tendon']
        plt.figure(figsize=(4,3))
        sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

    save_cm(m_train['confusion_matrix'], f'{name} - Train Confusion Matrix', IMAGES_DIR / f'cm_{name}_train.png')
    save_cm(m_test['confusion_matrix'], f'{name} - Test Confusion Matrix', IMAGES_DIR / f'cm_{name}_test.png')

# Save results
with open(RESULTS_DIR / 'metrics_simple_models.json', 'w') as f:
    json.dump(results, f, indent=2)

# Also produce a CSV summary
rows = []
for name, val in results.items():
    for split in ['train', 'test']:
        m = val[split]
        rows.append({
            'model': name,
            'split': split,
            'n': m['n'],
            'accuracy': m['accuracy'],
            'precision': m['precision'],
            'recall': m['recall'],
            'f1': m['f1'],
            'roc_auc': m['roc_auc'],
            'brier': m['brier']
        })

pd.DataFrame(rows).to_csv(RESULTS_DIR / 'metrics_simple_models.csv', index=False)
print('Done. Metrics saved to results/metrics_simple_models.json and results/metrics_simple_models.csv')
print('Confusion matrix images saved to images/ folder')
