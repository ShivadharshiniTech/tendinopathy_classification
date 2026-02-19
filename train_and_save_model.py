import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, balanced_accuracy_score, recall_score, precision_score, brier_score_loss

from utils import load_summary_csv
import random


def pick_feature_columns(df):
    cols = list(df.columns)
    features = []
    for name in ['Peak_Pain', 'Mean_Pain', 'Std_Pain', 'VAS_Model']:
        if name in cols:
            features.append(name)
    # try fuzzy names
    if not features:
        for c in cols:
            if 'peak' in c.lower():
                features.append(c)
            if 'mean' in c.lower() and c not in features:
                features.append(c)
    return features


def prepare_data(df):
    # ensure Subject and Condition exist
    if 'Subject' not in df.columns or 'Condition' not in df.columns:
        raise ValueError('Summary data must contain Subject and Condition columns')

    features = pick_feature_columns(df)
    if len(features) == 0:
        raise ValueError('No suitable feature columns found. Need Peak/Mean pain or VAS_Model')

    X = df[features].apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)
    y = df['Condition'].astype(str).str.lower().map(lambda s: 1 if 'tend' in s else 0)
    groups = df['Subject']
    return X, y, groups, features


def main(args):
    path = Path(args.summary)
    df = load_summary_csv(path)

    X, y, groups, features = prepare_data(df)
    # First try: stratified row-wise split (keeps class ratios by rows)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=42
        )
        # ensure both classes present in train
        if len(np.unique(y_train)) < 2:
            raise ValueError('train split has only one class')
    except Exception:
        # fallback: if Subject exists, try many random group-aware splits to preserve class balance
        if 'Subject' in df.columns:
            print('Row-wise stratified split failed; attempting group-aware stratified split by Subject')
            groups_arr = df['Subject'].values
            unique_groups = list(pd.Series(groups_arr).unique())
            n_trials = 2000
            success = False
            rng = random.Random(42)
            for _ in range(n_trials):
                rng.shuffle(unique_groups)
                test_groups = set()
                n_total = len(df)
                target_test_n = int(max(1, round(args.test_size * n_total)))
                curr_n = 0
                for g in unique_groups:
                    g_count = int((groups_arr == g).sum())
                    if curr_n + g_count <= target_test_n or len(test_groups) == 0:
                        test_groups.add(g)
                        curr_n += g_count
                    if curr_n >= target_test_n:
                        break
                test_idx = df[df.Subject.isin(test_groups)].index
                train_idx = df[~df.index.isin(test_idx)].index
                y_train_try = y.loc[train_idx]
                y_test_try = y.loc[test_idx]
                if len(y_train_try) > 0 and len(y_test_try) > 0 and len(np.unique(y_train_try)) >= 2:
                    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
                    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
                    success = True
                    print('Found group-aware split with', len(train_idx), 'train rows and', len(test_idx), 'test rows')
                    break
            if not success:
                print('Could not find a group-aware split that preserves classes. Will train on full data (no held-out test).')
                X_train, X_test, y_train, y_test = X, pd.DataFrame(columns=X.columns), y, pd.Series([], dtype=int)
        else:
            print('Stratified split failed and no Subject column found; will train on full data (no held-out test).')
            X_train, X_test, y_train, y_test = X, pd.DataFrame(columns=X.columns), y, pd.Series([], dtype=int)

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
    ])

    # Check class distribution in training set
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    if len(unique) < 2:
        print('WARNING: Training set contains only one class:', unique)
        print('Training will proceed but the model will predict a constant class. Collect more labeled data with both classes for meaningful results.')

    # cross-validate on train set (stratified) only if enough samples per class
    try:
        n_min_class = counts.min()
        if n_min_class < 2:
            print('Not enough samples per class for cross-validation (need >=2 per class). Skipping CV.')
            scores = []
        else:
            n_splits = min(5, n_min_class)
            if n_splits < 2:
                print('Not enough samples to run CV. Skipping CV.')
                scores = []
            else:
                skf = StratifiedKFold(n_splits=n_splits)
                scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='roc_auc')
                print('Train Stratified CV ROC AUC scores:', scores)
                print('Mean train AUC:', np.nanmean(scores))
    except Exception as e:
        print('Cross-validation on train set failed:', e)

    # fit on train set
    clf.fit(X_train, y_train)

    # evaluate on held-out test set if available
    if X_test.shape[0] == 0 or y_test.shape[0] == 0:
        print('No held-out test set available; skipping test evaluation.')
        prob = None
    else:
        # predict probabilities for positive class (1). Handle case when classifier saw only one class
        proba_all = clf.predict_proba(X_test)
        if proba_all.shape[1] == 1:
            single_class = clf.classes_[0]
            prob = (np.ones(len(X_test)) if single_class == 1 else np.zeros(len(X_test)))
        else:
            prob = proba_all[:, 1]
        pred = (prob >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y_test, prob)
        except Exception:
            auc = float('nan')
        acc = accuracy_score(y_test, pred)
        bal = balanced_accuracy_score(y_test, pred)
        recall = recall_score(y_test, pred)
        prec = precision_score(y_test, pred)
        brier = brier_score_loss(y_test, prob)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

        print(f'Test AUC: {auc:.3f}')
        print(f'Accuracy: {acc:.3f}  Balanced acc: {bal:.3f}  Sensitivity(recall): {recall:.3f}  Specificity: {specificity:.3f}  Brier: {brier:.4f}')

    # save model trained on train set and test results
    out = {'model': clf, 'features': features}
    joblib.dump(out, args.out)
    print('Saved model (trained on train set) to', args.out)

    # save test results if available
    if prob is not None:
        test_df = X_test.copy()
        # keep Condition column if present in original df
        if 'Condition' in df.columns:
            # align by index if possible
            test_df['Condition'] = df.loc[X_test.index, 'Condition'].values
        test_df['true_label'] = y_test.values
        test_df['pred'] = pred
        test_df['prob'] = prob
        test_df.to_csv(args.test_out, index=False)
        print('Saved test predictions to', args.test_out)
    else:
        print('No test predictions saved because no held-out test set was available.')

    # Optionally train final model on full data and save
    if args.save_full:
        clf_full = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
        ])
        clf_full.fit(X, y)
        joblib.dump({'model': clf_full, 'features': features}, args.full_out)
        print('Saved model trained on full data to', args.full_out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--summary', default='dataset/PainModel_Summary_AllSubjects_2 (1).csv')
    p.add_argument('--out', default='model.joblib')
    p.add_argument('--test-out', default='test_results.csv')
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--save-full', action='store_true', help='Also train on full dataset and save a second model')
    p.add_argument('--full-out', default='model_full_data.joblib')
    args = p.parse_args()
    main(args)
