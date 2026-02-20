import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GroupKFold, GroupShuffleSplit, cross_val_score, 
    train_test_split, StratifiedKFold, LeaveOneGroupOut, cross_validate
)
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, balanced_accuracy_score, 
    recall_score, precision_score, brier_score_loss, make_scorer
)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    ImbPipeline = Pipeline
    SMOTE = None
    HAS_IMBLEARN = False

from utils import load_summary_csv
import random
import warnings


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
    
    # Dataset diagnostics
    n_subjects = groups.nunique()
    n_samples = len(X)
    print(f'\n=== Dataset Info ===')
    print(f'Total samples: {n_samples}')
    print(f'Unique subjects: {n_subjects}')
    print(f'Class distribution: {dict(y.value_counts())}')
    print(f'Samples per subject: {groups.value_counts().describe().to_dict()}')
    
    # For very small datasets (<=5 subjects), recommend LeaveOneGroupOut
    if n_subjects <= 5:
        print(f'\n⚠️  WARNING: Only {n_subjects} subjects detected. This is a VERY SMALL dataset.')
        print('Recommended approach: Use LeaveOneGroupOut cross-validation for realistic estimates.')
        print('Model will likely have high variance and may not generalize well.\n')
    
    # Use group-aware split to prevent subject leakage
    if 'Subject' in df.columns and n_subjects >= 2:
        print('Using GROUP-AWARE split (by Subject) to prevent data leakage...')
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        
        # Verify no subject appears in both train and test
        train_subjects = set(groups_train.unique())
        test_subjects = set(groups.iloc[test_idx].unique())
        assert len(train_subjects & test_subjects) == 0, 'Subject leakage detected!'
        
        print(f'Train: {len(X_train)} samples, {len(train_subjects)} subjects')
        print(f'Test: {len(X_test)} samples, {len(test_subjects)} subjects')
        print(f'Train classes: {dict(y_train.value_counts())}')
        print(f'Test classes: {dict(y_test.value_counts())}')
    else:
        print('Using stratified row-wise split (no Subject grouping)...')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=42
        )
        groups_train = None

    # Choose model based on dataset size
    if args.model == 'logistic':
        print('Using Logistic Regression (good for small datasets)...')
        model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)
    else:
        print('Using Random Forest...')
        # Use fewer trees for small datasets to reduce overfitting
        n_estimators = 100 if n_subjects <= 5 else 200
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=3 if n_subjects <= 5 else None,  # Limit depth for small datasets
            class_weight='balanced', 
            random_state=42
        )
    
    # Build pipeline with optional SMOTE
    if args.use_smote and HAS_IMBLEARN:
        print('Using SMOTE for training data augmentation...')
        clf = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42, k_neighbors=1)),
            ('model', model)
        ])
    else:
        if args.use_smote and not HAS_IMBLEARN:
            print('⚠️  SMOTE requested but imbalanced-learn not installed. Install with: pip install imbalanced-learn')
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

    # Cross-validation strategy based on dataset size
    unique, counts = np.unique(y_train, return_counts=True)
    if len(unique) < 2:
        print('⚠️  WARNING: Training set contains only one class:', unique)
        print('Cannot train a meaningful classifier. Collect more diverse data.')
        return
    
    print('\n=== Cross-Validation ===')
    if groups_train is not None and n_subjects <= 5:
        # Use LeaveOneGroupOut for very small subject counts
        print(f'Using LeaveOneGroupOut CV ({n_subjects} folds)...')
        cv = LeaveOneGroupOut()
        cv_groups = groups_train
    elif groups_train is not None:
        # Use GroupKFold for larger datasets
        n_splits = min(5, n_subjects)
        print(f'Using GroupKFold CV ({n_splits} folds)...')
        cv = GroupKFold(n_splits=n_splits)
        cv_groups = groups_train
    else:
        # Stratified KFold when no groups
        n_splits = min(5, counts.min())
        print(f'Using StratifiedKFold CV ({n_splits} folds)...')
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_groups = None
    
    # Perform cross-validation with multiple metrics
    scoring = {
        'roc_auc': 'roc_auc',
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            cv_results = cross_validate(
                clf, X_train, y_train, 
                groups=cv_groups,
                cv=cv, 
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )
        
        print('\nCross-Validation Results (mean ± std):')
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            print(f'  {metric:20s}: {test_scores.mean():.3f} ± {test_scores.std():.3f}')
        
        print('\n⚠️  NOTE: With only {} subjects, these estimates have HIGH VARIANCE.'.format(n_subjects))
        print('Consider collecting more subjects for reliable model evaluation.')
        
    except Exception as e:
        print(f'Cross-validation failed: {e}')
        print('Proceeding with simple train/test split evaluation...')

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
    p = argparse.ArgumentParser(description='Train tendinopathy classifier with proper handling of small datasets')
    p.add_argument('--summary', default='dataset/PainModel_Summary_AllSubjects_2 (1).csv')
    p.add_argument('--out', default='model.joblib')
    p.add_argument('--test-out', default='test_results.csv')
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--model', choices=['rf', 'logistic'], default='rf', 
                   help='Model type: rf (Random Forest) or logistic (Logistic Regression, better for small data)')
    p.add_argument('--use-smote', action='store_true', 
                   help='Use SMOTE for training data augmentation (requires imbalanced-learn)')
    p.add_argument('--save-full', action='store_true', help='Also train on full dataset and save a second model')
    p.add_argument('--full-out', default='model_full_data.joblib')
    args = p.parse_args()
    main(args)
