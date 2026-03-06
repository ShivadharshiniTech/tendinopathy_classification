"""
Train model on temporal features extracted from EventCycle data
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, brier_score_loss
)
import argparse
import warnings
warnings.filterwarnings('ignore')

def train_temporal_model(model_type='logistic', test_size=0.25, random_state=42):
    """Train model on temporal features"""
    
    # Load temporal features
    features_df = pd.read_csv('dataset/temporal_features.csv')
    
    print(f"{'='*60}")
    print(f"TEMPORAL FEATURE MODEL TRAINING")
    print(f"{'='*60}")
    print(f"Loaded {len(features_df)} trials from temporal features")
    print(f"Subjects: {features_df['Subject'].nunique()}")
    print(f"Trials per subject: {features_df.groupby('Subject').size().to_dict()}")
    
    # Define features (exclude metadata)
    exclude_cols = ['Subject', 'Condition', 'Task', 'Speed', 'true_label']
    FEATURES = [col for col in features_df.columns if col not in exclude_cols]
    
    print(f"\nUsing {len(FEATURES)} features")
    
    # Prepare data
    X = features_df[FEATURES].values
    y = features_df['true_label'].values
    groups = features_df['Subject'].values
    
    n_subjects = len(set(groups))
    print(f"\nDataset: {len(X)} trials from {n_subjects} subjects")
    print(f"Class balance: {dict(pd.Series(y).value_counts())}")
    
    # Group-aware train-test split
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    groups_test = groups[test_idx]
    
    print(f"\n{'='*60}")
    print("TRAIN-TEST SPLIT (Group-Aware by Subject)")
    print(f"{'='*60}")
    print(f"Train: {len(X_train)} trials from subjects {sorted(set(groups_train))}")
    print(f"Test:  {len(X_test)} trials from subjects {sorted(set(groups_test))}")
    print(f"Train classes: {dict(pd.Series(y_train).value_counts())}")
    print(f"Test classes:  {dict(pd.Series(y_test).value_counts())}")
    
    # Verify no subject overlap
    train_subs = set(groups_train)
    test_subs = set(groups_test)
    assert train_subs & test_subs == set(), "❌ Subject leakage detected!"
    print("\n✓ No subject leakage verified (proper group-aware split)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize model
    if model_type == 'logistic':
        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced',
            random_state=random_state
        )
        print(f"\n✓ Using Logistic Regression (L2 regularization, C=1.0)")
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state
        )
        print(f"\n✓ Using Random Forest (constrained: max_depth=5)")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Cross-validation (LOGO-CV for very small subject count)
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION: Leave-One-Group-Out (LOGO-CV)")
    print(f"{'='*60}")
    
    logo = LeaveOneGroupOut()
    
    # Multiple metrics
    scoring = {
        'roc_auc': 'roc_auc',
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    
    # Perform CV on full dataset (use training scaler for consistency)
    # NOTE: transform only, don't refit the scaler!
    X_all_scaled = scaler.transform(X)
    cv_results = cross_validate(
        model, X_all_scaled, y, 
        groups=groups,
        cv=logo, 
        scoring=scoring,
        n_jobs=-1
    )
    
    # Display fold-by-fold results
    unique_subjects = sorted(set(groups))
    print(f"\nFold-by-fold results ({len(unique_subjects)} folds):")
    for i, subj in enumerate(unique_subjects):
        print(f"  Fold {i+1} (test Subject {subj}): "
              f"AUC={cv_results['test_roc_auc'][i]:.3f}, "
              f"Acc={cv_results['test_accuracy'][i]:.3f}, "
              f"F1={cv_results['test_f1'][i]:.3f}")
    
    print(f"\nCross-Validation Summary (mean ± std):")
    for metric in scoring.keys():
        scores = cv_results[f'test_{metric}']
        print(f"  {metric:20s}: {scores.mean():.3f} ± {scores.std():.3f}")
    
    print(f"\n⚠️  NOTE: With only {n_subjects} subjects, CV estimates have HIGH VARIANCE")
    print(f"⚠️  Generalization to new subjects is UNKNOWN")
    
    # Test set evaluation
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_brier = brier_score_loss(y_test, y_pred_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0
    
    print(f"\n{'='*60}")
    print("HELD-OUT TEST SET EVALUATION")
    print(f"{'='*60}")
    print(f"ROC AUC:        {test_auc:.3f}")
    print(f"Accuracy:       {test_acc:.3f}")
    print(f"Balanced Acc:   {((test_rec + specificity) / 2):.3f}")
    print(f"Precision:      {test_prec:.3f}")
    print(f"Recall (Sens):  {test_rec:.3f}")
    print(f"Specificity:    {specificity:.3f}")
    print(f"F1 Score:       {test_f1:.3f}")
    print(f"Brier Score:    {test_brier:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Tendon")
    if cm.shape[0] == 2 and cm.shape[1] == 2:
        print(f"Actual Normal    {cm[0,0]:3d}     {cm[0,1]:3d}")
        print(f"       Tendon    {cm[1,0]:3d}     {cm[1,1]:3d}")
    else:
        print(cm)
    
    # Feature importance (for Logistic Regression)
    if model_type == 'logistic':
        feature_importance = abs(model.coef_[0])
        importance_df = pd.DataFrame({
            'feature': FEATURES,
            'coefficient': model.coef_[0],
            'abs_importance': feature_importance
        }).sort_values('abs_importance', ascending=False)
        
        print(f"\n{'='*60}")
        print("TOP 10 MOST IMPORTANT FEATURES (Logistic Regression)")
        print(f"{'='*60}")
        for idx, row in importance_df.head(10).iterrows():
            direction = "↑" if row['coefficient'] > 0 else "↓"
            print(f"{row['feature']:25s} {direction} {row['abs_importance']:.3f}")
    
    # Save artifacts
    joblib.dump({'model': model, 'scaler': scaler, 'features': FEATURES}, 'model_temporal.joblib')
    joblib.dump(scaler, 'scaler_temporal.joblib')  # Also save separately for backward compatibility
    
    # Create results directory if needed
    from pathlib import Path
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Save test results
    test_df = features_df.iloc[test_idx].copy()
    test_df['pred'] = y_pred
    test_df['prob'] = y_pred_proba
    test_df.to_csv(results_dir / 'test_results_temporal.csv', index=False)
    
    # Save clean test set (for external validation)
    test_clean = test_df[['Subject', 'Condition', 'Task', 'Speed'] + FEATURES + ['true_label']]
    test_clean.to_csv(results_dir / 'test_set_temporal.csv', index=False)
    
    print(f"\n{'='*60}")
    print("ARTIFACTS SAVED")
    print(f"{'='*60}")
    print(f"✓ Model + features: model_temporal.joblib")
    print(f"✓ Scaler:           scaler_temporal.joblib")
    print(f"✓ Test predictions: results/test_results_temporal.csv")
    print(f"✓ Test set (clean): results/test_set_temporal.csv")
    
    # Final warning
    print(f"\n{'⚠'*30}")
    print(f"⚠ CRITICAL LIMITATION: Only {n_subjects} subjects in dataset")
    print(f"⚠ Perfect/near-perfect metrics likely indicate memorization")
    print(f"⚠ Model will likely perform worse on new subjects")
    print(f"⚠ RECOMMENDATION: Collect 30-50 subjects minimum")
    print(f"{'⚠'*30}")
    
    return model, scaler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on temporal features from EventCycle data')
    parser.add_argument('--model', choices=['logistic', 'rf'], default='logistic',
                        help='Model type: logistic (recommended) or rf')
    parser.add_argument('--test-size', type=float, default=0.25,
                        help='Test set proportion (default: 0.25)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    train_temporal_model(
        model_type=args.model,
        test_size=args.test_size,
        random_state=args.random_state
    )
