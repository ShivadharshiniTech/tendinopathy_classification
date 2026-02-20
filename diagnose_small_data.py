"""
Diagnostic script for small dataset analysis.
Shows learning curves, permutation tests, and realistic performance estimates.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, permutation_test_score, LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils import load_summary_csv
from train_and_save_model import prepare_data
import warnings
warnings.filterwarnings('ignore')


def plot_learning_curve(estimator, X, y, groups=None, title="Learning Curve"):
    """Generate learning curve to show model performance vs training size."""
    train_sizes = np.linspace(0.3, 1.0, 5)
    
    if groups is not None:
        # For grouped data, use LeaveOneGroupOut
        cv = LeaveOneGroupOut()
    else:
        cv = 5
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, 
        groups=groups,
        cv=cv, 
        train_sizes=train_sizes,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', color='r', label='Training score')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.plot(train_sizes_abs, test_mean, 'o-', color='g', label='CV score')
    plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    
    plt.xlabel('Training examples')
    plt.ylabel('ROC AUC Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    # Highlight gap
    gap = train_mean[-1] - test_mean[-1]
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label='Chance')
    plt.text(0.5, 0.05, f'Train-Test Gap: {gap:.3f}\n(Large gap = overfitting)', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=150)
    print('Saved learning_curve.png')
    plt.close()
    
    return train_mean, test_mean


def run_permutation_test(estimator, X, y, groups=None, n_permutations=100):
    """Run permutation test to check if performance is better than chance."""
    print('\n=== Permutation Test ===')
    print(f'Running {n_permutations} permutations to test against chance...')
    
    if groups is not None:
        cv = LeaveOneGroupOut()
    else:
        cv = 5
    
    score, perm_scores, pvalue = permutation_test_score(
        estimator, X, y, 
        groups=groups,
        cv=cv,
        scoring='roc_auc',
        n_permutations=n_permutations,
        n_jobs=-1,
        random_state=42
    )
    
    print(f'Actual score: {score:.3f}')
    print(f'Permutation scores: {perm_scores.mean():.3f} ± {perm_scores.std():.3f}')
    print(f'p-value: {pvalue:.4f}')
    
    if pvalue < 0.05:
        print('✓ Model is significantly better than chance (p < 0.05)')
    else:
        print('✗ Model is NOT significantly better than chance (p >= 0.05)')
        print('  This suggests the model may be overfitting or lacks signal.')
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(perm_scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(score, color='red', linestyle='--', linewidth=2, label=f'Actual score: {score:.3f}')
    plt.axvline(perm_scores.mean(), color='gray', linestyle='--', linewidth=1, label=f'Chance: {perm_scores.mean():.3f}')
    plt.xlabel('ROC AUC Score')
    plt.ylabel('Frequency')
    plt.title(f'Permutation Test (p={pvalue:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('permutation_test.png', dpi=150)
    print('Saved permutation_test.png')
    plt.close()
    
    return score, pvalue


def main():
    print('=== Small Dataset Diagnostic Tool ===\n')
    
    # Load data
    df = load_summary_csv('dataset/PainModel_Summary_AllSubjects_2 (1).csv')
    X, y, groups, features = prepare_data(df)
    
    print(f'Dataset: {len(X)} samples, {groups.nunique()} subjects')
    print(f'Features: {features}')
    print(f'Class balance: {dict(y.value_counts())}')
    print(f'\n⚠️  WARNING: With only {groups.nunique()} subjects, expect:')
    print('   - High variance in performance estimates')
    print('   - Likely overfitting')
    print('   - Poor generalization to new subjects\n')
    
    # Test both models
    models = {
        'Logistic Regression (Regularized)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42))
        ]),
        'Random Forest (Constrained)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=50, max_depth=3, class_weight='balanced', random_state=42))
        ])
    }
    
    for name, model in models.items():
        print(f'\n{"="*60}')
        print(f'Model: {name}')
        print("="*60)
        
        # Learning curve
        print('\nGenerating learning curve...')
        train_scores, test_scores = plot_learning_curve(
            model, X, y, groups=groups, 
            title=f'Learning Curve: {name}'
        )
        
        final_train = train_scores[-1]
        final_test = test_scores[-1]
        gap = final_train - final_test
        
        print(f'Final training score: {final_train:.3f}')
        print(f'Final CV score: {final_test:.3f}')
        print(f'Overfitting gap: {gap:.3f}')
        
        if gap > 0.15:
            print('⚠️  Large gap suggests severe overfitting!')
        
        # Permutation test
        run_permutation_test(model, X, y, groups=groups, n_permutations=100)
    
    print('\n' + '='*60)
    print('RECOMMENDATIONS:')
    print('='*60)
    print('1. Collect more subjects (aim for 30+ for reliable estimates)')
    print('2. Use simpler models (Logistic Regression) until more data available')
    print('3. Report confidence intervals from cross-validation')
    print('4. Consider Leave-One-Subject-Out CV as the gold standard')
    print('5. Be skeptical of any "perfect" metrics - likely indicates leakage')
    print('\nSee generated plots: learning_curve.png, permutation_test.png')


if __name__ == '__main__':
    main()
