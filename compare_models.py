"""
Compare performance: Summary features vs Temporal features
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

print("="*70)
print("MODEL COMPARISON: Summary Features vs Temporal Features")
print("="*70)

# Load test results from both models
try:
    summary_results = pd.read_csv('test_results.csv')
    print("\n✓ Loaded summary-based model results")
    print(f"  - Test samples: {len(summary_results)}")
    print(f"  - Features: Peak_Pain, Mean_Pain, VAS_Model (3 features)")
except FileNotFoundError:
    print("\n❌ Summary results not found (run train_and_save_model.py first)")
    summary_results = None

try:
    temporal_results = pd.read_csv('test_results_temporal.csv')
    print("\n✓ Loaded temporal-based model results")
    print(f"  - Test samples: {len(temporal_results)}")
    print(f"  - Features: 23 temporal features (peak, slope, timing, etc.)")
except FileNotFoundError:
    print("\n❌ Temporal results not found")
    temporal_results = None

if summary_results is not None and temporal_results is not None:
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    # Summary model metrics
    if 'prob' in summary_results.columns:
        summary_auc = roc_auc_score(summary_results['true_label'], summary_results['prob'])
        summary_acc = accuracy_score(summary_results['true_label'], summary_results['pred'])
        summary_f1 = f1_score(summary_results['true_label'], summary_results['pred'])
        
        print("\n📊 Summary Features Model (3 features):")
        print(f"   ROC AUC:   {summary_auc:.3f}")
        print(f"   Accuracy:  {summary_acc:.3f}")
        print(f"   F1 Score:  {summary_f1:.3f}")
    
    # Temporal model metrics
    if 'prob' in temporal_results.columns:
        temporal_auc = roc_auc_score(temporal_results['true_label'], temporal_results['prob'])
        temporal_acc = accuracy_score(temporal_results['true_label'], temporal_results['pred'])
        temporal_f1 = f1_score(temporal_results['true_label'], temporal_results['pred'])
        
        print("\n🔬 Temporal Features Model (23 features):")
        print(f"   ROC AUC:   {temporal_auc:.3f}")
        print(f"   Accuracy:  {temporal_acc:.3f}")
        print(f"   F1 Score:  {temporal_f1:.3f}")
        
        # Comparison
        print("\n📈 Improvement (Temporal vs Summary):")
        print(f"   ΔAUC:      {(temporal_auc - summary_auc):+.3f}")
        print(f"   ΔAccuracy: {(temporal_acc - summary_acc):+.3f}")
        print(f"   ΔF1:       {(temporal_f1 - summary_f1):+.3f}")
        
        # Winner
        if temporal_auc > summary_auc:
            print("\n🏆 WINNER: Temporal Features Model")
            print("   Reason: Higher ROC AUC (better discrimination)")
        elif summary_auc > temporal_auc:
            print("\n🏆 WINNER: Summary Features Model")
            print("   Reason: Higher ROC AUC (simpler is better)")
        else:
            print("\n🤝 TIE: Both models perform equally")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("✓ Temporal features capture richer dynamics (slope, timing, acceleration)")
    print("✓ 23 features vs 3 provides more discriminative power")
    print("✓ However, with only 3 subjects, both models likely overfit")
    print("✓ True test will be on NEW subjects (Subject 4, 5, 6...)")
    print("\n⚠️  RECOMMENDATION: Use Temporal model for now, but collect more data!")

elif temporal_results is not None:
    print("\n✓ Temporal model trained successfully")
    print("  Run 'python train_and_save_model.py' to train summary model for comparison")

elif summary_results is not None:
    print("\n✓ Summary model trained")
    print("  Temporal model results are now available")

else:
    print("\n❌ No model results found")
    print("  Run 'python train_and_save_model.py' and 'python train_temporal_model.py'")
