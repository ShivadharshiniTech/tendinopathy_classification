"""
Extract rich temporal features from EventCycle data
"""
import pandas as pd
import numpy as np

def load_and_extract_features():
    """Load EventCycle CSV and extract temporal features"""
    
    # Load data
    event_df = pd.read_csv('dataset/PainModel_EventCycle_AllSubjects_1 (1).csv')
    
    print(f"✓ Loaded {len(event_df)} EventCycle rows")
    print(f"✓ Unique subjects: {event_df['Subject'].nunique()}")
    print(f"✓ Conditions: {event_df['Condition'].unique()}")
    
    # Check available columns
    print(f"✓ Columns: {list(event_df.columns)}")
    
    # Group by trial (Subject × Condition × Task × Speed)
    trials = []
    
    for (subj, cond, task, speed), group in event_df.groupby(
        ['Subject', 'Condition', 'Task', 'Speed']
    ):
        pain = group['Pain_pred'].values
        
        # Basic statistics
        features = {
            'Subject': subj,
            'Condition': cond,
            'Task': task,
            'Speed': speed,
            
            # Central tendency
            'peak_pain': np.max(pain),
            'mean_pain': np.mean(pain),
            'median_pain': np.median(pain),
            'min_pain': np.min(pain),
            
            # Dispersion
            'std_pain': np.std(pain),
            'pain_range': np.max(pain) - np.min(pain),
            'pain_cv': np.std(pain) / np.mean(pain) if np.mean(pain) > 0 else 0,
            'iqr_pain': np.percentile(pain, 75) - np.percentile(pain, 25),
            
            # Percentiles
            'percentile_95': np.percentile(pain, 95),
            'percentile_75': np.percentile(pain, 75),
            'percentile_25': np.percentile(pain, 25),
            'percentile_5': np.percentile(pain, 5),
            
            # Temporal dynamics
            'time_to_peak': np.argmax(pain) / len(pain) if len(pain) > 0 else 0,  # Normalized to [0,1]
            'pain_slope': np.polyfit(range(len(pain)), pain, 1)[0] if len(pain) > 1 else 0,
            'pain_curvature': np.polyfit(range(len(pain)), pain, 2)[0] if len(pain) > 2 else 0,
            
            # Phase-based (divide cycle into early/mid/late)
            'early_pain': np.mean(pain[:len(pain)//3]) if len(pain) >= 3 else np.mean(pain),
            'mid_pain': np.mean(pain[len(pain)//3:2*len(pain)//3]) if len(pain) >= 3 else np.mean(pain),
            'late_pain': np.mean(pain[2*len(pain)//3:]) if len(pain) >= 3 else np.mean(pain),
            
            # Derived metrics
            'pain_acceleration': np.mean(np.diff(np.diff(pain))) if len(pain) > 2 else 0,
            'pain_skewness': pd.Series(pain).skew(),
            'pain_kurtosis': pd.Series(pain).kurtosis(),
            
            # Onset/offset
            'onset_rate': (pain[10] - pain[0]) / 10 if len(pain) > 10 else 0,
            'offset_rate': (pain[-1] - pain[-10]) / 10 if len(pain) > 10 else 0,
        }
        
        trials.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(trials)
    
    # Add target label
    features_df['true_label'] = (features_df['Condition'] == 'tendon').astype(int)
    
    print(f"\n✓ Extracted features for {len(features_df)} trials")
    print(f"✓ Features per trial: {len(features_df.columns) - 5}")  # Exclude metadata
    print(f"✓ Class distribution:")
    print(features_df['Condition'].value_counts())
    
    return features_df

def save_features(features_df, filename='dataset/temporal_features.csv'):
    """Save extracted features"""
    features_df.to_csv(filename, index=False)
    print(f"\n✓ Saved features to {filename}")

if __name__ == '__main__':
    # Extract and save
    features_df = load_and_extract_features()
    save_features(features_df)
    
    # Display sample
    print("\nSample features (first 3 trials):")
    feature_cols = [col for col in features_df.columns 
                    if col not in ['Subject', 'Condition', 'Task', 'Speed', 'true_label']]
    print(features_df[['Subject', 'Condition', 'Task', 'Speed'] + feature_cols[:5]].head(3))
    
    # Feature correlation with target
    print(f"\nTop 10 features correlated with target (tendon vs normal):")
    correlations = features_df[feature_cols].corrwith(features_df['true_label']).abs().sort_values(ascending=False)
    print(correlations.head(10))
