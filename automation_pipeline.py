"""
Automated Pipeline: C3D → Tendinopathy Prediction
End-to-end automation of the traditional workflow
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Import our custom modules
from c3d_reader import read_c3d_file
from biomechanics import process_marker_data_to_forces
from pain_model import PainModel
from extract_temporal_features import extract_temporal_features_from_event_cycle


def c3d_to_prediction_pipeline(c3d_filepath, model_obj, scaler_obj, 
                               condition='unknown', subject_id=1,
                               task=None, speed=None, intermediate_dir=None):
    print(f"[DEBUG] c3d_to_prediction_pipeline called with intermediate_dir={intermediate_dir}")
    """
    Complete automated pipeline: C3D file → Tendinopathy prediction
    
    Traditional workflow:
        C3D → Mokka → TRC → OpenSim IK → OpenSim SO → Excel → MATLAB → CSV → ML
    
    Automated workflow:
        C3D → Python biomechanics → Pain model → Feature extraction → ML prediction
    
    Args:
        c3d_filepath: Path to C3D motion capture file
        model_obj: Trained ML model dict
        scaler_obj: StandardScaler object
        condition: 'normal', 'tendon', or 'unknown' (for pain model calibration)
        subject_id: Subject identifier
        task: Task type description
        speed: Movement speed
    
    Returns:
        dict with:
            - prediction: 0 (Normal) or 1 (Tendinopathy)
            - probability: Probability of tendinopathy
            - pain_cycle_df: Pain over event cycle (101 points)
            - temporal_features: Extracted features
            - marker_data: Raw marker data
            - muscle_forces: Estimated muscle forces
    """
    print(f"\n{'='*60}")
    print("AUTOMATED C3D → TENDINOPATHY PREDICTION")
    print(f"{'='*60}")
    
    # --- Parse Task and Speed from filename (match MATLAB logic) ---
    fname = Path(c3d_filepath).name.lower()
    # Task
    if task is None:
        if 'rest' in fname:
            task_val = 'Rest'
        else:
            task_val = 'WithTask'
    else:
        task_val = task
    # Speed
    if speed is None:
        if 'fast' in fname:
            speed_val = 'Fast'
        elif 'med' in fname:
            speed_val = 'Medium'
        elif 'slow' in fname:
            speed_val = 'Slow'
        else:
            speed_val = 'Unknown'
    else:
        speed_val = speed

    # Step 1: Read C3D file
    print("\n[1/6] Reading C3D file...")
    try:
        c3d_data = read_c3d_file(c3d_filepath)
        print(f"  ✓ Loaded {len(c3d_data['marker_names'])} markers")
        print(f"  ✓ Frame rate: {c3d_data['frame_rate']:.1f} Hz")
        print(f"  ✓ Duration: {c3d_data['time'][-1]:.2f} seconds")
        print(f"  ✓ Markers: {', '.join(c3d_data['marker_names'][:5])}...")
    except Exception as e:
        raise RuntimeError(f"Failed to read C3D file: {e}")
    
    # Step 2: Estimate muscle forces from kinematics
    print("\n[2/6] Estimating muscle forces (ECU, ECRL, ECRB)...")
    try:
        # Use best guess for condition if unknown
        force_condition = condition if condition in ['normal', 'tendon'] else 'normal'
        
        forces_df = process_marker_data_to_forces(
            c3d_data['markers'],
            c3d_data['marker_names'],
            task_type=force_condition
        )
        print(f"  ✓ Estimated forces for {len(forces_df)} time points")
        print(f"  ✓ Mean ECU: {forces_df['ECU_N'].mean():.1f}N")
        print(f"  ✓ Mean ECRL: {forces_df['ECRL_N'].mean():.1f}N")
        print(f"  ✓ Mean ECRB: {forces_df['ECRB_N'].mean():.1f}N")
    except Exception as e:
        raise RuntimeError(f"Failed to estimate muscle forces: {e}")
    
    # Step 3: Apply pain model
    print("\n[3/6] Applying biomechanical pain model...")
    try:
        pain_model = PainModel()
        # Pass intermediate_dir to save all arrays for debugging
        pain_cycle_df = pain_model.process_muscle_forces(
            forces_df,
            condition=force_condition,
            subject_id=subject_id,
            task=task_val,
            speed=speed_val,
            intermediate_dir=intermediate_dir
        )
        print(f"  ✓ Computed pain over 101-point event cycle")
        print(f"  ✓ Peak pain: {pain_cycle_df['Pain_pred'].max():.2f}/10")
        print(f"  ✓ Mean pain: {pain_cycle_df['Pain_pred'].mean():.2f}/10")
    except Exception as e:
        raise RuntimeError(f"Failed to compute pain model: {e}")
    
    # Step 4: Extract temporal features
    print("\n[4/6] Extracting temporal features...")
    try:
        # Create temporary mini dataset (single trial)
        temp_df = pain_cycle_df.copy()
        
        # Extract features
        features = extract_temporal_features_from_event_cycle(temp_df)
        
        if features is None or len(features) == 0:
            raise ValueError("Feature extraction returned empty result")
        
        print(f"  ✓ Extracted {len(features.columns) - 4} temporal features")
        print(f"  ✓ Sample features: peak_pain={features['peak_pain'].iloc[0]:.2f}, "
              f"mean_pain={features['mean_pain'].iloc[0]:.2f}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract features: {e}")
    
    # Step 5: Make prediction
    print("\n[5/6] Making ML prediction...")
    try:
        model = model_obj['model']
        scaler = model_obj.get('scaler', scaler_obj)
        feature_names = model_obj['features']
        
        # Prepare feature vector
        X = features[feature_names].values
        X_scaled = scaler.transform(X)
        
        # Predict
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        prediction = int(y_pred[0])
        probability = float(y_pred_proba[0])
        
        result_str = "TENDINOPATHY" if prediction == 1 else "NORMAL"
        print(f"  ✓ Prediction: {result_str}")
        print(f"  ✓ Confidence: {probability:.1%}")
    except Exception as e:
        raise RuntimeError(f"Failed to make prediction: {e}")
    
    # Step 6: Package results and save intermediates
    print("\n[6/6] Packaging results and saving intermediates...")

    # Create intermediate output directory for this file
    input_base = Path(c3d_filepath).stem
    intermediate_dir = Path("intermediate") / input_base
    os.makedirs(intermediate_dir, exist_ok=True)

    # Save event-cycle data
    pain_cycle_path = intermediate_dir / "event_cycle.csv"
    pain_cycle_df.to_csv(pain_cycle_path, index=False)
    print(f"  ✓ Saved event-cycle data to {pain_cycle_path}")

    # Save temporal features
    temporal_features_path = intermediate_dir / "temporal_features.csv"
    features.to_csv(temporal_features_path, index=False)
    print(f"  ✓ Saved temporal features to {temporal_features_path}")

    results = {
        'prediction': prediction,
        'prediction_label': result_str,
        'probability': probability,
        'confidence': probability if prediction == 1 else (1 - probability),
        'pain_cycle_df': pain_cycle_df,
        'temporal_features': features,
        'marker_data': c3d_data,
        'muscle_forces': forces_df,
        'metadata': {
            'subject_id': subject_id,
            'condition': condition,
            'task': task,
            'speed': speed,
            'n_markers': len(c3d_data['marker_names']),
            'duration_s': c3d_data['time'][-1],
            'peak_pain': float(pain_cycle_df['Pain_pred'].max()),
            'mean_pain': float(pain_cycle_df['Pain_pred'].mean())
        }
    }

    print(f"\n{'='*60}")
    print(f"  🎯 FINAL PREDICTION: {result_str} ({probability:.1%} confidence)")
    print(f"{'='*60}\n")

    return results


def batch_process_c3d_files(c3d_files, model_obj, scaler_obj, output_dir='results'):
    """
    Process multiple C3D files in batch
    
    Args:
        c3d_files: List of C3D file paths
        model_obj: Trained ML model
        scaler_obj: Feature scaler
        output_dir: Directory to save results
    
    Returns:
        DataFrame with predictions for all files
    """
    results_list = []
    
    for i, c3d_path in enumerate(c3d_files):
        print(f"\n\nProcessing file {i+1}/{len(c3d_files)}: {Path(c3d_path).name}")
        
        try:
            result = c3d_to_prediction_pipeline(
                c3d_path,
                model_obj,
                scaler_obj,
                subject_id=i+1
            )
            
            results_list.append({
                'file': Path(c3d_path).name,
                'prediction': result['prediction_label'],
                'probability': result['probability'],
                'confidence': result['confidence'],
                'peak_pain': result['metadata']['peak_pain'],
                'mean_pain': result['metadata']['mean_pain']
            })
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results_list.append({
                'file': Path(c3d_path).name,
                'prediction': 'ERROR',
                'probability': None,
                'confidence': None,
                'peak_pain': None,
                'mean_pain': None,
                'error': str(e)
            })
    
    # Save results
    results_df = pd.DataFrame(results_list)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / 'batch_predictions.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n\nBatch results saved to: {output_path}")
    
    return results_df


if __name__ == '__main__':
    import sys
    import joblib
    
    # Load trained model
    print("Loading trained model...")
    try:
        model_data = joblib.load('model_temporal.joblib')
        scaler_data = model_data.get('scaler')
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model file not found. Train model first: python train_temporal_model.py")
        sys.exit(1)
    
    # Process C3D file if provided
    if len(sys.argv) > 1:
        c3d_path = sys.argv[1]
        
        if not os.path.exists(c3d_path):
            print(f"❌ File not found: {c3d_path}")
            sys.exit(1)
        
        result = c3d_to_prediction_pipeline(
            c3d_path,
            model_data,
            scaler_data
        )
        
        print("\nResults summary:")
        print(f"  Prediction: {result['prediction_label']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Peak pain: {result['metadata']['peak_pain']:.2f}/10")
    else:
        print("\nUsage: python automation_pipeline.py <c3d_file>")
        print("Example: python automation_pipeline.py data/subject1_trial1.c3d")
