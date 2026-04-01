"""
Pain Model - Port of MATLAB pain prediction model
Converts muscle force time series to pain predictions over event cycle
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import medfilt
from scipy.interpolate import interp1d


class PainModel:
    """
    Biomechanical pain model for tendinopathy assessment
    Based on force-pain relationships and damage accumulation
    """
    
    def __init__(self):
        # Default pain model parameters (from MATLAB)
        self.theta = 0.85      # Force threshold
        self.beta = 1.5        # Sensitization rate
        self.gamma = 0.8       # Recovery rate
        self.lambda_param = 4.0  # Sensitization multiplier
        self.k_rate = 0.15     # Rate of change sensitivity
        self.N50 = 0.12        # Pain threshold (nociceptive input)
        self.a = 60.0          # Sigmoid slope
        
        # Muscle weights for combined force
        self.w_ECU = 0.2
        self.w_ECRL = 0.3
        self.w_ECRB = 0.5
        
        # Event cycle parameters
        self.thresh_frac = 0.15  # Force threshold for event detection
        self.n_cycle_pts = 101   # Number of points in normalized cycle
        
        # Damage memory (for sequential processing)
        self.damage_memory = {}
    
    def compute_combined_force(self, ECU, ECRL, ECRB):
        """Combine muscle forces using weighted sum"""
        F = self.w_ECU * ECU + self.w_ECRL * ECRL + self.w_ECRB * ECRB
        return F
    
    def normalize_to_event_cycle(self, force_signal):
        """
        Extract and normalize force signal to 0-100% event cycle
        
        Args:
            force_signal: Raw force time series
        
        Returns:
            Normalized force over 101 points (0-100%)
        """
        # Normalize force to [0, 1]
        F_norm = force_signal / (np.max(force_signal) + 1e-10)
        
        # Find event boundaries (where force > threshold)
        idx = np.where(F_norm > self.thresh_frac)[0]
        
        if len(idx) == 0:
            # No significant force detected, return zeros
            return np.zeros(self.n_cycle_pts)
        
        # Extract event segment
        Fe = F_norm[idx[0]:idx[-1]+1]
        
        if len(Fe) < 2:
            return np.zeros(self.n_cycle_pts)
        
        # Interpolate to 101 points (0-100%)
        original_cycle = np.linspace(0, 100, len(Fe))
        target_cycle = np.linspace(0, 100, self.n_cycle_pts)
        
        interp_func = interp1d(original_cycle, Fe, kind='linear', 
                               bounds_error=False, fill_value='extrapolate')
        Fe_cycle = interp_func(target_cycle)
        
        # Ensure non-negative
        Fe_cycle = np.maximum(Fe_cycle, 0)
        
        # Apply median filter for smoothing (window size 5) with edge padding to match MATLAB movmedian
        pad_width = 2  # (kernel_size - 1) // 2
        Fe_cycle_padded = np.pad(Fe_cycle, (pad_width, pad_width), mode='edge')
        Fe_cycle_smoothed = medfilt(Fe_cycle_padded, kernel_size=5)
        Fe_cycle = Fe_cycle_smoothed[pad_width:-pad_width]
        
        return Fe_cycle
    
    def compute_pain(self, Fe_cycle, condition='normal', subject_id=1):
        """
        Compute pain prediction from normalized force cycle
        
        Args:
            Fe_cycle: Normalized force over event cycle (101 points)
            condition: 'normal' or 'tendon'
            subject_id: Subject identifier for damage tracking
        
        Returns:
            DataFrame with EventCycle (0-100) and Pain_pred (0-10)
        """
        # Condition-dependent parameters
        if condition == 'tendon':
            theta_eff = 0.75
            lambda_eff = 4.5
            N50_eff = 0.15
        else:
            theta_eff = self.theta
            lambda_eff = self.lambda_param
            N50_eff = self.N50
        
        # Compute force rate of change
        dF = np.diff(Fe_cycle, prepend=0)
        
        # Stimulus (overload + rate of change)
        S = np.maximum(Fe_cycle - theta_eff, 0) + self.k_rate * np.abs(dF)
        
        # Sensitization state (temporal integration)
        x = np.zeros(len(S))
        for i in range(1, len(S)):
            x[i] = max(x[i-1] + (self.beta * S[i-1] - self.gamma * x[i-1]), 0)
        
        # Nociceptive input (stimulus amplified by sensitization)
        N = (1 + lambda_eff * x) * S
        
        # Pain output (sigmoid transformation)
        Pain = 10.0 / (1 + np.exp(-self.a * (N - N50_eff)))
        Pain = np.clip(Pain, 0, 10)
        
        # Damage accumulation (for sequential trials)
        subject_key = f"S{subject_id}_{condition}"
        if subject_key not in self.damage_memory:
            self.damage_memory[subject_key] = 0
        
        overload = np.mean(np.maximum(Fe_cycle - theta_eff, 0))
        rep_load = np.trapz(Fe_cycle)
        
        self.damage_memory[subject_key] += 0.4 * overload + 0.02 * rep_load
        
        # Create output DataFrame
        event_cycle = np.linspace(0, 100, self.n_cycle_pts)
        
        result_df = pd.DataFrame({
            'EventCycle': event_cycle,
            'Pain_pred': Pain
        })
        
        return result_df
    
    def process_muscle_forces(self, forces_df, condition='normal', subject_id=1, 
                              task='WithTask', speed='Medium', intermediate_dir=None):
        """
        Complete pipeline: muscle forces → pain prediction
        
        Args:
            forces_df: DataFrame with Time_s, ECU_N, ECRL_N, ECRB_N
            condition: 'normal' or 'tendon'
            subject_id: Subject ID
            task: Task description
            speed: Movement speed
        
        Returns:
            DataFrame with Subject, Condition, Task, Speed, EventCycle, Pain_pred
        """
        print(f"[DEBUG] process_muscle_forces called with intermediate_dir={intermediate_dir}")
        # Compute combined force
        F_combined = self.compute_combined_force(
            forces_df['ECU_N'].values,
            forces_df['ECRL_N'].values,
            forces_df['ECRB_N'].values
        )

        # Ensure intermediate_dir exists if requested
        if intermediate_dir is not None:
            Path(intermediate_dir).mkdir(parents=True, exist_ok=True)
            print(f"[DEBUG] Saving intermediates to: {intermediate_dir}")

        # Save F_combined if requested
        if intermediate_dir is not None:
            pd.DataFrame({'F_combined': F_combined}).to_csv(
                Path(intermediate_dir) / 'F_combined.csv', index=False)
            print(f"[DEBUG] Saved F_combined.csv ({len(F_combined)} values)")

        # --- Normalize to event cycle (save all intermediates) ---
        # Normalize force to [0, 1]
        F_norm = F_combined / (np.max(F_combined) + 1e-10)
        if intermediate_dir is not None:
            pd.DataFrame({'F_norm': F_norm}).to_csv(
                Path(intermediate_dir) / 'F_norm.csv', index=False)
            print(f"[DEBUG] Saved F_norm.csv ({len(F_norm)} values)")

        # Find event boundaries (where force > threshold)
        idx = np.where(F_norm > self.thresh_frac)[0]
        if intermediate_dir is not None:
            pd.DataFrame({'idx': idx}).to_csv(
                Path(intermediate_dir) / 'event_indices.csv', index=False)
            print(f"[DEBUG] Saved event_indices.csv ({len(idx)} values)")

        if len(idx) == 0:
            Fe = np.zeros(self.n_cycle_pts)
        else:
            Fe = F_norm[idx[0]:idx[-1]+1]
        if intermediate_dir is not None:
            pd.DataFrame({'Fe': Fe}).to_csv(
                Path(intermediate_dir) / 'Fe.csv', index=False)
            print(f"[DEBUG] Saved Fe.csv ({len(Fe)} values)")

        # Interpolate to 101 points (0-100%)
        if len(Fe) < 2:
            Fe_cycle = np.zeros(self.n_cycle_pts)
        else:
            original_cycle = np.linspace(0, 100, len(Fe))
            target_cycle = np.linspace(0, 100, self.n_cycle_pts)
            from scipy.interpolate import interp1d
            interp_func = interp1d(original_cycle, Fe, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
            Fe_cycle = interp_func(target_cycle)
            Fe_cycle = np.maximum(Fe_cycle, 0)
            from scipy.signal import medfilt
            Fe_cycle = medfilt(Fe_cycle, kernel_size=5)
        if intermediate_dir is not None:
            pd.DataFrame({'Fe_cycle': Fe_cycle}).to_csv(
                Path(intermediate_dir) / 'Fe_cycle.csv', index=False)
            print(f"[DEBUG] Saved Fe_cycle.csv ({len(Fe_cycle)} values)")

        # Compute pain
        pain_df = self.compute_pain(Fe_cycle, condition=condition, subject_id=subject_id)
        if intermediate_dir is not None:
            pd.DataFrame({'Pain': pain_df['Pain_pred'].values}).to_csv(
                Path(intermediate_dir) / 'Pain.csv', index=False)
            print(f"[DEBUG] Saved Pain.csv ({len(pain_df['Pain_pred'].values)} values)")

        # Add metadata
        pain_df.insert(0, 'Subject', subject_id)
        pain_df.insert(1, 'Condition', condition)
        pain_df.insert(2, 'Task', task)
        pain_df.insert(3, 'Speed', speed)

        return pain_df
    
    def reset_damage_memory(self):
        """Reset damage accumulation (for new analysis session)"""
        self.damage_memory = {}


def process_c3d_to_pain_cycle(c3d_data, condition='normal', subject_id=1, 
                               task='WithTask', speed='Medium'):
    """
    Convenience function: C3D data → Pain cycle in one call
    
    Args:
        c3d_data: Dict from c3d_reader.read_c3d_file()
        condition: 'normal' or 'tendon'
        subject_id: Subject ID
        task: Task type
        speed: Movement speed
    
    Returns:
        DataFrame with pain predictions over event cycle
    """
    from biomechanics import process_marker_data_to_forces
    
    # Step 1: Marker data → Muscle forces
    forces_df = process_marker_data_to_forces(
        c3d_data['markers'],
        c3d_data['marker_names'],
        task_type=condition
    )
    
    # Step 2: Muscle forces → Pain predictions
    pain_model = PainModel()
    pain_cycle_df = pain_model.process_muscle_forces(
        forces_df,
        condition=condition,
        subject_id=subject_id,
        task=task,
        speed=speed
    )
    
    return pain_cycle_df


if __name__ == '__main__':
    # Test with synthetic data
    print("Testing Pain Model...")
    
    # Create synthetic muscle force data
    n_frames = 200
    time_s = np.arange(n_frames) / 100.0
    
    # Simulate force profile (bell curve)
    t_norm = np.linspace(0, 1, n_frames)
    ECU = 20 + 30 * np.exp(-((t_norm - 0.5) ** 2) / 0.05)
    ECRL = 25 + 35 * np.exp(-((t_norm - 0.5) ** 2) / 0.05)
    ECRB = 22 + 33 * np.exp(-((t_norm - 0.5) ** 2) / 0.05)
    
    forces_df = pd.DataFrame({
        'Time_s': time_s,
        'FUN_N': np.zeros(n_frames),
        'ECU_N': ECU,
        'ECRL_N': ECRL,
        'ECRB_N': ECRB
    })
    
    # Process
    pain_model = PainModel()
    result = pain_model.process_muscle_forces(forces_df, condition='tendon')
    
    print("\nPain prediction summary:")
    print(result['Pain_pred'].describe())
    print(f"\nPeak pain: {result['Pain_pred'].max():.2f}/10")
    print(f"Mean pain: {result['Pain_pred'].mean():.2f}/10")
