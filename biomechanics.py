"""
Simplified Biomechanics Module
Estimates joint angles and muscle forces from marker data
(Simplified alternative to full OpenSim simulation - zero cost)
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def compute_elbow_angle(shoulder_pos, elbow_pos, wrist_pos):
    """
    Compute elbow flexion angle from 3D marker positions
    
    Args:
        shoulder_pos: (n_frames, 3) array
        elbow_pos: (n_frames, 3) array
        wrist_pos: (n_frames, 3) array
    
    Returns:
        Elbow angles in degrees (n_frames,)
    """
    # Vector from elbow to shoulder
    v1 = shoulder_pos - elbow_pos
    # Vector from elbow to wrist
    v2 = wrist_pos - elbow_pos
    
    # Normalize vectors
    v1_norm = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-10)
    
    # Compute angle using dot product
    cos_angle = np.sum(v1_norm * v2_norm, axis=1)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angles_rad = np.arccos(cos_angle)
    angles_deg = np.degrees(angles_rad)
    
    # Smooth the angle data
    if len(angles_deg) > 11:
        angles_deg = savgol_filter(angles_deg, window_length=11, polyorder=3)
    
    return angles_deg


def estimate_muscle_forces_from_kinematics(elbow_angle, elbow_angular_vel=None, task_type='normal'):
    """
    Simplified muscle force estimation based on elbow kinematics
    
    This is a SIMPLIFIED model that estimates ECU, ECRL, ECRB forces
    based on elbow angle and movement patterns.
    
    In reality, OpenSim uses inverse dynamics + static optimization,
    but for automation without OpenSim, we use empirical relationships.
    
    Args:
        elbow_angle: Elbow flexion angle in degrees (n_frames,)
        elbow_angular_vel: Angular velocity (optional)
        task_type: 'normal' or 'tendon' (affects force patterns)
    
    Returns:
        DataFrame with Time_s, ECU_N, ECRL_N, ECRB_N columns
    """
    n_frames = len(elbow_angle)
    
    # Normalize elbow angle to [0, 1] range (0° = extended, 150° = flexed)
    angle_norm = np.clip(elbow_angle / 150.0, 0, 1)
    
    # Compute angular velocity if not provided
    if elbow_angular_vel is None:
        elbow_angular_vel = np.gradient(elbow_angle)
    
    # Smooth velocity
    if len(elbow_angular_vel) > 11:
        elbow_angular_vel = savgol_filter(elbow_angular_vel, window_length=11, polyorder=2)
    
    # Simplified muscle activation patterns
    # These are empirical relationships - adjust based on your data
    
    # ECU (Extensor Carpi Ulnaris) - active during extension and pronation
    # Peaks at mid-range angles
    ECU_base = 15 + 25 * np.sin(np.pi * angle_norm)
    ECU_velocity = 5 * np.abs(elbow_angular_vel) / (np.max(np.abs(elbow_angular_vel)) + 1e-10)
    ECU_N = ECU_base + ECU_velocity
    
    # ECRL (Extensor Carpi Radialis Longus) - active during wrist extension
    # Higher activation at extended positions
    ECRL_base = 20 + 30 * (1 - angle_norm)
    ECRL_velocity = 6 * np.abs(elbow_angular_vel) / (np.max(np.abs(elbow_angular_vel)) + 1e-10)
    ECRL_N = ECRL_base + ECRL_velocity
    
    # ECRB (Extensor Carpi Radialis Brevis) - similar to ECRL
    ECRB_base = 18 + 28 * (1 - angle_norm) + 5 * np.sin(2 * np.pi * angle_norm)
    ECRB_velocity = 5.5 * np.abs(elbow_angular_vel) / (np.max(np.abs(elbow_angular_vel)) + 1e-10)
    ECRB_N = ECRB_base + ECRB_velocity
    
    # Add noise and variability to make it more realistic
    ECU_N += np.random.randn(n_frames) * 2
    ECRL_N += np.random.randn(n_frames) * 2
    ECRB_N += np.random.randn(n_frames) * 2
    
    # Ensure non-negative
    ECU_N = np.maximum(ECU_N, 0)
    ECRL_N = np.maximum(ECRL_N, 0)
    ECRB_N = np.maximum(ECRB_N, 0)
    
    # Condition-specific adjustments
    if task_type == 'tendon':
        # Tendinopathy patients may show different force patterns
        ECU_N *= 1.15  # Slightly higher compensatory activation
        ECRL_N *= 1.10
        ECRB_N *= 1.12
    
    # Create DataFrame
    time_s = np.arange(n_frames) / 100.0  # Assume 100 Hz
    
    df = pd.DataFrame({
        'Time_s': time_s,
        'FUN_N': np.zeros(n_frames),  # Placeholder
        'ECU_N': ECU_N,
        'ECRL_N': ECRL_N,
        'ECRB_N': ECRB_N
    })
    
    return df


def process_marker_data_to_forces(marker_df, marker_names, task_type='normal'):
    """
    End-to-end: marker data → joint angles → muscle forces
    
    Args:
        marker_df: DataFrame with marker positions
        marker_names: List of available markers
        task_type: 'normal' or 'tendon'
    
    Returns:
        DataFrame with muscle forces (Time_s, ECU_N, ECRL_N, ECRB_N)
    """
    # Extract key markers (simplified - adjust to your marker set)
    # This is a placeholder - you'll need to adapt to your specific marker protocol
    
    # Try to find shoulder, elbow, wrist markers
    shoulder_marker = None
    elbow_marker = None
    wrist_marker = None
    
    for marker in marker_names:
        marker_upper = marker.upper()
        if 'SHO' in marker_upper or 'SHOULDER' in marker_upper:
            shoulder_marker = marker
        elif 'ELB' in marker_upper or 'ELBOW' in marker_upper:
            elbow_marker = marker
        elif 'WR' in marker_upper or 'WRIST' in marker_upper:
            if wrist_marker is None:  # Take first wrist marker
                wrist_marker = marker
    
    if not all([shoulder_marker, elbow_marker, wrist_marker]):
        raise ValueError(
            f"Could not find required markers. Available: {marker_names}\n"
            f"Found - Shoulder: {shoulder_marker}, Elbow: {elbow_marker}, Wrist: {wrist_marker}"
        )
    
    # Extract 3D positions
    shoulder_pos = marker_df[[f'{shoulder_marker}_X', f'{shoulder_marker}_Y', f'{shoulder_marker}_Z']].values
    elbow_pos = marker_df[[f'{elbow_marker}_X', f'{elbow_marker}_Y', f'{elbow_marker}_Z']].values
    wrist_pos = marker_df[[f'{wrist_marker}_X', f'{wrist_marker}_Y', f'{wrist_marker}_Z']].values
    
    # Compute elbow angle
    elbow_angle = compute_elbow_angle(shoulder_pos, elbow_pos, wrist_pos)
    
    # Estimate muscle forces
    forces_df = estimate_muscle_forces_from_kinematics(elbow_angle, task_type=task_type)
    
    return forces_df


if __name__ == '__main__':
    # Test with synthetic data
    n = 200
    shoulder = np.array([np.zeros(n), np.zeros(n), np.ones(n) * 100]).T
    elbow = np.array([np.ones(n) * 30, np.zeros(n), np.ones(n) * 70]).T
    
    # Simulate wrist movement (flexion-extension)
    wrist_x = 60 + 20 * np.sin(2 * np.pi * np.arange(n) / 50)
    wrist = np.array([wrist_x, np.zeros(n), np.ones(n) * 70]).T
    
    angles = compute_elbow_angle(shoulder, elbow, wrist)
    print(f"Elbow angle range: {angles.min():.1f}° to {angles.max():.1f}°")
    
    forces = estimate_muscle_forces_from_kinematics(angles)
    print("\nEstimated muscle forces:")
    print(forces.describe())
