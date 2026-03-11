"""
C3D File Reader
Extracts marker trajectories and metadata from motion capture C3D files
"""
import numpy as np
import pandas as pd
try:
    import ezc3d
    _HAS_EZC3D = True
except ImportError:
    _HAS_EZC3D = False


def read_c3d_file(filepath):
    """
    Read C3D file and extract marker data
    
    Args:
        filepath: Path to C3D file
    
    Returns:
        dict with:
            - markers: DataFrame with marker positions (time x markers x XYZ)
            - marker_names: List of marker names
            - frame_rate: Sampling frequency
            - time: Time vector
    """
    if not _HAS_EZC3D:
        raise ImportError("ezc3d not installed. Run: pip install ezc3d")
    
    # Read C3D file
    c3d = ezc3d.c3d(filepath)
    
    # Extract marker data
    # Shape: (4, n_markers, n_frames) - [X, Y, Z, confidence]
    points = c3d['data']['points']
    
    # Get metadata
    marker_labels = c3d['parameters']['POINT']['LABELS']['value']
    frame_rate = c3d['parameters']['POINT']['RATE']['value'][0]
    n_frames = points.shape[2]
    
    # Clean marker names (replace spaces with underscores)
    marker_names = [name.strip().replace(' ', '_') for name in marker_labels]
    
    # Create time vector
    time = np.arange(n_frames) / frame_rate
    
    # Organize data into DataFrame
    marker_data = {}
    for i, marker in enumerate(marker_names):
        marker_data[f'{marker}_X'] = points[0, i, :]
        marker_data[f'{marker}_Y'] = points[1, i, :]
        marker_data[f'{marker}_Z'] = points[2, i, :]
    
    df = pd.DataFrame(marker_data)
    df.insert(0, 'Time', time)
    
    return {
        'markers': df,
        'marker_names': marker_names,
        'frame_rate': frame_rate,
        'time': time,
        'n_frames': n_frames
    }


def extract_upper_limb_markers(marker_df, marker_names):
    """
    Extract relevant upper limb markers for elbow analysis
    
    Common marker set (adjust based on your specific marker protocol):
    - Shoulder: RSHO (right shoulder) or similar
    - Elbow: RELB (right elbow lateral epicondyle)
    - Wrist: RWRA, RWRB (wrist markers)
    - Hand: Hand markers
    
    Returns:
        Dictionary with key anatomical positions
    """
    # This is a simplified extraction - adjust based on your actual marker set
    required_markers = {}
    
    # Try to find shoulder marker
    shoulder_candidates = ['RSHO', 'R_SHOULDER', 'SHOULDER_R', 'SHO_R']
    for candidate in shoulder_candidates:
        if candidate in marker_names:
            required_markers['shoulder'] = candidate
            break
    
    # Try to find elbow marker
    elbow_candidates = ['RELB', 'R_ELBOW', 'ELBOW_R', 'ELB_R', 'RLE', 'LELB']
    for candidate in elbow_candidates:
        if candidate in marker_names:
            required_markers['elbow'] = candidate
            break
    
    # Try to find wrist markers
    wrist_candidates = ['RWRA', 'RWRB', 'R_WRIST', 'WRIST_R', 'WRA_R', 'WRB_R']
    for candidate in wrist_candidates:
        if candidate in marker_names:
            if 'wrist1' not in required_markers:
                required_markers['wrist1'] = candidate
            else:
                required_markers['wrist2'] = candidate
    
    return required_markers


def save_as_trc(marker_data, output_path):
    """
    Save marker data in TRC format (OpenSim compatible)
    
    Args:
        marker_data: Dict from read_c3d_file()
        output_path: Output .trc file path
    """
    df = marker_data['markers']
    marker_names = marker_data['marker_names']
    frame_rate = marker_data['frame_rate']
    
    with open(output_path, 'w') as f:
        # Header
        f.write('PathFileType\t4\t(X/Y/Z)\t' + output_path + '\n')
        f.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
        f.write(f'{frame_rate}\t{frame_rate}\t{len(df)}\t{len(marker_names)}\tmm\t{frame_rate}\t1\t{len(df)}\n')
        
        # Marker names header
        f.write('Frame#\tTime\t')
        for marker in marker_names:
            f.write(f'{marker}\t\t\t')
        f.write('\n')
        
        # XYZ labels
        f.write('\t\t')
        for _ in marker_names:
            f.write('X\tY\tZ\t')
        f.write('\n')
        
        # Data
        for idx, row in df.iterrows():
            f.write(f'{idx+1}\t{row["Time"]:.6f}\t')
            for marker in marker_names:
                x = row.get(f'{marker}_X', 0)
                y = row.get(f'{marker}_Y', 0)
                z = row.get(f'{marker}_Z', 0)
                f.write(f'{x:.6f}\t{y:.6f}\t{z:.6f}\t')
            f.write('\n')
    
    print(f'Saved TRC file: {output_path}')


if __name__ == '__main__':
    # Example usage
    import sys
    if len(sys.argv) > 1:
        c3d_path = sys.argv[1]
        data = read_c3d_file(c3d_path)
        print(f"Loaded {len(data['marker_names'])} markers")
        print(f"Markers: {data['marker_names']}")
        print(f"Frame rate: {data['frame_rate']} Hz")
        print(f"Duration: {data['time'][-1]:.2f} seconds")
