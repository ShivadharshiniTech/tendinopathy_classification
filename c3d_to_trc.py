import os
import sys
import ezc3d
import numpy as np
import pandas as pd

# Directory containing C3D files
C3D_DIR = 'c3d inputs'
TRC_DIR = 'trc_outputs'
os.makedirs(TRC_DIR, exist_ok=True)

def c3d_to_trc(c3d_path, trc_path):
    c3d = ezc3d.c3d(c3d_path)
    points = c3d['data']['points']  # shape: (4, n_markers, n_frames)
    labels = [label.replace(' ', '_') for label in c3d['parameters']['POINT']['LABELS']['value']]
    n_frames = points.shape[2]
    n_markers = points.shape[1]
    rate = c3d['parameters']['POINT']['RATE']['value'][0]
    first_frame = 1  # TRC is 1-based
    time = np.arange(n_frames) / rate

    # Prepare TRC header in OpenSim standard format
    # Line 1-3: file info and metadata
    header = [
        f'PathFileType\t4\t(X/Y/Z)\t{os.path.basename(trc_path)}',
        'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames',
        f'{rate:.2f}\t{rate:.2f}\t{n_frames}\t{n_markers}\tmm\t{rate:.2f}\t{first_frame}\t{n_frames}',
        # Line 4: Frame#, Time, then one entry per marker name (no X/Y/Z here)
        'Frame#\tTime\t' + '\t'.join(labels),
        # Line 5: coordinate labels X1 Y1 Z1 X2 Y2 Z2 ...
        '\t' + '\t'.join(
            coord
            for i in range(1, n_markers + 1)
            for coord in (f'X{i}', f'Y{i}', f'Z{i}')
        ),
    ]

    # Prepare TRC data
    data = []
    for i in range(n_frames):
        row = [str(i+1), f'{time[i]:.5f}']
        for m in range(n_markers):
            x, y, z = points[0, m, i], points[1, m, i], points[2, m, i]
            row += [f'{x:.5f}', f'{y:.5f}', f'{z:.5f}']
        data.append('\t'.join(row))

    # Write TRC file
    with open(trc_path, 'w') as f:
        for line in header:
            f.write(line + '\n')
        for row in data:
            f.write(row + '\n')


def main():
    if len(sys.argv) == 3:
        c3d_to_trc(sys.argv[1], sys.argv[2])
        return

if __name__ == "__main__":
    if len(sys.argv) == 3:
        main()
    else:
        # Batch convert all C3D files in the input directory
        for fname in os.listdir(C3D_DIR):
            if fname.lower().endswith('.c3d'):
                c3d_path = os.path.join(C3D_DIR, fname)
                trc_path = os.path.join(TRC_DIR, fname.replace('.c3d', '.trc'))
                print(f'Converting {fname} to {os.path.basename(trc_path)}...')
                c3d_to_trc(c3d_path, trc_path)
        print('All C3D files converted to TRC.')
