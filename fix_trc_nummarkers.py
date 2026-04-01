import os

TRC_DIR = 'trc_outputs'

def fix_num_markers(trc_path):
    with open(trc_path, 'r') as f:
        lines = f.readlines()
    if len(lines) < 4:
        print(f"Skipping {trc_path}: too few lines.")
        return
    # Parse marker headers from line 4 (index 3)
    marker_headers = lines[3].strip().split('\t')[2:]  # skip Frame# and Time
    n_marker_columns = len(marker_headers)
    if n_marker_columns % 3 != 0:
        print(f"Warning: {os.path.basename(trc_path)} has {n_marker_columns} marker columns, not divisible by 3.")
        return
    n_markers = n_marker_columns // 3
    # Fix NumMarkers in line 3 (index 2)
    header_fields = lines[2].strip().split('\t')
    if len(header_fields) < 4:
        print(f"Warning: {os.path.basename(trc_path)} header line 3 is malformed.")
        return
    old_num_markers = header_fields[3]
    header_fields[3] = str(n_markers)
    lines[2] = '\t'.join(header_fields) + '\n'
    with open(trc_path, 'w') as f:
        f.writelines(lines)
    print(f"Fixed NumMarkers: {os.path.basename(trc_path)} (was {old_num_markers}, now {n_markers})")

if __name__ == "__main__":
    for fname in os.listdir(TRC_DIR):
        if fname.lower().endswith('.trc'):
            fix_num_markers(os.path.join(TRC_DIR, fname))
    print('All TRC NumMarkers fields checked and fixed.')
