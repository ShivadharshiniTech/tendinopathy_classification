import os

TRC_DIR = 'trc_outputs'

# This script will fix the headers of all TRC files in the trc_outputs directory to be OpenSim-compatible.
def fix_trc_header(trc_path):
    with open(trc_path, 'r') as f:
        lines = f.readlines()
    # The 4th line is the marker names, the 5th line is X/Y/Z, then data
    if len(lines) < 6:
        print(f"Skipping {trc_path}: too few lines.")
        return
    # Parse marker names from the 4th line
    marker_line = lines[3].strip().split('\t')[2:]  # skip Frame# and Time
    marker_names = [m for m in marker_line if m]
    # Build new marker headers
    marker_headers = []
    for m in marker_names:
        marker_headers.extend([f'{m}X', f'{m}Y', f'{m}Z'])
    # Rebuild header
    new_lines = lines[:3]
    new_lines.append('Frame#\tTime\t' + '\t'.join(marker_headers) + '\n')
    # Add data (skip old marker lines)
    new_lines.extend(lines[5:])
    with open(trc_path, 'w') as f:
        f.writelines(new_lines)
    print(f"Fixed header: {os.path.basename(trc_path)}")

if __name__ == "__main__":
    for fname in os.listdir(TRC_DIR):
        if fname.lower().endswith('.trc'):
            fix_trc_header(os.path.join(TRC_DIR, fname))
    print('All TRC headers fixed.')
