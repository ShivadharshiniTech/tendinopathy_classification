import os
import sys
import opensim

TRC_DIR = 'trc_outputs'
RESULTS_DIR = 'muscle_forces_result'
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_trc_time_range(trc_path):
    """Return (start_time, end_time) from a TRC file.

    Assumes standard TRC layout with 5 header lines and
    time as the second column in each data row.
    """
    with open(trc_path, 'r') as f:
        lines = f.readlines()
    # Data starts after the 5 header lines
    data_lines = [ln.strip() for ln in lines[5:] if ln.strip()]
    if not data_lines:
        raise RuntimeError(f"No data rows found in TRC file: {trc_path}")
    first_fields = data_lines[0].split()  # Frame, Time, ...
    last_fields = data_lines[-1].split()
    start_time = float(first_fields[1])
    end_time = float(last_fields[1])
    return start_time, end_time

# Choose model based on trial name: loaded vs unloaded
def pick_model_for_trial(trial_name: str) -> str:
    """Return appropriate model path for this trial.

    Convention: filenames containing "_nor_" are loaded trials
    and use Elbow_model_load.osim; all others use Elbow_model.osim.
    """
    name_lower = trial_name.lower()
    if "_nor_" in name_lower:
        return os.path.join('osim models', 'Elbow_model_load.osim')
    else:
        return os.path.join('osim models', 'Elbow_model.osim')


# Helper: Run IK for each TRC file; SO is handled separately
def process_trc(trc_path, results_dir):
    basename = os.path.splitext(os.path.basename(trc_path))[0]
    ik_output = os.path.join(results_dir, f'{basename}_ik.mot')

    # Select and load model based on trial name
    model_path = pick_model_for_trial(basename)
    model = opensim.Model(model_path)
    model.initSystem()

    # Determine time range from TRC
    start_time, end_time = get_trc_time_range(trc_path)

    # Inverse Kinematics
    ik_tool = opensim.InverseKinematicsTool()
    ik_tool.setModel(model)
    ik_tool.setMarkerDataFileName(trc_path)
    ik_tool.setStartTime(start_time)
    ik_tool.setEndTime(end_time)
    ik_tool.setOutputMotionFileName(ik_output)
    ik_tool.run()
    print(f'IK done: {ik_output}')
    return ik_output


def main():
    if len(sys.argv) == 3:
        trc_path = sys.argv[1]
        results_dir = sys.argv[2]
        os.makedirs(results_dir, exist_ok=True)
        process_trc(trc_path, results_dir)
        return

if __name__ == "__main__":
    if len(sys.argv) == 3:
        main()
    else:
        # Batch process all TRC files
        for fname in os.listdir(TRC_DIR):
            if fname.lower().endswith('.trc'):
                trc_path = os.path.join(TRC_DIR, fname)
                print(f'Processing {fname} with appropriate elbow model...')
                process_trc(trc_path, RESULTS_DIR)

        print('All TRC files processed. IK results saved in muscle_forces_result. Static Optimization is run separately.')
