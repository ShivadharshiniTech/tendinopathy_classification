import os
import subprocess

# Folder containing IK results and where per-trial XMLs will be written
RESULTS_DIR = "muscle_forces_result"

# Root-level templates for unloaded vs loaded models
TEMPLATE_XML_UNLOADED = "setup_static_template_unloaded.xml"
TEMPLATE_XML_LOADED = "setup_static_template_loaded.xml"

# This must match the coordinates_file IK filename present in both templates
TEMPLATE_IK_NAME = "1H_nor_fast_left_ik.mot"


def pick_template_for_trial(trial_name: str) -> str:
    """Return appropriate SO template path for this trial.

    Convention: filenames containing "_nor_" are loaded trials
    and use the loaded-model template; all others use the
    unloaded-model template.
    """
    name_lower = trial_name.lower()
    if "_nor_" in name_lower:
        return TEMPLATE_XML_LOADED
    else:
        return TEMPLATE_XML_UNLOADED


def run_static_for_all_trials():
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith("_ik.mot"):
            continue

        trial_name = fname.replace("_ik.mot", "")

        template_path = pick_template_for_trial(trial_name)
        if not os.path.isfile(template_path):
            print(f"Skipping {trial_name}: template not found -> {template_path}")
            continue

        # Read the appropriate template XML
        with open(template_path, "r") as f:
            template_text = f.read()

        # Customize XML for this trial: update only the coordinates_file value
        xml_text = template_text.replace(TEMPLATE_IK_NAME, fname)

        trial_xml_path = os.path.join(RESULTS_DIR, f"setup_static_{trial_name}.xml")
        with open(trial_xml_path, "w") as f:
            f.write(xml_text)

        print(f"Running Static Optimization for {trial_name} using template {os.path.basename(template_path)}...")
        # Run OpenSim tool (opensim-cmd must be on PATH)
        subprocess.run(["opensim-cmd", "run-tool", trial_xml_path], check=True)
        print(f"Static Optimization finished for {trial_name}.")


if __name__ == "__main__":
    run_static_for_all_trials()
    print("All Static Optimization runs completed.")
