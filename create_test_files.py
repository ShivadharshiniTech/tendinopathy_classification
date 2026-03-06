"""
Create individual test files from EventCycle data for testing the Streamlit app
"""
import pandas as pd
from pathlib import Path

# Load EventCycle data
df = pd.read_csv('dataset/PainModel_EventCycle_AllSubjects_1 (1).csv')

print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# Create test data folder
test_dir = Path('test data')
test_dir.mkdir(exist_ok=True)

# Group by trial (Subject × Condition × Task × Speed)
trial_count = 0

for (subj, cond, task, speed), group in df.groupby(['Subject', 'Condition', 'Task', 'Speed']):
    trial_count += 1
    
    # Create filename
    filename = f"trial_{trial_count:03d}_S{subj}_{cond}_{task}_{speed}.csv"
    filepath = test_dir / filename
    
    # Save as CSV
    group.to_csv(filepath, index=False)
    
    # Also create a few Excel samples (first 5 trials)
    if trial_count <= 5:
        excel_path = filepath.with_suffix('.xlsx')
        group.to_excel(excel_path, index=False)
    
    print(f"Created: {filename} ({len(group)} cycles)")

print(f"\n✓ Created {trial_count} test files in 'test data/' folder")
print(f"✓ Also created 5 Excel samples (.xlsx)")
print(f"\nYou can now upload any of these files to the Streamlit app for testing!")
