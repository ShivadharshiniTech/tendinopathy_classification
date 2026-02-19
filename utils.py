import pandas as pd
from pathlib import Path


def load_summary_csv(path):
    """Load a messy summary CSV and return a cleaned DataFrame.

    The file has an unusual format:
    - First row: generic headers (Var1, Var2, Var3, Var4, VAS_Model, ...)
    - Data rows: Subject, Condition, Peak_Pain, Mean_Pain, VAS_Model are in columns 0-4
    - Columns 5+ contain repeated metadata/column names that we ignore
    
    Strategy: Read with first row as header, keep only first 5 columns, rename them properly.
    """
    p = Path(path)
    
    # Read the file normally
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    
    # Check if this is the messy format with data in first few columns
    # Look for rows where later columns contain "Subject", "Condition" etc as values
    if df.shape[1] > 5:
        # Sample first data row to check if it has metadata in later columns
        first_row = df.iloc[0] if len(df) > 0 else pd.Series()
        later_cols = first_row.iloc[5:].values if len(first_row) > 5 else []
        has_metadata = any('Subject' in str(v) or 'Condition' in str(v) for v in later_cols)
        
        if has_metadata:
            # This is the messy format - keep only first 5 columns and rename them
            df = df.iloc[:, :5].copy()
            df.columns = ['Subject', 'Condition', 'Peak_Pain', 'Mean_Pain', 'VAS_Model']
            return df
    
    # Otherwise, try to find the header row dynamically (old fallback logic)
    # Read without header and search for row with column names
    raw = pd.read_csv(p, header=None, dtype=str, keep_default_na=False)
    
    for i, row in raw.iterrows():
        vals = [str(x).strip() for x in row.values]
        if 'Subject' in vals and 'Condition' in vals:
            # Found header row - use it
            df = pd.read_csv(p, header=i, dtype=str, keep_default_na=False)
            df = df.replace('', pd.NA).dropna(axis=1, how='all')
            return df
    
    # Final fallback: return as-is
    return df


def load_eventcycle_csv(path):
    """Load event-cycle CSV and return DataFrame.

    Tries to find a pain prediction column (commonly 'Pain_pred').
    """
    p = Path(path)
    df = pd.read_csv(p)

    # common pain column
    candidates = [c for c in df.columns if 'pain' in c.lower() or 'Pain' in c]
    if len(candidates) == 0:
        # fall back to 6th column if present
        if df.shape[1] >= 6:
            pain_col = df.columns[5]
        else:
            raise ValueError('Could not find pain column in event-cycle CSV')
    else:
        pain_col = candidates[0]

    df = df.rename(columns={pain_col: 'Pain_pred'})

    return df


def aggregate_eventcycle_to_summary(df):
    """Aggregate event-cycle DataFrame into per-subject summary features.

    Returns a DataFrame with columns: Subject, Condition, Peak_Pain, Mean_Pain
    """
    # ensure Subject and Condition exist
    if 'Subject' not in df.columns or 'Condition' not in df.columns:
        raise ValueError('Event-cycle data missing Subject or Condition columns')

    agg = (
        df.groupby(['Subject', 'Condition'])['Pain_pred']
        .agg(Peak_Pain='max', Mean_Pain='mean', Std_Pain='std')
        .reset_index()
    )
    # fill NaN std with 0
    agg['Std_Pain'] = agg['Std_Pain'].fillna(0)
    return agg
