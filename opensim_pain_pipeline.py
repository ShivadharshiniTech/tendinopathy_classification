import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _read_opensim_sto(sto_path: str) -> pd.DataFrame:
    """Read OpenSim .sto into a DataFrame."""
    path = Path(sto_path)
    if not path.exists():
        raise FileNotFoundError(f"STO file not found: {sto_path}")

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            start_idx = i + 1
            break

    df = pd.read_csv(
        sto_path,
        sep=r"\s+",
        engine="python",
        skiprows=start_idx,
    )
    return df


def _pick_force_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    def find_one(token: str) -> str | None:
        for c in cols:
            if token in c.lower():
                return c
        return None

    ecu_col = find_one("ecu")
    ecrl_col = find_one("ecrl")
    ecrb_col = find_one("ecrb")

    if ecu_col and ecrl_col and ecrb_col:
        return ecu_col, ecrl_col, ecrb_col

    # Fallback: first three numeric columns after time.
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    time_like = [c for c in numeric_cols if c.lower() == "time"]
    for t in time_like:
        numeric_cols.remove(t)
    if len(numeric_cols) >= 3:
        return numeric_cols[0], numeric_cols[1], numeric_cols[2]

    raise ValueError(
        "Could not find ECU/ECRL/ECRB columns in STO. "
        f"Available columns: {cols}"
    )


def _infer_task_speed_from_name(name: str) -> tuple[str, str]:
    s = name.lower()
    task = "Rest" if "rest" in s else "WithTask"
    if "fast" in s:
        speed = "Fast"
    elif "med" in s:
        speed = "Medium"
    elif "slow" in s:
        speed = "Slow"
    else:
        speed = "Unknown"
    return task, speed


def compute_event_cycle_from_sto(
    sto_path: str,
    subject: int,
    condition: str,
    task: str | None = None,
    speed: str | None = None,
    damage_memory: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Python port of provided MATLAB pain model for one STO file."""
    # Settings
    w_ecu = 0.2
    w_ecrl = 0.3
    w_ecrb = 0.5
    thresh_frac = 0.15
    n_cycle_pts = 101

    # Pain model params
    p_hat = np.array([0.85, 1.5, 0.8, 4.0, 0.15, 0.12, 60.0], dtype=float)
    theta, beta, gamma, lambda_param, k_rate, n50, a = p_hat

    if damage_memory is None:
        damage_memory = {}

    df = _read_opensim_sto(sto_path)
    ecu_col, ecrl_col, ecrb_col = _pick_force_columns(df)

    trial_name = Path(sto_path).stem
    if task is None or speed is None:
        t_infer, s_infer = _infer_task_speed_from_name(trial_name)
        if task is None:
            task = t_infer
        if speed is None:
            speed = s_infer

    ecu = pd.to_numeric(df[ecu_col], errors="coerce").fillna(0.0).to_numpy()
    ecrl = pd.to_numeric(df[ecrl_col], errors="coerce").fillna(0.0).to_numpy()
    ecrb = pd.to_numeric(df[ecrb_col], errors="coerce").fillna(0.0).to_numpy()

    f = w_ecu * ecu + w_ecrl * ecrl + w_ecrb * ecrb
    max_f = float(np.max(np.abs(f))) if len(f) else 0.0
    if max_f <= 0:
        raise ValueError("Combined force signal is zero; cannot normalize.")
    f = f / max_f

    idx = np.where(f > thresh_frac)[0]
    if len(idx) == 0:
        raise ValueError("No force samples above threshold; cannot build event cycle.")

    fe = f[idx[0] : idx[-1] + 1]

    event_cycle = np.linspace(0.0, 100.0, n_cycle_pts)
    x_old = np.linspace(0.0, 100.0, len(fe))
    fe_cycle = np.interp(event_cycle, x_old, fe)
    fe_cycle = np.maximum(fe_cycle, 0.0)
    fe_cycle = (
        pd.Series(fe_cycle).rolling(window=5, center=True, min_periods=1).median().to_numpy()
    )

    df_force = np.diff(fe_cycle, prepend=0.0)

    if str(condition).lower() == "tendon":
        theta_eff = 0.75
        lambda_eff = 4.5
        n50_eff = 0.15
    else:
        theta_eff = theta
        lambda_eff = lambda_param
        n50_eff = n50

    s = np.maximum(fe_cycle - theta_eff, 0.0) + k_rate * np.abs(df_force)

    x = np.zeros_like(s)
    for i in range(1, len(s)):
        x[i] = max(x[i - 1] + (beta * s[i - 1] - gamma * x[i - 1]), 0.0)

    n = (1.0 + lambda_eff * x) * s
    pain = 10.0 / (1.0 + np.exp(-a * (n - n50_eff)))
    pain = np.clip(pain, 0.0, 10.0)

    key = f"S{int(subject)}_{str(condition)}"
    if key not in damage_memory:
        damage_memory[key] = 0.0

    overload = float(np.mean(np.maximum(fe_cycle - theta_eff, 0.0)))
    rep_load = float(np.trapz(fe_cycle))
    damage_memory[key] = damage_memory[key] + 0.4 * overload + 0.02 * rep_load
    damage = float(damage_memory[key])

    vas_model = 10.0 / (1.0 + np.exp(-4.0 * (damage - 1.0)))
    vas_model = float(np.clip(vas_model, 0.0, 10.0))

    event_df = pd.DataFrame(
        {
            "Subject": int(subject),
            "Condition": str(condition),
            "Task": str(task),
            "Speed": str(speed),
            "EventCycle": event_cycle,
            "Pain_pred": pain,
        }
    )

    summary_df = pd.DataFrame(
        {
            "Subject": [int(subject)],
            "Condition": [str(condition)],
            "Peak_Pain": [float(np.max(pain))],
            "Mean_Pain": [float(np.mean(pain))],
            "VAS_Model": [vas_model],
        }
    )

    return event_df, summary_df, damage_memory


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OpenSim STO to pain event-cycle CSV")
    parser.add_argument("--sto", required=True, help="Input STO file path")
    parser.add_argument("--event-out", required=True, help="Output event-cycle CSV path")
    parser.add_argument("--summary-out", default="", help="Optional output summary CSV path")
    parser.add_argument("--subject", type=int, required=True, help="Subject ID")
    parser.add_argument("--condition", required=True, help="Condition: normal/tendon")
    parser.add_argument("--task", default="", help="Task label (optional)")
    parser.add_argument("--speed", default="", help="Speed label (optional)")
    args = parser.parse_args()

    task = args.task if args.task else None
    speed = args.speed if args.speed else None
    event_df, summary_df, _ = compute_event_cycle_from_sto(
        sto_path=args.sto,
        subject=args.subject,
        condition=args.condition,
        task=task,
        speed=speed,
        damage_memory=None,
    )

    Path(args.event_out).parent.mkdir(parents=True, exist_ok=True)
    event_df.to_csv(args.event_out, index=False)
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.summary_out, index=False)

    print(f"Saved event-cycle CSV: {args.event_out}")
    if args.summary_out:
        print(f"Saved summary CSV: {args.summary_out}")


if __name__ == "__main__":
    main()
