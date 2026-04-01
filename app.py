import os
import subprocess
import tempfile
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from extract_temporal_features import extract_temporal_features_from_event_cycle
from opensim_pain_pipeline import compute_event_cycle_from_sto
from run_static_batch import TEMPLATE_IK_NAME, pick_template_for_trial

try:
    from openai import OpenAI

    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

try:
    from google import genai

    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False


SMOKE_TEST_MAX_DURATION_SECONDS = 0.2


def run_static_optimization(
    trial_name: str,
    ik_path: str,
    output_dir: str,
    max_duration_seconds: float | None = None,
    log_callback=None,
) -> list[str]:
    """Run OpenSim Static Optimization for one IK result and return .sto outputs."""
    template_path = pick_template_for_trial(trial_name)
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    def get_mot_time_range(mot_file: str) -> tuple[float, float]:
        with open(mot_file, "r", encoding="utf-8") as f:
            data_lines = [line.strip() for line in f.readlines() if line.strip()]

        numeric_rows = []
        for line in data_lines:
            first_token = line.split()[0]
            try:
                float(first_token)
                numeric_rows.append(line)
            except ValueError:
                continue

        if not numeric_rows:
            raise RuntimeError(f"No numeric data rows found in IK file: {mot_file}")

        start_time = float(numeric_rows[0].split()[0])
        end_time = float(numeric_rows[-1].split()[0])
        return start_time, end_time

    def get_trial_side(name: str) -> str:
        lowered = name.lower()
        if "_left" in lowered:
            return "left"
        if "_right" in lowered:
            return "right"
        return "right"

    def build_tuned_model_copy(base_model_path: Path, trial_name: str) -> Path:
        trial_side = get_trial_side(trial_name)
        locked_side = "right" if trial_side == "left" else "left"
        tuned_model_path = base_model_path.with_name(f"{base_model_path.stem}_tuned_{trial_name}.osim")

        lock_coords = {
            "pelvis_tilt",
            "pelvis_list",
            "pelvis_rotation",
            "pelvis_tx",
            "pelvis_ty",
            "pelvis_tz",
            "lumbar_extension",
            "lumbar_bending",
            "lumbar_rotation",
            f"arm_flex_{locked_side[0]}",
            f"arm_add_{locked_side[0]}",
            f"arm_rot_{locked_side[0]}",
            f"elbow_flex_{locked_side[0]}",
            f"pro_sup_{locked_side[0]}",
            f"wrist_flex_{locked_side[0]}",
            f"wrist_dev_{locked_side[0]}",
        }

        tree = ET.parse(base_model_path)
        root = tree.getroot()

        for coordinate in root.findall(".//Coordinate"):
            coordinate_name = coordinate.get("name")
            if coordinate_name in lock_coords:
                locked_element = coordinate.find("locked")
                if locked_element is not None:
                    locked_element.text = "true"

        # Convergence-oriented tuning: give the optimizer a little more muscle reserve.
        for muscle in root.findall(".//Millard2012EquilibriumMuscle"):
            force_element = muscle.find("max_isometric_force")
            if force_element is None or force_element.text is None:
                continue
            try:
                force_value = float(force_element.text)
            except ValueError:
                continue
            force_element.text = f"{force_value * 1.5:.10f}"

        tree.write(tuned_model_path, encoding="utf-8", xml_declaration=True)
        return tuned_model_path

    base_dir = Path(__file__).resolve().parent
    abs_output_dir = Path(output_dir).resolve()
    base_model_path = base_dir / "osim models" / ("Elbow_model_load.osim" if "_nor_" in trial_name.lower() else "Elbow_model.osim")

    if log_callback is not None:
        log_callback(f"Tuning model for convergence: {base_model_path.name}")
    tuned_model_path = build_tuned_model_copy(base_model_path, trial_name)
    if log_callback is not None:
        log_callback(f"Tuned model copy created: {tuned_model_path}")

    start_time, end_time = get_mot_time_range(ik_path)
    if max_duration_seconds is not None:
        capped_end_time = min(end_time, start_time + max_duration_seconds)
        if capped_end_time < end_time and log_callback is not None:
            log_callback(
                f"Smoke test mode: limiting Static Optimization to {max_duration_seconds:.3f} s "
                f"from {start_time:.3f} to {capped_end_time:.3f}."
            )
        end_time = capped_end_time

    with open(template_path, "r", encoding="utf-8") as f:
        xml_text = f.read()

    xml_text = xml_text.replace("muscle_forces_result", abs_output_dir.as_posix())
    xml_text = xml_text.replace("osim models/Elbow_model_load.osim", tuned_model_path.as_posix())
    xml_text = xml_text.replace("osim models/Elbow_model.osim", tuned_model_path.as_posix())
    xml_text = xml_text.replace(TEMPLATE_IK_NAME, os.path.basename(ik_path))
    xml_text = xml_text.replace("<initial_time>0</initial_time>", f"<initial_time>{start_time}</initial_time>")
    xml_text = xml_text.replace("<final_time>30.484000000000002</final_time>", f"<final_time>{end_time}</final_time>")
    xml_text = xml_text.replace("<start_time>0</start_time>", f"<start_time>{start_time}</start_time>")
    xml_text = xml_text.replace("<end_time>30.484000000000002</end_time>", f"<end_time>{end_time}</end_time>")
    xml_text = xml_text.replace("<optimizer_convergence_criterion>0.0001</optimizer_convergence_criterion>", "<optimizer_convergence_criterion>0.001</optimizer_convergence_criterion>")
    xml_text = xml_text.replace("<optimizer_max_iterations>100</optimizer_max_iterations>", "<optimizer_max_iterations>250</optimizer_max_iterations>")

    trial_xml_path = abs_output_dir / f"setup_static_{trial_name}.xml"
    with open(trial_xml_path, "w", encoding="utf-8") as f:
        f.write(xml_text)

    command = ["opensim-cmd", "run-tool", str(trial_xml_path)]
    if log_callback is not None:
        log_callback(f"Running: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        clean_line = line.rstrip()
        if clean_line:
            output_lines.append(clean_line)
            if log_callback is not None:
                log_callback(clean_line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError("\n".join(output_lines) or "OpenSim Static Optimization failed")

    sto_paths = []
    for name in os.listdir(abs_output_dir):
        if name.lower().endswith(".sto"):
            sto_paths.append(str(abs_output_dir / name))
    sto_paths.sort()
    return sto_paths


def run_c3d_to_trc_subprocess(c3d_path: str, trc_path: str) -> None:
    script_path = Path(__file__).with_name("c3d_to_trc.py")
    result = subprocess.run(
        [sys.executable, str(script_path), c3d_path, trc_path],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "C3D conversion failed").strip())


def run_ik_subprocess(trc_path: str, results_dir: str) -> str:
    script_path = Path(__file__).with_name("run_opensim_batch.py")
    result = subprocess.run(
        [sys.executable, str(script_path), trc_path, results_dir],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "IK run failed").strip())

    ik_path = os.path.join(results_dir, f"{Path(trc_path).stem}_ik.mot")
    if not os.path.exists(ik_path):
        raise RuntimeError(f"IK completed but output not found: {ik_path}")
    return ik_path


def pick_primary_sto(sto_paths: list[str]) -> str:
    """Pick most likely force .sto output."""
    if not sto_paths:
        raise RuntimeError("No .sto files were generated by Static Optimization.")
    for p in sto_paths:
        name = os.path.basename(p).lower()
        if "force" in name or "staticoptimization" in name:
            return p
    return sto_paths[0]


def _register_numpy_core_compat() -> None:
    try:
        import numpy.core as numpy_core

        sys.modules.setdefault("numpy._core", numpy_core)
        for submodule in (
            "multiarray",
            "numeric",
            "umath",
            "overrides",
            "shape_base",
            "fromnumeric",
            "function_base",
            "arrayprint",
            "defchararray",
            "records",
            "einsumfunc",
            "machar",
            "getlimits",
            "_methods",
            "_multiarray_umath",
        ):
            try:
                module = __import__(f"numpy.core.{submodule}", fromlist=["*"])
                sys.modules.setdefault(f"numpy._core.{submodule}", module)
            except Exception:
                pass
    except Exception:
        pass


def load_model_bundle(path: str = "model_temporal.joblib"):
    p = Path(path)
    if not p.exists():
        return None
    try:
        _register_numpy_core_compat()
        return joblib.load(p)
    except Exception as exc:
        st.warning(
            f"Could not load {p.name}: {exc}. "
            "The app will still run, but ML prediction will be disabled until the model is retrained or resaved in this environment."
        )
        return None


def predict_from_temporal_features(model_obj, features_df):
    model = model_obj["model"]
    scaler = model_obj.get("scaler")
    feature_names = model_obj["features"]

    for col in feature_names:
        if col not in features_df.columns:
            features_df[col] = 0.0

    x = features_df[feature_names]
    x_scaled = scaler.transform(x) if scaler is not None else x

    proba = model.predict_proba(x_scaled)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = features_df.copy()
    out["prob_tendinopathy"] = proba
    out["pred_tendinopathy"] = pred
    return out


def plot_shap_like_explanation(model_obj, features_df, idx: int = 0):
    """Coefficient-based SHAP-like plot for logistic regression models."""
    model = model_obj["model"]
    scaler = model_obj.get("scaler")
    feature_names = model_obj["features"]

    x = features_df[feature_names].iloc[[idx]]
    x_scaled = scaler.transform(x) if scaler is not None else x.values
    vals = x_scaled[0]

    if not hasattr(model, "coef_"):
        return None
    shap_vals = model.coef_[0] * vals

    import matplotlib.pyplot as plt

    order = np.argsort(np.abs(shap_vals))[::-1][:10]
    top_vals = shap_vals[order]
    top_names = [feature_names[i] for i in order]
    colors = ["#FF6B6B" if v > 0 else "#4ECDC4" for v in top_vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_vals)), top_vals, color=colors)
    ax.set_yticks(range(len(top_vals)))
    ax.set_yticklabels(top_names)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Contribution")
    ax.set_title("Top 10 Feature Contributions")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close()
    return None


def get_top_feature_table(model_obj, features_df, idx: int = 0):
    model = model_obj["model"]
    scaler = model_obj.get("scaler")
    feature_names = model_obj["features"]
    x = features_df[feature_names].iloc[[idx]]
    x_scaled = scaler.transform(x) if scaler is not None else x.values
    vals = x_scaled[0]
    if not hasattr(model, "coef_"):
        return None
    shap_vals = model.coef_[0] * vals
    import pandas as pd

    return pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": shap_vals,
            "abs_shap": np.abs(shap_vals),
        }
    ).sort_values("abs_shap", ascending=False)


def generate_llm_explanation(prediction, prob, top_features, trial_info, groq_key, gemini_key):
    condition = "Tendinopathy" if prediction == 1 else "Normal"
    confidence = prob if prediction == 1 else (1 - prob)
    top_text = "\n".join(
        [
            f"- {row['feature']}: contribution={row['shap_value']:.4f}"
            for _, row in top_features.head(5).iterrows()
        ]
    )
    prompt = f"""You are a clinical biomechanics expert.
Prediction: {condition}
Confidence: {confidence*100:.1f}%
Trial: {trial_info}
Top features:
{top_text}

Provide:
1) Clinical interpretation
2) Why top features matter
3) Short patient-friendly explanation
4) Rehab suggestion
"""

    if _HAS_OPENAI and groq_key:
        try:
            client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a clinical biomechanics expert."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=1200,
            )
            return resp.choices[0].message.content, "Groq"
        except Exception:
            pass

    if _HAS_GEMINI and gemini_key:
        try:
            client = genai.Client(api_key=gemini_key)
            resp = client.models.generate_content(model="gemini-1-5-flash", contents=prompt)
            text = resp.text if hasattr(resp, "text") else str(resp)
            return text, "Gemini"
        except Exception:
            pass

    return None, None


def main() -> None:
    st.set_page_config(page_title="OpenSim Tendinopathy Pipeline", page_icon="🏥", layout="wide")
    st.title("🏥 OpenSim-Only C3D Pipeline")
    st.caption(
        "Flow: C3D -> TRC -> Inverse Kinematics (.mot) -> Static Optimization (.sto) "
        "-> Python port of your MATLAB pain model -> EventCycle CSV"
    )
    st.warning(
        "The simplified direct method is removed. Only OpenSim + converted pain model path runs."
    )
    st.sidebar.subheader("AI Configuration (optional)")
    run_mode = st.sidebar.selectbox(
        "Static Optimization mode",
        ["Smoke test (0.2 s)", "Full run"],
        index=0,
        help="Smoke test is faster and useful to confirm the flow. Full run uses the entire IK duration.",
    )
    groq_key = st.sidebar.text_input("Groq API Key", type="password")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    max_duration_seconds = None if run_mode == "Full run" else SMOKE_TEST_MAX_DURATION_SECONDS
    model_obj = load_model_bundle("model_temporal.joblib")
    if model_obj is None:
        st.error("`model_temporal.joblib` not found. Train/load model before running classification.")
    else:
        st.success("Temporal classifier model loaded.")

    c3d_file = st.file_uploader("Upload C3D file", type=["c3d"])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        subject = st.number_input("Subject ID", min_value=1, value=1, step=1)
    with col2:
        condition = st.selectbox("Condition", ["normal", "tendon"], index=0)
    with col3:
        task = st.selectbox("Task", ["WithTask", "Rest"], index=0)
    with col4:
        speed = st.selectbox("Speed", ["Fast", "Medium", "Slow"], index=1)

    if c3d_file is None:
        st.info("Upload a C3D file to start.")
        return

    trial = Path(c3d_file.name).stem
    trial_dir = Path("intermediate") / trial
    trial_dir.mkdir(parents=True, exist_ok=True)

    if st.button("Run OpenSim Pipeline", type="primary"):
        status = st.status("Starting pipeline...", expanded=True)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
                tmp.write(c3d_file.read())
                c3d_tmp_path = tmp.name

            trc_path = str(trial_dir / f"{trial}.trc")

            status.write("Converting C3D to TRC...")
            try:
                run_c3d_to_trc_subprocess(c3d_tmp_path, trc_path)
            except Exception as exc:
                raise RuntimeError(f"C3D to TRC conversion failed for {c3d_file.name}: {exc}")
            status.write(f"TRC generated: {trc_path}")

            status.write("Running Inverse Kinematics...")
            try:
                ik_path = run_ik_subprocess(trc_path, str(trial_dir))
            except Exception as exc:
                raise RuntimeError(f"IK failed for {c3d_file.name}: {exc}")
            status.write(f"IK MOT generated: {ik_path}")

            status.write("Running Static Optimization...")
            sto_paths = run_static_optimization(
                trial,
                ik_path,
                str(trial_dir),
                max_duration_seconds=max_duration_seconds,
                log_callback=status.write,
            )
            primary_sto = pick_primary_sto(sto_paths)
            status.write(f"Static Optimization completed: {primary_sto}")

            event_cycle_csv = str(trial_dir / "event_cycle.csv")
            summary_csv = str(trial_dir / "summary.csv")

            status.write("Calculating event-cycle pain data...")
            event_df, summary_df, _ = compute_event_cycle_from_sto(
                sto_path=primary_sto,
                subject=subject,
                condition=condition,
                task=task,
                speed=speed,
            )
            event_df.to_csv(event_cycle_csv, index=False)
            summary_df.to_csv(summary_csv, index=False)
            status.write(f"EventCycle CSV generated: {event_cycle_csv}")

            status.write("Extracting temporal features...")
            temporal_features_df = extract_temporal_features_from_event_cycle(event_df)
            temporal_features_csv = str(trial_dir / "temporal_features.csv")
            temporal_features_df.to_csv(temporal_features_csv, index=False)
            status.write(f"Temporal features generated: {temporal_features_csv}")

            prediction_df = None
            prediction_csv = str(trial_dir / "prediction.csv")
            if model_obj is not None:
                status.write("Running ML prediction...")
                prediction_df = predict_from_temporal_features(model_obj, temporal_features_df)
                prediction_df.to_csv(prediction_csv, index=False)
                status.write(f"Prediction CSV generated: {prediction_csv}")

            status.update(label="Pipeline completed successfully.", state="complete", expanded=False)
            st.success("Pipeline completed successfully.")
            st.write(f"TRC: `{trc_path}`")
            st.write(f"IK: `{ik_path}`")
            st.write(f"Primary STO: `{primary_sto}`")
            st.write(f"EventCycle CSV: `{event_cycle_csv}`")
            st.write(f"Summary CSV: `{summary_csv}`")
            st.write(f"Temporal Features CSV: `{temporal_features_csv}`")
            if prediction_df is not None:
                st.write(f"Prediction CSV: `{prediction_csv}`")

                st.subheader("Generated EventCycle CSV preview")
                st.dataframe(event_df.head(20), use_container_width=True)
                st.subheader("Generated Summary CSV preview")
                st.dataframe(summary_df, use_container_width=True)
                st.subheader("Generated Temporal Features preview")
                st.dataframe(temporal_features_df, use_container_width=True)

                if prediction_df is not None:
                    st.subheader("Classification Result")
                    row = prediction_df.iloc[0]
                    pred_label = "TENDINOPATHY" if int(row["pred_tendinopathy"]) == 1 else "NORMAL"
                    prob = float(row["prob_tendinopathy"])
                    confidence = prob if int(row["pred_tendinopathy"]) == 1 else (1.0 - prob)
                    c1, c2 = st.columns(2)
                    with c1:
                        if pred_label == "TENDINOPATHY":
                            st.error(f"Prediction: {pred_label}")
                        else:
                            st.success(f"Prediction: {pred_label}")
                    with c2:
                        st.metric("Confidence", f"{confidence:.1%}")
                    st.dataframe(
                        prediction_df[
                            [
                                "Subject",
                                "Condition",
                                "Task",
                                "Speed",
                                "prob_tendinopathy",
                                "pred_tendinopathy",
                            ]
                        ],
                        use_container_width=True,
                    )

                    st.subheader("SHAP-like Feature Explanation")
                    top_features = get_top_feature_table(model_obj, temporal_features_df, idx=0)
                    if top_features is not None:
                        plot_shap_like_explanation(model_obj, temporal_features_df, idx=0)
                        st.dataframe(top_features.head(10), use_container_width=True)

                        st.subheader("LLM Clinical Report")
                        trial_info = f"Subject {subject}, Task: {task}, Speed: {speed}"
                        with st.spinner("Generating report..."):
                            explanation, api_used = generate_llm_explanation(
                                int(row["pred_tendinopathy"]),
                                float(row["prob_tendinopathy"]),
                                top_features,
                                trial_info,
                                groq_key.strip(),
                                gemini_key.strip(),
                            )
                        if explanation:
                            st.success(f"Report generated using {api_used}")
                            st.markdown(explanation)
                        else:
                            st.info("Provide Groq/Gemini key in sidebar to generate LLM report.")

                st.subheader("Downloads")
                with open(trc_path, "rb") as f:
                    st.download_button(
                        "Download TRC",
                        data=f.read(),
                        file_name=os.path.basename(trc_path),
                        mime="text/plain",
                    )
                with open(ik_path, "rb") as f:
                    st.download_button(
                        "Download IK MOT",
                        data=f.read(),
                        file_name=os.path.basename(ik_path),
                        mime="text/plain",
                    )
                for sto_path in sto_paths:
                    with open(sto_path, "rb") as f:
                        st.download_button(
                            f"Download STO ({os.path.basename(sto_path)})",
                            data=f.read(),
                            file_name=os.path.basename(sto_path),
                            mime="text/plain",
                            key=f"sto_{os.path.basename(sto_path)}",
                        )
                st.download_button(
                    "Download EventCycle CSV",
                    data=event_df.to_csv(index=False),
                    file_name=f"{trial}_event_cycle.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Download Summary CSV",
                    data=summary_df.to_csv(index=False),
                    file_name=f"{trial}_summary.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Download Temporal Features CSV",
                    data=temporal_features_df.to_csv(index=False),
                    file_name=f"{trial}_temporal_features.csv",
                    mime="text/csv",
                )
                if prediction_df is not None:
                    st.download_button(
                        "Download Prediction CSV",
                        data=prediction_df.to_csv(index=False),
                        file_name=f"{trial}_prediction.csv",
                        mime="text/csv",
                    )

        except subprocess.CalledProcessError as e:
                st.error(f"External command failed: {e}")
        except Exception as e:
                st.error(f"Pipeline failed: {e}")
        finally:
            if "c3d_tmp_path" in locals() and os.path.exists(c3d_tmp_path):
                os.unlink(c3d_tmp_path)


if __name__ == "__main__":
    main()
