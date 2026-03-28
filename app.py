import streamlit as st
import joblib
import time
from pathlib import Path
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    px = None
    go = None
    _HAS_PLOTLY = False

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

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

from utils import load_temporal_features
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
)


def read_uploaded_file(uploaded_file):
    """Read CSV or Excel file from uploaded file object"""
    filename = uploaded_file.name.lower()
    
    if filename.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif filename.endswith(('.xlsx', '.xls')):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError(f'Unsupported file format. Please upload CSV or Excel (.xlsx, .xls) files.')


def extract_temporal_features_from_eventcycle(event_df):
    """Extract temporal features from raw event-cycle data
    
    Args:
        event_df: DataFrame with columns: Subject, Condition, Task, Speed, EventCycle, Pain_pred
        
    Returns:
        DataFrame with extracted temporal features for each trial
    """
    trials = []
    
    for (subj, cond, task, speed), group in event_df.groupby(
        ['Subject', 'Condition', 'Task', 'Speed']
    ):
        pain = group['Pain_pred'].values
        
        # Basic statistics
        features = {
            'Subject': subj,
            'Condition': cond,
            'Task': task,
            'Speed': speed,
            
            # Central tendency
            'peak_pain': np.max(pain),
            'mean_pain': np.mean(pain),
            'median_pain': np.median(pain),
            'min_pain': np.min(pain),
            
            # Dispersion
            'std_pain': np.std(pain),
            'pain_range': np.max(pain) - np.min(pain),
            'pain_cv': np.std(pain) / np.mean(pain) if np.mean(pain) > 0 else 0,
            'iqr_pain': np.percentile(pain, 75) - np.percentile(pain, 25),
            
            # Percentiles
            'percentile_95': np.percentile(pain, 95),
            'percentile_75': np.percentile(pain, 75),
            'percentile_25': np.percentile(pain, 25),
            'percentile_5': np.percentile(pain, 5),
            
            # Temporal dynamics
            'time_to_peak': np.argmax(pain) / len(pain) if len(pain) > 0 else 0,
            'pain_slope': np.polyfit(range(len(pain)), pain, 1)[0] if len(pain) > 1 else 0,
            'pain_curvature': np.polyfit(range(len(pain)), pain, 2)[0] if len(pain) > 2 else 0,
            
            # Phase-based
            'early_pain': np.mean(pain[:len(pain)//3]) if len(pain) >= 3 else np.mean(pain),
            'mid_pain': np.mean(pain[len(pain)//3:2*len(pain)//3]) if len(pain) >= 3 else np.mean(pain),
            'late_pain': np.mean(pain[2*len(pain)//3:]) if len(pain) >= 3 else np.mean(pain),
            
            # Derived metrics
            'pain_acceleration': np.mean(np.diff(np.diff(pain))) if len(pain) > 2 else 0,
            'pain_skewness': pd.Series(pain).skew(),
            'pain_kurtosis': pd.Series(pain).kurtosis(),
            
            # Onset/offset
            'onset_rate': (pain[10] - pain[0]) / 10 if len(pain) > 10 else 0,
            'offset_rate': (pain[-1] - pain[-10]) / 10 if len(pain) > 10 else 0,
        }
        
        trials.append(features)
    
    features_df = pd.DataFrame(trials)
    
    # Add target label if Condition is present
    if 'Condition' in features_df.columns:
        features_df['true_label'] = (features_df['Condition'].str.lower().str.contains('tend')).astype(int)
    
    return features_df


@st.cache_resource
def load_saved_model(path='model_temporal.joblib'):
    p = Path(path)
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_resource
def load_saved_scaler(path='scaler_temporal.joblib'):
    p = Path(path)
    if not p.exists():
        return None
    return joblib.load(p)


def predict_df(model_obj, scaler_obj, df, prob_col_name='prob_tendinopathy'):
    model = model_obj['model']
    features = model_obj['features']
    
    # Try to get scaler from model_obj first (new format), fall back to scaler_obj parameter
    if 'scaler' in model_obj:
        scaler = model_obj['scaler']
    else:
        scaler = scaler_obj
    
    X = df[features]
    
    # Apply scaler if available (temporal model requires scaling)
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    proba = model.predict_proba(X_scaled)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = df.copy()
    out[prob_col_name] = proba
    out['pred_tendinopathy'] = pred
    return out


@st.cache_resource
def load_test_results(path='results/test_results_temporal.csv'):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def evaluation_panel(df):
    st.header('Evaluation: Test set')

    st.write('Rows in test set:', len(df))

    # determine columns
    prob_col = None
    if 'prob' in df.columns:
        prob_col = 'prob'
    elif 'prob_tendinopathy' in df.columns:
        prob_col = 'prob_tendinopathy'

    if 'true_label' not in df.columns and 'true' in df.columns:
        df = df.rename(columns={'true': 'true_label'})

    if 'true_label' not in df.columns:
        st.warning('No true labels found in test set (no `true_label` column). Metrics unavailable.')
        st.dataframe(df.head())
        return

    y_true = df['true_label'].astype(int).values

    # default threshold slider
    if prob_col is not None:
        probs = df[prob_col].astype(float).values
        thresh = st.slider('Classification threshold', 0.0, 1.0, 0.5)
        y_pred = (probs >= thresh).astype(int)
    elif 'pred' in df.columns:
        y_pred = df['pred'].astype(int).values
        probs = None
        thresh = None
    else:
        st.warning('No probability or prediction column found. Cannot compute metrics.')
        st.dataframe(df.head())
        return

    # metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cols = st.columns(4)
    cols[0].metric('Accuracy', f'{acc:.3f}')
    cols[1].metric('Precision', f'{prec:.3f}')
    cols[2].metric('Recall', f'{rec:.3f}')
    cols[3].metric('F1', f'{f1:.3f}')

    if probs is not None:
        try:
            auc = roc_auc_score(y_true, probs)
        except Exception:
            auc = float('nan')
        brier = brier_score_loss(y_true, probs)
        st.write(f'ROC AUC: {auc:.3f}  —  Brier score: {brier:.4f}')

    # If plotly isn't available, show instructions and a compact metrics summary
    if not _HAS_PLOTLY:
        st.error('The `plotly` package is not installed in this environment. Install it with:')
        st.code('pip install plotly', language='bash')
        st.write('Metrics:')
        st.write(f'Accuracy: {acc:.3f}  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}')
        if probs is not None:
            try:
                auc = roc_auc_score(y_true, probs)
            except Exception:
                auc = float('nan')
            brier = brier_score_loss(y_true, probs)
            st.write(f'ROC AUC: {auc:.3f}  —  Brier score: {brier:.4f}')
        st.subheader('Test set preview')
        st.dataframe(df.head())
        return

    # confusion matrix (plotly)
    cm = confusion_matrix(y_true, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x='Predicted', y='Actual'), x=[0,1], y=[0,1], color_continuous_scale='Blues')
    st.plotly_chart(fig_cm, use_container_width=True)

    # ROC curve
    if probs is not None:
        fpr, tpr, _ = roc_curve(y_true, probs)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), showlegend=False))
        fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=400)
        st.plotly_chart(fig_roc, use_container_width=True)

        # Precision-Recall
        precision, recall_vals, _ = precision_recall_curve(y_true, probs)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall_vals, y=precision, mode='lines', name='PR'))
        fig_pr.update_layout(xaxis_title='Recall', yaxis_title='Precision', height=400)
        st.plotly_chart(fig_pr, use_container_width=True)

        # Probability histogram
        hist_df = pd.DataFrame({'prob': probs, 'true': y_true, 'pred': y_pred})
        fig_hist = px.histogram(hist_df, x='prob', color='true', nbins=20, barmode='overlay', opacity=0.7, labels={'true':'True label'})
        fig_hist.update_layout(height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader('Test set preview')
    st.dataframe(df.head())


def plot_shap_explanation(model_obj, scaler_obj, features_df, idx=0):
    """Generate SHAP waterfall plot for a single prediction"""
    if not _HAS_SHAP:
        st.error('SHAP not installed. Run: pip install shap')
        return None
    
    model = model_obj['model']
    feature_names = model_obj['features']
    
    # Get scaler
    if 'scaler' in model_obj:
        scaler = model_obj['scaler']
    else:
        scaler = scaler_obj
    
    # Prepare data
    X = features_df[feature_names].iloc[[idx]]
    X_scaled = scaler.transform(X) if scaler is not None else X.values
    X_scaled_flat = X_scaled[0]  # Get 1D array
    
    # For LogisticRegression, compute SHAP values from coefficients
    # SHAP value = coefficient * feature_value (for scaled features with mean=0)
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    shap_vals = coefficients * X_scaled_flat
    
    # Plot horizontal bar chart of SHAP values
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by absolute SHAP value
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:10]
    top_shap = shap_vals[sorted_idx]
    top_names = [feature_names[i] for i in sorted_idx]
    
    colors = ['#FF6B6B' if val > 0 else '#4ECDC4' for val in top_shap]
    ax.barh(range(len(top_shap)), top_shap, color=colors)
    ax.set_yticks(range(len(top_shap)))
    ax.set_yticklabels(top_names)
    ax.set_xlabel('SHAP Value (Impact on Prediction)')
    ax.set_title('Top 10 Feature Importance')
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.8)
    ax.invert_yaxis()
    plt.tight_layout()
    
    st.pyplot(fig, clear_figure=True)
    plt.close()
    
    # Return top features
    top_features = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_vals,
        'abs_shap': np.abs(shap_vals)
    }).sort_values('abs_shap', ascending=False).head(10)
    
    return top_features


def plot_pain_curve(raw_event_data, trial_info):
    """Plot pain vs event cycle"""
    if not _HAS_PLOTLY:
        st.error('Plotly not installed. Run: pip install plotly')
        return
    
    pain_values = raw_event_data['Pain_pred'].values
    cycles = range(len(pain_values))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(cycles), 
        y=pain_values,
        mode='lines+markers',
        name='Pain',
        line=dict(color='#FF6B6B', width=2),
        marker=dict(size=4)
    ))
    
    # Add peak pain marker
    peak_idx = np.argmax(pain_values)
    fig.add_trace(go.Scatter(
        x=[peak_idx],
        y=[pain_values[peak_idx]],
        mode='markers',
        name='Peak Pain',
        marker=dict(size=12, color='red', symbol='star')
    ))
    
    fig.update_layout(
        title=f"Pain Progression - {trial_info}",
        xaxis_title="Event Cycle",
        yaxis_title="Pain Prediction Value",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def generate_llm_explanation(prediction, prob, top_features, trial_info, groq_api_key=None, gemini_api_key=None):
    """Generate natural language explanation using Groq (primary) or Gemini (fallback) LLM"""
    
    # Build prompt
    condition = "Tendinopathy" if prediction == 1 else "Normal"
    confidence = prob if prediction == 1 else (1 - prob)
    
    top_features_text = "\n".join([
        f"- {row['feature']}: SHAP value = {row['shap_value']:.4f}"
        for _, row in top_features.head(5).iterrows()
    ])
    
    prompt = f"""You are a clinical biomechanics expert providing a comprehensive tendinopathy classification report.

**Model Prediction:**
- Classification: {condition}
- Confidence: {confidence*100:.1f}%

**Trial Information:**
{trial_info}

**Top Contributing Features (SHAP values):**
{top_features_text}

**Task:**
Provide a comprehensive analysis with the following sections:

## 1. CLINICAL INTERPRETATION (for Healthcare Professionals)
Provide a detailed clinical interpretation of this {condition} prediction with {confidence*100:.1f}% confidence. Explain what this means for patient assessment and diagnosis.

## 2. FEATURE EXPLANATION
Explain the top 3 biomechanical features that influenced this prediction:
- What each feature measures (e.g., peak_pain, pain_acceleration)
- Why high/low values of these features indicate tendinopathy or normal condition
- The clinical relevance of each feature

## 3. MOVEMENT PHASE ANALYSIS
Analyze the pain patterns across different movement phases:
- Early phase pain characteristics
- Mid-phase pain behavior
- Late phase pain progression
- What these patterns suggest about tissue health

## 4. REHABILITATION SUGGESTIONS (if Tendinopathy predicted)
Provide evidence-based rehabilitation recommendations:
- Activity modifications
- Load management strategies
- Progressive exercise considerations
- When to seek further clinical evaluation
(If Normal: suggest general maintenance strategies)

## 5. SIMPLE EXPLANATION (for Patients/Non-Experts)
Explain the results in simple, non-technical language that a patient without medical background can understand. Avoid jargon and use analogies where helpful.

Keep each section clear, actionable, and evidence-based."""

    # Try Groq API first (primary)
    if _HAS_OPENAI and groq_api_key and groq_api_key.strip():
        try:
            client = OpenAI(
                api_key=groq_api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a clinical biomechanics expert specializing in tendinopathy assessment."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return completion.choices[0].message.content, "Groq (Llama-3.1-70b)"
        except Exception as e:
            st.warning(f'⚠️ Groq API failed: {e}. Trying Gemini fallback...')
    
    # Fallback to Gemini API
    if not _HAS_GEMINI:
        st.error('Neither Groq nor Gemini API available. Install: pip install openai google-genai')
        return None, None
    
    if gemini_api_key is None or gemini_api_key.strip() == '':
        st.warning('No API keys provided. Enter Groq or Gemini API key in the sidebar.')
        return None, None
    
    # Try Gemini models
    client = genai.Client(api_key=gemini_api_key)
    
    model_names = [
        'gemini-2-0-flash-exp',
        'gemini-exp-1206',
        'gemini-2-5-flash-preview',
        'gemini-1-5-flash',
        'gemini-1-5-pro',
        'gemini-3-flash-preview'
    ]
    
    last_error = None
    
    for model_name in model_names:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            result = response.text if hasattr(response, 'text') else str(response)
            return result, f"Gemini ({model_name})"
        except Exception as e:
            last_error = e
            continue
    
    # If all models failed
    st.error(f'LLM generation failed with all models. Last error: {last_error}')
    return None, None


def realtime_prediction_mode(model_obj, scaler_obj):
    """Real-time prediction mode for unlabeled data"""
    st.header('🔬 Real-time Tendinopathy Prediction')
    
    # Groq API key: try .env first, then sidebar input (PRIMARY)
    st.sidebar.subheader('🤖 AI Configuration')
    groq_env_key = os.getenv('GROQ_API_KEY', '').strip()
    if groq_env_key:
        groq_key = groq_env_key
        st.sidebar.success('✓ Groq API key loaded from .env (Primary)')
    else:
        groq_key = st.sidebar.text_input('Groq API Key (Primary)', type='password', help='Get your key at https://console.groq.com/')
    
    # Gemini API key: try .env first, then sidebar input (FALLBACK)
    gemini_env_key = os.getenv('GEMINI_API_KEY', '').strip()
    if gemini_env_key:
        gemini_key = gemini_env_key
        st.sidebar.info('✓ Gemini API key loaded from .env (Fallback)')
    else:
        gemini_key = st.sidebar.text_input('Gemini API Key (Fallback)', type='password', help='Get your key at https://makersuite.google.com/app/apikey')
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 C3D File (Full Automation)", "📊 EventCycle Data (Processed)"])
    
    with tab1:
        st.caption('🚀 Upload C3D motion capture file for complete automated analysis')
        st.info('**New!** Upload raw C3D files directly - no OpenSim or MATLAB needed')
        
        # C3D file upload
        c3d_uploaded = st.file_uploader('Upload C3D file from motion capture', 
                                        type=['c3d'], key='c3d_upload')
        
        if c3d_uploaded is not None:
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.c3d') as tmp_file:
                tmp_file.write(c3d_uploaded.read())
                tmp_path = tmp_file.name
            
            try:
                # Import automation pipeline
                from automation_pipeline import c3d_to_prediction_pipeline
                
                st.write('Processing C3D file...')
                
                # Optional metadata inputs
                col1, col2, col3 = st.columns(3)
                with col1:
                    subject_id = st.number_input('Subject ID', min_value=1, value=1)
                with col2:
                    task = st.selectbox('Task', ['WithTask', 'Rest'])
                with col3:
                    speed = st.selectbox('Speed', ['Fast', 'Medium', 'Slow'])
                
                if st.button('🚀 Run Automated Analysis', type='primary'):
                    with st.spinner('Running complete pipeline: C3D → Kinematics → Forces → Pain Model → ML Prediction...'):
                        # Set intermediate_dir to intermediate/<uploaded_file_name_without_ext>
                        from pathlib import Path
                        upload_base = Path(c3d_uploaded.name).stem
                        intermediate_dir = os.path.join('intermediate', upload_base)
                        os.makedirs(intermediate_dir, exist_ok=True)
                        result = c3d_to_prediction_pipeline(
                            tmp_path,
                            model_obj,
                            scaler_obj,
                            condition='unknown',
                            subject_id=subject_id,
                            task=task,
                            speed=speed,
                            # Pass intermediate_dir for debug saving
                            intermediate_dir=intermediate_dir
                        )
                    
                    # Display prediction
                    pred_class = result['prediction']
                    prob = result['probability']
                    
                    st.markdown('---')
                    st.subheader('🎯 Prediction Results')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if pred_class == 1:
                            st.error(f'**Prediction:** TENDINOPATHY')
                        else:
                            st.success(f'**Prediction:** NORMAL')
                    
                    with col2:
                        confidence = result['confidence']
                        st.metric('Confidence', f'{confidence:.1%}')
                    
                    # Show pain curve
                    st.subheader('🔁 Pain Over Movement Cycle')
                    pain_df = result['pain_cycle_df']
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=pain_df['EventCycle'],
                        y=pain_df['Pain_pred'],
                        mode='lines',
                        line=dict(color='red' if pred_class == 1 else 'green', width=3),
                        name='Pain'
                    ))
                    fig.update_layout(
                        title='Predicted Pain Over Event Cycle',
                        xaxis_title='Event Cycle (%)',
                        yaxis_title='Pain (0-10)',
                        yaxis_range=[0, 10],
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # SHAP explanation
                    st.subheader('📊 Feature Importance (SHAP)')
                    features_df = result['temporal_features']
                    with st.spinner('📊 Calculating SHAP values...'):
                        top_features = plot_shap_explanation(model_obj, scaler_obj, features_df, idx=0)
                    
                    # LLM Explanation
                    st.subheader('🤖 AI Clinical Report')
                    trial_info = f"Subject {subject_id}, Task: {task}, Speed: {speed}"
                    
                    if top_features is not None:
                        with st.spinner('🤖 Generating comprehensive clinical report with AI... This may take 10-20 seconds.'):
                            explanation, api_used = generate_llm_explanation(pred_class, prob, top_features, trial_info, groq_key, gemini_key)
                        if explanation:
                            st.success(f'✅ Report generated successfully using {api_used}!')
                            st.markdown(explanation)
                        else:
                            st.error('Failed to generate report. Please check your API keys and try again.')
                    
                    # Summary metrics
                    st.subheader('📈 Summary Metrics')
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric('Peak Pain', f"{result['metadata']['peak_pain']:.2f}/10")
                    with metrics_col2:
                        st.metric('Mean Pain', f"{result['metadata']['mean_pain']:.2f}/10")
                    with metrics_col3:
                        st.metric('Duration', f"{result['metadata']['duration_s']:.1f}s")
                
            except Exception as e:
                st.error(f'Error processing C3D file: {e}')
                st.info('Make sure ezc3d is installed: pip install ezc3d')
                import traceback
                st.code(traceback.format_exc())
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    with tab2:
        st.caption('Upload pre-processed EventCycle data (CSV or Excel) for quick prediction')
        uploaded = st.file_uploader('Upload EventCycle file (CSV or Excel)', type=['csv', 'xlsx', 'xls'], key='realtime')
        
        if uploaded is not None:
            try:
                raw_df = read_uploaded_file(uploaded)
            except Exception as e:
                st.error(f'Failed to read file: {e}')
                return
            
            st.write(f'**Loaded:** {len(raw_df)} event cycles')
            
            # Validate required columns
            required_cols = ['Subject', 'Task', 'Speed', 'Pain_pred']
            missing = [c for c in required_cols if c not in raw_df.columns]
            
            if missing:
                st.error(f'Missing required columns: {missing}')
                st.info('Required: Subject, Task, Speed, EventCycle, Pain_pred')
                return
            
            # Extract temporal features
            with st.spinner('🔄 Extracting temporal features from event cycles...'):
                # Add dummy Condition for extraction function
                if 'Condition' not in raw_df.columns:
                    raw_df['Condition'] = 'unknown'
                features_df = extract_temporal_features_from_eventcycle(raw_df)
            
            if len(features_df) == 0:
                st.error('No trials extracted. Check your data format.')
                return
            
            # Predict
            # Predict
            if model_obj is not None:
                for f in model_obj['features']:
                    if f not in features_df.columns:
                        features_df[f] = 0
                
                preds = predict_df(model_obj, scaler_obj, features_df)
                
                # Display each trial
                for idx, row in preds.iterrows():
                    with st.container():
                        st.markdown('---')
                        
                        # Trial info
                        subj = row.get('Subject', 'Unknown')
                        task = row.get('Task', 'Unknown')
                        speed = row.get('Speed', 'Unknown')
                        trial_info = f"Subject {subj}, Task: {task}, Speed: {speed}"
                        
                        st.subheader(f"Trial: {trial_info}")
                        
                        # 1️⃣ Prediction
                        prob = row['prob_tendinopathy']
                        pred_class = int(row['pred_tendinopathy'])
                        condition = "🔴 **Tendinopathy**" if pred_class == 1 else "🟢 **Normal**"
                        confidence = prob if pred_class == 1 else (1 - prob)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric('Prediction', condition)
                        with col2:
                            st.metric('Confidence', f'{confidence*100:.1f}%')
                        
                        # 2️⃣ SHAP Feature Importance
                        st.subheader('2️⃣ Feature Importance (SHAP)')
                        try:
                            with st.spinner('📊 Calculating feature importance...'):
                                top_features = plot_shap_explanation(model_obj, scaler_obj, features_df, idx)
                        except Exception as e:
                            st.error(f'SHAP explanation failed: {e}')
                            top_features = None
                        
                        # 3️⃣ Pain Curve
                        st.subheader('3️⃣ Pain Progression')
                        try:
                            # Get raw event data for this trial
                            trial_raw = raw_df[
                                (raw_df['Subject'] == subj) & 
                                (raw_df['Task'] == task) & 
                                (raw_df['Speed'] == speed)
                            ].sort_values('EventCycle')
                            plot_pain_curve(trial_raw, trial_info)
                        except Exception as e:
                            st.error(f'Pain curve visualization failed: {e}')
                        
                        # 4️⃣ LLM Clinical Explanation
                        st.subheader('4️⃣ Comprehensive Clinical Report (AI-Generated)')
                        explanation = None
                        api_used = None
                        if top_features is not None:
                            with st.spinner('🤖 Generating comprehensive clinical report with AI... This may take 10-20 seconds.'):
                                explanation, api_used = generate_llm_explanation(pred_class, prob, top_features, trial_info, groq_key, gemini_key)
                        if explanation:
                            st.success(f'✅ Report generated successfully using {api_used}!')
                            st.markdown(explanation)
                        else:
                            st.error('Failed to generate report. Please check your API keys and try again.')

                        
                        # 5️⃣ Download Report
                        st.subheader('5️⃣ Download Report')
                        if top_features is not None:
                            # Build comprehensive report
                            report_text = f"""TENDINOPATHY CLASSIFICATION REPORT
{'='*60}

Trial Information:
{trial_info}

Prediction Results:
- Classification: {condition.replace('🔴 **', '').replace('🟢 **', '').replace('**', '')}
- Confidence: {confidence*100:.1f}%
- Probability (Tendinopathy): {prob:.3f}

Top 10 Contributing Features (SHAP Analysis):
"""
                            for i, feat_row in top_features.iterrows():
                                report_text += f"  {feat_row['feature']:25s} | SHAP: {feat_row['shap_value']:+.4f} | Importance: {feat_row['abs_shap']:.4f}\n"
                            
                            if explanation:
                                report_text += f"\n\nCOMPREHENSIVE CLINICAL ANALYSIS:\n{'='*60}\n"
                                if api_used:
                                    report_text += f"(Generated using: {api_used})\n\n"
                                report_text += f"{explanation}\n"
                            
                            report_text += f"\n\n{'='*60}\nReport Generated: {pd.Timestamp.now()}\n"
                            
                            # Download button
                            st.download_button(
                                label='📥 Download Complete Report (TXT)',
                                data=report_text,
                                file_name=f'tendinopathy_report_{subj}_{task}_{speed}.txt',
                                mime='text/plain',
                                key=f'download_txt_{idx}'
                            )
                            
                            # Also create CSV version with key data
                            csv_data = pd.DataFrame([{
                                'Subject': subj,
                                'Task': task,
                                'Speed': speed,
                                'Prediction': condition.replace('🔴 **', '').replace('🟢 **', '').replace('**', ''),
                                'Confidence': f'{confidence*100:.1f}%',
                                'Prob_Tendinopathy': prob,
                                'Top_Feature_1': top_features.iloc[0]['feature'] if len(top_features) > 0 else '',
                                'Top_Feature_1_SHAP': top_features.iloc[0]['shap_value'] if len(top_features) > 0 else 0,
                                'Top_Feature_2': top_features.iloc[1]['feature'] if len(top_features) > 1 else '',
                                'Top_Feature_2_SHAP': top_features.iloc[1]['shap_value'] if len(top_features) > 1 else 0,
                                'Top_Feature_3': top_features.iloc[2]['feature'] if len(top_features) > 2 else '',
                                'Top_Feature_3_SHAP': top_features.iloc[2]['shap_value'] if len(top_features) > 2 else 0,
                            }])
                            
                            st.download_button(
                                label='📥 Download Key Metrics (CSV)',
                                data=csv_data.to_csv(index=False),
                                file_name=f'tendinopathy_metrics_{subj}_{task}_{speed}.csv',
                                mime='text/csv',
                                key=f'download_csv_{idx}'
                            )


def model_evaluation_mode(model_obj, scaler_obj):
    """Model evaluation mode with labeled data (current functionality)"""
    st.header('📊 Model Evaluation & Testing')
    st.caption('Upload data WITH condition labels to evaluate model performance')

    st.header('1) Upload Raw Event-Cycle Data (Recommended)')
    st.caption('Upload your raw EventCycle CSV with columns: Subject, Condition, Task, Speed, EventCycle, Pain_pred. The app will automatically extract temporal features and predict.')
    
    uploaded_raw = st.file_uploader('Upload EventCycle file (CSV or Excel)', type=['csv', 'xlsx', 'xls'], key='raw_upload')

    if uploaded_raw is not None:
        try:
            raw_df = read_uploaded_file(uploaded_raw)
        except Exception as e:
            st.error(f'Failed to read file: {e}')
            return

        st.write('**Raw data loaded:**', len(raw_df), 'rows')
        st.dataframe(raw_df.head())

        # Validate required columns
        required_cols = ['Subject', 'Condition', 'Task', 'Speed', 'Pain_pred']
        missing = [c for c in required_cols if c not in raw_df.columns]
        
        if missing:
            st.error(f'Missing required columns: {missing}')
            st.info('Expected columns: Subject, Condition, Task, Speed, EventCycle, Pain_pred')
            return

        # Extract temporal features
        with st.spinner('Extracting temporal features...'):
            features_df = extract_temporal_features_from_eventcycle(raw_df)
        
        st.success(f'✓ Extracted features for {len(features_df)} trials')
        st.dataframe(features_df.head())

        if model_obj is not None:
            # Ensure model feature columns exist
            for f in model_obj['features']:
                if f not in features_df.columns:
                    st.warning(f'Model expects feature `{f}` but it is missing. Filling with 0.')
                    features_df[f] = 0

            preds = predict_df(model_obj, scaler_obj, features_df)
            st.subheader('Predictions')
            
            # Build output columns
            cols = ['Subject', 'Condition', 'Task', 'Speed']
            
            # Check if true labels exist
            has_true_label = 'true_label' in preds.columns
            
            if has_true_label:
                cols += ['true_label']
            
            cols += ['prob_tendinopathy', 'pred_tendinopathy']
            
            # Add key features
            key_features = ['peak_pain', 'mean_pain', 'pain_acceleration', 'time_to_peak']
            for feat in key_features:
                if feat in preds.columns and feat not in cols:
                    cols.append(feat)
            
            st.dataframe(preds[cols])
            
            # Show metrics if true labels are present
            if has_true_label:
                st.subheader('Performance Metrics')
                y_true = preds['true_label'].astype(int).values
                y_prob = preds['prob_tendinopathy'].values
                y_pred = preds['pred_tendinopathy'].astype(int).values
                
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                cols_metric = st.columns(4)
                cols_metric[0].metric('Accuracy', f'{acc:.3f}')
                cols_metric[1].metric('Precision', f'{prec:.3f}')
                cols_metric[2].metric('Recall', f'{rec:.3f}')
                cols_metric[3].metric('F1 Score', f'{f1:.3f}')
                
                try:
                    auc = roc_auc_score(y_true, y_prob)
                    st.write(f'**ROC AUC:** {auc:.3f}')
                except Exception:
                    pass
            
            # Download predictions
            csv_output = preds[cols].to_csv(index=False)
            st.download_button(
                label='Download predictions as CSV',
                data=csv_output,
                file_name='tendinopathy_predictions.csv',
                mime='text/csv'
            )

    st.header('2) Upload Temporal Features Data (Advanced)')
    st.caption('Upload a file with 23 temporal features extracted from EventCycle data. Use extract_temporal_features.py to generate this from raw EventCycle CSV/Excel.')
    uploaded = st.file_uploader('Upload temporal features file (CSV or Excel)', type=['csv', 'xlsx', 'xls'])

    if uploaded is not None:
        try:
            data_df = read_uploaded_file(uploaded)
        except Exception as e:
            st.error(f'Failed to read file: {e}')
            return

        st.write('Rows:', len(data_df))
        st.dataframe(data_df.head())

        if model_obj is not None:
            # ensure model feature columns exist
            for f in model_obj['features']:
                if f not in data_df.columns:
                    st.warning(f'Model expects feature `{f}` but it is missing. Filling with 0.')
                    data_df[f] = 0

            preds = predict_df(model_obj, scaler_obj, data_df)
            st.subheader('Predictions')
            
            # Build output columns
            cols = []
            if 'Subject' in preds.columns:
                cols += ['Subject']
            if 'Condition' in preds.columns:
                cols += ['Condition']
            
            # Check if true labels exist
            has_true_label = False
            true_label_col = None
            if 'true_label' in preds.columns:
                true_label_col = 'true_label'
                has_true_label = True
            elif 'target' in preds.columns:
                true_label_col = 'target'
                has_true_label = True
            elif 'label' in preds.columns:
                true_label_col = 'label'
                has_true_label = True
            
            if has_true_label:
                cols += [true_label_col]
            
            cols += ['prob_tendinopathy', 'pred_tendinopathy']
            
            # Add key features at the end
            key_features = ['peak_pain', 'mean_pain', 'pain_acceleration', 'time_to_peak']
            for feat in key_features:
                if feat in preds.columns and feat not in cols:
                    cols.append(feat)
            
            st.dataframe(preds[cols])
            
            # Show metrics if true labels are present
            if has_true_label:
                st.subheader('Performance Metrics')
                y_true = preds[true_label_col].astype(int).values
                y_prob = preds['prob_tendinopathy'].values
                y_pred = preds['pred_tendinopathy'].astype(int).values
                
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                cols_metric = st.columns(4)
                cols_metric[0].metric('Accuracy', f'{acc:.3f}')
                cols_metric[1].metric('Precision', f'{prec:.3f}')
                cols_metric[2].metric('Recall', f'{rec:.3f}')
                cols_metric[3].metric('F1 Score', f'{f1:.3f}')
                
                try:
                    auc = roc_auc_score(y_true, y_prob)
                    st.write(f'**ROC AUC:** {auc:.3f}')
                except Exception:
                    pass

    st.header('3) Model Evaluation (Interactive)')
    st.write('Load the `test_results_temporal.csv` from training, or upload your own test file with `true_label` and `prob`/`pred` columns.')

    # Option to load bundled test_results_temporal.csv or upload
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button('Load test_results_temporal.csv'):
            test_df = load_test_results('results/test_results_temporal.csv')
        else:
            test_df = None
    with col2:
        uploaded_test = st.file_uploader('Or upload test file (CSV or Excel)', type=['csv', 'xlsx', 'xls'], key='test_upload')
        if uploaded_test is not None:
            try:
                test_df = read_uploaded_file(uploaded_test)
            except Exception as e:
                st.error(f'Failed to read uploaded test file: {e}')
                test_df = None

    if test_df is not None:
        evaluation_panel(test_df)


def main():
    """Main function with sidebar mode selector"""
    st.set_page_config(page_title='Tendinopathy Classifier', page_icon='🏥', layout='wide')
    
    # Sidebar: Mode selection
    st.sidebar.title('🏥 Tendinopathy Classifier')
    mode = st.sidebar.radio(
        'Select Mode',
        ['Real-time Prediction', 'Model Evaluation'],
        help='**Real-time**: Predict on unlabeled data with SHAP explainability and LLM insights\n\n**Evaluation**: Test model performance on labeled data'
    )
    
    # Load model once
    st.sidebar.header('Model Status')
    model_obj = load_saved_model('model_temporal.joblib')
    scaler_obj = load_saved_scaler('scaler_temporal.joblib')
    
    if model_obj is None:
        st.sidebar.error('❌ No model found')
        st.error('Model file `model_temporal.joblib` not found. Run `train_temporal_model.py` first.')
        return
    else:
        st.sidebar.success('✓ Model loaded')
        st.sidebar.info(f'Features: {len(model_obj["features"])} temporal features')
        st.sidebar.caption('Model: Logistic Regression with L2 regularization')
        
        # Check scaler
        if 'scaler' in model_obj:
            st.sidebar.success('✓ Scaler included')
        elif scaler_obj is not None:
            st.sidebar.success('✓ Scaler loaded (legacy)')
        else:
            st.sidebar.warning('⚠ No scaler found')
    
    # Route to appropriate mode
    if mode == 'Real-time Prediction':
        realtime_prediction_mode(model_obj, scaler_obj)
    else:
        model_evaluation_mode(model_obj, scaler_obj)


if __name__ == '__main__':
    main()
