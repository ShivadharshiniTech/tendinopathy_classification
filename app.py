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
    import google.generativeai as genai
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
    
    # Create SHAP explainer
    explainer = shap.LinearExplainer(model, X_scaled, feature_names=feature_names)
    shap_values = explainer(X_scaled)
    
    # Plot waterfall
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig, clear_figure=True)
    plt.close()
    
    # Return top features
    shap_vals = shap_values.values[0]
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


def generate_llm_explanation(prediction, prob, top_features, trial_info, gemini_api_key=None):
    """Generate natural language explanation using Gemini LLM"""
    if not _HAS_GEMINI:
        st.error('Google GenerativeAI not installed. Run: pip install google-generativeai')
        return None
    
    if gemini_api_key is None or gemini_api_key.strip() == '':
        st.warning('No Gemini API key provided. Enter your key in the sidebar.')
        return None
    
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Build prompt
        condition = "Tendinopathy" if prediction == 1 else "Normal"
        confidence = prob if prediction == 1 else (1 - prob)
        
        top_features_text = "\n".join([
            f"- {row['feature']}: SHAP value = {row['shap_value']:.4f}"
            for _, row in top_features.head(5).iterrows()
        ])
        
        prompt = f"""You are a clinical biomechanics expert explaining tendinopathy classification results to a healthcare professional.

**Model Prediction:**
- Classification: {condition}
- Confidence: {confidence*100:.1f}%

**Trial Information:**
{trial_info}

**Top Contributing Features (SHAP values):**
{top_features_text}

**Task:**
Write a clear, concise 3-4 sentence clinical explanation for this prediction. Include:
1. The prediction and confidence level
2. Which biomechanical features most influenced the decision
3. What this means clinically (e.g., pain patterns, movement dynamics)

Keep it professional but accessible to clinicians."""

        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        st.error(f'LLM generation failed: {e}')
        return None


def realtime_prediction_mode(model_obj, scaler_obj):
    """Real-time prediction mode for unlabeled data"""
    st.header('🔬 Real-time Tendinopathy Prediction')
    st.caption('Upload patient EventCycle data to get instant prediction with explainable AI')
    
    # Gemini API key: try .env first, then sidebar input
    env_key = os.getenv('GEMINI_API_KEY', '').strip()
    if env_key:
        gemini_key = env_key
        st.sidebar.success('✓ Gemini API key loaded from .env')
    else:
        gemini_key = st.sidebar.text_input('Gemini API Key (for LLM explanations)', type='password', help='Get your key at https://makersuite.google.com/app/apikey or set GEMINI_API_KEY in .env file')
    
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
        with st.spinner('Extracting temporal features...'):
            # Add dummy Condition for extraction function
            if 'Condition' not in raw_df.columns:
                raw_df['Condition'] = 'unknown'
            features_df = extract_temporal_features_from_eventcycle(raw_df)
        
        if len(features_df) == 0:
            st.error('No trials extracted. Check your data format.')
            return
        
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
                    if _HAS_SHAP:
                        top_features = plot_shap_explanation(model_obj, scaler_obj, features_df, idx)
                        if top_features is not None:
                            st.dataframe(top_features[['feature', 'shap_value']].head(10), hide_index=True)
                    else:
                        st.warning('Install SHAP for explainability: `pip install shap`')
                        top_features = None
                    
                    # 3️⃣ Pain Curve Visualization
                    st.subheader('3️⃣ Pain Progression Curve')
                    # Get raw event cycles for this trial
                    trial_mask = (raw_df['Subject'] == subj) & (raw_df['Task'] == task) & (raw_df['Speed'] == speed)
                    trial_data = raw_df[trial_mask]
                    
                    if len(trial_data) > 0:
                        plot_pain_curve(trial_data, trial_info)
                    else:
                        st.warning('Could not find raw event cycle data for visualization')
                    
                    # 4️⃣ LLM Explanation
                    st.subheader('4️⃣ Clinical Explanation (AI-Generated)')
                    if _HAS_GEMINI and gemini_key and top_features is not None:
                        with st.spinner('Generating explanation...'):
                            explanation = generate_llm_explanation(pred_class, prob, top_features, trial_info, gemini_key)
                            if explanation:
                                st.info(explanation)
                    elif not _HAS_GEMINI:
                        st.warning('Install google-generativeai: `pip install google-generativeai`')
                    elif not gemini_key:
                        st.warning('Enter your Gemini API key in the sidebar for AI explanations')
                    
                    # Download individual report
                    report_data = {
                        'Trial': trial_info,
                        'Prediction': 'Tendinopathy' if pred_class == 1 else 'Normal',
                        'Confidence': f'{confidence*100:.1f}%',
                        'Peak_Pain': row.get('peak_pain', 'N/A'),
                        'Mean_Pain': row.get('mean_pain', 'N/A'),
                        'Pain_Acceleration': row.get('pain_acceleration', 'N/A')
                    }
                    report_df = pd.DataFrame([report_data])
                    csv = report_df.to_csv(index=False)
                    st.download_button(
                        'Download Prediction Report (CSV)',
                        csv,
                        f'prediction_{subj}_{task}_{speed}.csv',
                        'text/csv',
                        key=f'download_{idx}'
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

    st.header('3) Live Replay Simulator')
    st.caption('Upload temporal features file to replay predictions row-by-row.')
    sim_file = st.file_uploader('Upload temporal features file (CSV or Excel)', type=['csv', 'xlsx', 'xls'], key='sim')
    if sim_file is not None:
        try:
            sim_df = read_uploaded_file(sim_file)
        except Exception as e:
            st.error(f'Failed to read file: {e}')
            return
        start = st.button('Start simulation')
        if start:
            placeholder = st.empty()
            for _, row in sim_df.iterrows():
                text = ''
                if 'Subject' in row:
                    text += f'Subject {row.Subject} '
                if 'Condition' in row:
                    text += f'— {row.Condition} '
                
                # Check for true label
                true_label = None
                if 'true_label' in row:
                    true_label = int(row.true_label)
                elif 'target' in row:
                    true_label = int(row.target)
                elif 'label' in row:
                    true_label = int(row.label)
                
                if true_label is not None:
                    text += f'— True: {true_label} '
                
                # ensure features
                r = row.to_frame().T
                for f in (model_obj['features'] if model_obj is not None else []):
                    if f not in r.columns:
                        r[f] = 0
                if model_obj is not None:
                    p = predict_df(model_obj, scaler_obj, r)[['prob_tendinopathy', 'pred_tendinopathy']].iloc[0]
                    pred_class = int(p.pred_tendinopathy)
                    text += f'— Pred: {pred_class} (prob={p.prob_tendinopathy:.3f})'
                    
                    # Add checkmark or X if we have true label
                    if true_label is not None:
                        if pred_class == true_label:
                            text += ' ✓'
                        else:
                            text += ' ✗'
                
                placeholder.write(text)
                time.sleep(0.6)

    st.header('4) Model Evaluation (Interactive)')
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
