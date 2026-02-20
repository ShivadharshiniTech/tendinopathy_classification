import streamlit as st
import joblib
import time
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    px = None
    go = None
    _HAS_PLOTLY = False

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
def load_test_results(path='test_results_temporal.csv'):
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


def main():
    st.title('Tendinopathy Classifier — Temporal Features Model')

    st.sidebar.header('Model')
    model_obj = load_saved_model('model_temporal.joblib')
    scaler_obj = load_saved_scaler('scaler_temporal.joblib')
    
    if model_obj is None:
        st.sidebar.warning('No model_temporal.joblib found. Run `train_temporal_model.py` first.')
    else:
        st.sidebar.success('✓ Loaded temporal features model')
        st.sidebar.info(f'Features: {len(model_obj["features"])} temporal features')
        st.sidebar.caption('Model: Logistic Regression with L2 regularization')
        
        # Check if scaler is in model or needs separate file
        if 'scaler' in model_obj:
            st.sidebar.success('✓ Feature scaler included in model')
        elif scaler_obj is not None:
            st.sidebar.success('✓ Loaded feature scaler (legacy format)')
        else:
            st.sidebar.warning('⚠ No scaler found. Predictions may be incorrect!')

    st.header('1) Upload Temporal Features Data')
    st.caption('Upload a CSV with 23 temporal features extracted from EventCycle data. Use extract_temporal_features.py to generate this from raw EventCycle CSV.')
    uploaded = st.file_uploader('Upload temporal features CSV', type=['csv'])

    if uploaded is not None:
        try:
            data_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f'Failed to read CSV: {e}')
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

    st.header('2) Live Replay Simulator')
    st.caption('Upload temporal features CSV to replay predictions row-by-row.')
    sim_file = st.file_uploader('Upload temporal features CSV', type=['csv'], key='sim')
    if sim_file is not None:
        try:
            sim_df = pd.read_csv(sim_file)
        except Exception as e:
            st.error(f'Failed to read CSV: {e}')
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

    st.header('3) Model Evaluation (Interactive)')
    st.write('Load the `test_results_temporal.csv` from training, or upload your own test CSV with `true_label` and `prob`/`pred` columns.')

    # Option to load bundled test_results_temporal.csv or upload
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button('Load test_results_temporal.csv'):
            test_df = load_test_results('test_results_temporal.csv')
        else:
            test_df = None
    with col2:
        uploaded_test = st.file_uploader('Or upload test CSV', type=['csv'], key='test_upload')
        if uploaded_test is not None:
            try:
                test_df = pd.read_csv(uploaded_test)
            except Exception as e:
                st.error(f'Failed to read uploaded test CSV: {e}')
                test_df = None

    if test_df is not None:
        evaluation_panel(test_df)


if __name__ == '__main__':
    main()
