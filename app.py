import streamlit as st
import joblib
import time
from pathlib import Path
import pandas as pd

from utils import load_summary_csv


@st.cache_resource
def load_saved_model(path='model.joblib'):
    p = Path(path)
    if not p.exists():
        return None
    return joblib.load(p)


def predict_df(model_obj, df):
    model = model_obj['model']
    features = model_obj['features']
    X = df[features]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = df.copy()
    out['prob_tendinopathy'] = proba
    out['pred_tendinopathy'] = pred
    return out


def main():
    st.title('Tendinopathy classifier — demo')

    st.sidebar.header('Model')
    model_obj = load_saved_model('model.joblib')
    if model_obj is None:
        st.sidebar.warning('No model.joblib found. Run `train_and_save_model.py` first.')
    else:
        st.sidebar.success('Loaded model.joblib')

    st.header('1) Upload data (per-row)')
    uploaded = st.file_uploader('Upload CSV where each row is one observation', type=['csv'])

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

            preds = predict_df(model_obj, data_df)
            st.subheader('Predictions')
            cols = []
            if 'Subject' in preds.columns:
                cols += ['Subject']
            if 'Condition' in preds.columns:
                cols += ['Condition']
            cols += model_obj['features'] + ['prob_tendinopathy', 'pred_tendinopathy']
            st.dataframe(preds[cols])

    st.header('2) Live replay simulator (CSV rows)')
    sim_file = st.file_uploader('Upload CSV to replay rows', type=['csv'], key='sim')
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
                # ensure features
                r = row.to_frame().T
                for f in (model_obj['features'] if model_obj is not None else []):
                    if f not in r.columns:
                        r[f] = 0
                if model_obj is not None:
                    p = predict_df(model_obj, r)[['prob_tendinopathy', 'pred_tendinopathy']].iloc[0]
                    text += f'— Pred prob={p.prob_tendinopathy:.3f} — class={int(p.pred_tendinopathy)}'
                placeholder.write(text)
                time.sleep(0.6)


if __name__ == '__main__':
    main()
