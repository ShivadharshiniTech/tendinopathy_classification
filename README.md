# Tendinopathy Classification Demo

This repository contains a complete pipeline for classifying tendinopathy using temporal biomechanical features extracted from event-cycle pain data.

## Project Structure

```
├── dataset/                          # Input data (gitignored)
├── results/                          # Model metrics and test results
│   ├── metrics_simple_models.csv
│   ├── metrics_simple_models.json
│   ├── metrics_temporal.json
│   ├── test_results_temporal.csv
│   └── test_set_temporal.csv
├── images/                           # Confusion matrices and plots
│   ├── cm_<model>_train.png
│   ├── cm_<model>_test.png
│   └── learning_curve.png
├── extract_temporal_features.py     # Extract 23 temporal features from event-cycle data
├── train_temporal_model.py          # Train model on temporal features (Logistic Regression)
├── compare_simple_models.py         # Compare KNN, Decision Tree, Naive Bayes, LDA, Logistic
├── compute_temporal_metrics.py      # Compute metrics from saved model
├── app.py                           # Streamlit demo app
├── utils.py                         # Helper functions for CSV loading
├── model_temporal.joblib            # Trained model + scaler + features
├── scaler_temporal.joblib           # Scaler (for backward compatibility)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Quick Start

### 1. Setup Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Extract Temporal Features

```bash
python extract_temporal_features.py
```

This reads `dataset/PainModel_EventCycle_AllSubjects_1 (1).csv` and creates `dataset/temporal_features.csv` with 23 engineered features per trial.

### 3. Train Model

Train logistic regression model with group-aware split (by Subject):

```bash
python train_temporal_model.py --model logistic --test-size 0.25 --random-state 42
```

Or compare multiple simple algorithms:

```bash
python compare_simple_models.py
```

### 4. Run Streamlit Demo

```bash
streamlit run app.py
```

**The app now has TWO modes:**

#### 🔮 Real-time Prediction Mode
For **unlabeled** data (without condition labels) - get predictions with explainability:
1. Upload raw EventCycle data (CSV or Excel with 101 cycles)
2. For each trial, see:
   - **Prediction**: Normal vs Tendinopathy with confidence score
   - **SHAP Explanation**: Feature importance waterfall plot showing which features drove the prediction
   - **Pain Curve**: Visualization of pain progression over 101 event cycles
   - **Comprehensive AI Report**: Detailed clinical analysis powered by Groq (Llama-3.1-70b) or Gemini (fallback)
     - Clinical interpretation for healthcare professionals
     - Feature explanations
     - Movement phase analysis
     - Rehabilitation suggestions
     - Simple explanation for patients
3. **Download Reports**: Get complete TXT report or key metrics CSV

**AI API Keys (Optional for LLM Explanations):**
- **Primary**: Groq API key from [Groq Console](https://console.groq.com/)
- **Fallback**: Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Set in `.env` file or enter in sidebar
- App automatically tries Groq first, then falls back to Gemini if unavailable

#### 📊 Model Evaluation Mode
For **labeled** data (with condition column) - test model performance:
1. Upload raw Event-Cycle data or pre-extracted temporal features
2. See predictions with key features
3. View performance metrics (accuracy, precision, recall, F1, AUC)
4. Interactive confusion matrix and ROC/PR curves

**Supported file formats:** CSV (.csv), Excel (.xlsx, .xls)

## Features

- **Temporal Feature Extraction**: 23 features including peak pain, pain slope, curvature, early/mid/late pain, statistical moments
- **Group-Aware Splitting**: Train/test split by Subject to prevent data leakage
- **Multiple Algorithms**: KNN, Decision Tree, Naive Bayes, LDA, Logistic Regression
- **Cross-Validation**: Leave-One-Group-Out CV for robust evaluation
- **Organized Outputs**: Results and images saved to dedicated folders

## Models Trained

- **Logistic Regression** (L2 regularization, C=1.0)
- **KNN** (k=3, distance-weighted)
- **Decision Tree** (max_depth=5, class_weight='balanced')
- **Naive Bayes** (Gaussian)
- **LDA** (with shrinkage)
