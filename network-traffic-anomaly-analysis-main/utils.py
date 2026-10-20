import streamlit as st
import joblib
import pandas as pd
import json
from scripts.config import MODEL_DIR, PROCESSED_DIR

@st.cache_resource
def load_pipeline():
    """Loads the full 3-stage detection pipeline."""
    # Stage 1: Isolation Forest (anomaly detection)
    iso = joblib.load(MODEL_DIR / 'isolation_forest.pkl')
    # Shared scaler
    scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
    # Stage 2: Random Forest (false-positive reduction)
    rf = joblib.load(MODEL_DIR / 'rf_stage2.pkl')

    # Stage 3: Attack Category Classifier (loads only if model file exists)
    s3_model   = None
    s3_scaler  = None
    s3_classes = None
    s3_path = MODEL_DIR / 'rf_stage3_category.pkl'
    if s3_path.exists():
        s3_model   = joblib.load(MODEL_DIR / 'rf_stage3_category.pkl')
        s3_scaler  = joblib.load(MODEL_DIR / 'rf_stage3_category_scaler.pkl')
        s3_classes = joblib.load(MODEL_DIR / 'rf_stage3_category_classes.pkl')

    return iso, scaler, rf, 0.0262, s3_model, s3_scaler, s3_classes


@st.cache_data
def load_data():
    df_windowed  = pd.read_csv(PROCESSED_DIR / 'test_windowed_results.csv')
    df_normal    = pd.read_csv(PROCESSED_DIR / 'normal_windowed.csv')
    attack_stats = pd.read_csv(PROCESSED_DIR / 'attack_category_results.csv')

    with open(PROCESSED_DIR / 'windowed_columns.json') as f:
        expected_cols = json.load(f)

    return df_windowed, df_normal, attack_stats, expected_cols


def predict_window(window_features, feature_cols, scaler, s1_model, s2_model,
                   s3_model=None, s3_scaler=None, s3_classes=None):
    """
    Run a single window through the full 3-stage pipeline.

    Returns
    -------
    score     : float  – Stage 1 suspicion score (higher = more suspicious)
    threshold : float  – Decision threshold for Stage 1
    s1        : int    – 1 if Stage 1 flagged the window, else 0
    s2        : int    – 1 if Stage 2 confirmed attack, else 0
    category  : str    – Attack category from Stage 3, or '' if not an attack
    """
    X = window_features[feature_cols].values

    X_scaled = scaler.transform(X)

    # Stage 1
    raw_score = s1_model.decision_function(X_scaled)[0]
    score     = -raw_score
    threshold = -0.0262
    s1        = int(raw_score < 0.0262)

    # Stage 2
    s2 = s1
    if s1 == 1:
        s2 = int(s2_model.predict(X)[0])

    # Stage 3 — only runs when Stage 2 confirms an attack
    category = ''
    if s2 == 1 and s3_model is not None and s3_scaler is not None:
        X_s3 = s3_scaler.transform(X)
        category = str(s3_model.predict(X_s3)[0])

    return score, threshold, s1, s2, category