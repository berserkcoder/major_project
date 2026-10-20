import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scripts.config import PROCESSED_DIR, STAGE1_MODELS, STAGE2_MODELS

st.set_page_config(page_title="Comparative Analysis", page_icon="📊", layout="wide")
st.markdown("""
<style>
    .main { background-color: #FFFFFF; }
    .stMetric { background-color: #F8F9FA; padding: 10px; border-radius: 8px; border: 1px solid #E9ECEF; }
    h1, h2, h3 { color: #212529; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Comparative Analysis")

tab1, tab2, tab3 = st.tabs([
    "Stage 1 Baselines",
    "Stage 2 Models",
    "Category-Specific Models"
])

with tab1:
    st.markdown("### Stage 1: Anomaly Detection Models")
    
    col1, col2, col3 = st.columns(3)
    
    models_info = [
        ("Isolation Forest", "Operational",
         f"Estimators: {STAGE1_MODELS['isolation_forest']['n_estimators']}\nContamination: {STAGE1_MODELS['isolation_forest']['contamination']}"),
        ("Local Outlier Factor", "Operational",
         f"Neighbors: {STAGE1_MODELS['lof']['n_neighbors']}\nContamination: {STAGE1_MODELS['lof']['contamination']}"),
        ("Autoencoder", "Operational",
         f"Encoding Dim: {STAGE1_MODELS['autoencoder']['encoding_dim']}\nEpochs: {STAGE1_MODELS['autoencoder']['epochs']}")
    ]
    
    for col, (name, status, params) in zip([col1, col2, col3], models_info):
        with col:
            st.markdown(f"#### {name}")
            st.code(params, language="text")
            st.caption(status)
    
    results_file = PROCESSED_DIR / 'stage1_predictions.csv'
    if results_file.exists():
        st.success("Results available!")
        df = pd.read_csv(results_file)
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.info("Run comparative analysis notebooks to generate results.")

with tab2:
    st.markdown("### Stage 2: False Positive Reduction Models")
    st.markdown("Compare three classifiers for filtering Stage 1 false alerts (9 total combinations).")
    
    col1, col2, col3 = st.columns(3)
    
    stage2_info = [
        ("Random Forest", "Production",
         f"Estimators: {STAGE2_MODELS['random_forest']['n_estimators']}\nMax Depth: {STAGE2_MODELS['random_forest']['max_depth']}"),
        ("SVM", "Candidate",
         f"Kernel: {STAGE2_MODELS['svm']['kernel']}\nC: {STAGE2_MODELS['svm']['C']}"),
        ("XGBoost", "Candidate",
         f"Estimators: {STAGE2_MODELS['xgboost']['n_estimators']}\nMax Depth: {STAGE2_MODELS['xgboost']['max_depth']}")
    ]
    
    for col, (name, status, params) in zip([col1, col2, col3], stage2_info):
        with col:
            st.markdown(f"#### {name}")
            st.code(params, language="text")
            st.caption(status)
    
    results_file = PROCESSED_DIR / 'stage2_comparison.csv'
    if results_file.exists():
        st.success("Results available!")
        df = pd.read_csv(results_file)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Run comparative analysis notebooks to generate results.")

with tab3:
    st.markdown("### Per-Attack-Category Modeling")
    st.markdown("""
    **Hypothesis**: Different attack types have different anomaly signatures.
    Train separate detectors per category for better detection.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Attack Categories**:")
        categories = [
            "DoS — Denial of Service (traffic surge)",
            "Backdoor — Unauthorized access",
            "Reconnaissance — Information gathering",
            "Exploit — Vulnerability explotation",
            "Others — Worms, Shellcode, Fuzzers, etc."
        ]
        for cat in categories:
            st.markdown(f"- {cat}")
    
    with col2:
        st.markdown("**Benefits**:")
        st.markdown("""
        - Category-specific threshold
        - Adaptive detector selection
        - Better recall per attack type
        - Reduced FP per category
        """)
    
    st.markdown("---")
    st.markdown("""
    **Approach**:
    1. Separate training by attack category
    2. Train IF/LOF/AE per category
    3. At detection, route to best detector for that category
    """)
    
    results_file = PROCESSED_DIR / 'category_specific_results.csv'
    if results_file.exists():
        st.success("Results !")
        df = pd.read_csv(results_file)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Run category-specific analysis notebook to generate results.")
