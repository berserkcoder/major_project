import streamlit as st
from PIL import Image
from utils import load_pipeline, load_data
from scripts.config import FIG_DIR, WINDOW_SIZE

st.set_page_config(
    page_title="Network Traffic Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)
st.markdown("""
<style>
    .main { background-color: #FFFFFF; }
    .stMetric { background-color: #F8F9FA; padding: 15px; border-radius: 10px; border: 1px solid #E9ECEF; }
    [data-testid="stSidebar"] { background-color: #F8F9FA; border-right: 1px solid #E9ECEF; }
    h1, h2, h3 { color: #212529; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Network Traffic Anomaly Detection & FP Reduction")
st.markdown("*Interactive dashboard for three-stage intrusion detection.*")

s1, scaler, s2, threshold, s3_model, s3_scaler, s3_classes = load_pipeline()
df_windowed, df_normal, attack_stats, expected_cols = load_data()

col1, col2, col3, col4 = st.columns(4)
attack_count = int(df_windowed['window_attack'].sum())
normal_count = len(df_windowed) - attack_count
feature_count = len([c for c in expected_cols if c not in ['window_id', 'window_attack']])

with col1:
    st.metric("Test Windows", f"{len(df_windowed):,}")
with col2:
    st.metric("Total Attacks", attack_count)
with col3:
    st.metric("Normal Windows", normal_count)
with col4:
    st.metric("Features", feature_count)

st.markdown("---")

# ── Three-stage pipeline overview ─────────────────────────────────────────
col1, col2, col3 = st.columns(3)

s1_name, s1_desc = "Isolation Forest", "Detects statistical anomalies by isolating observations in feature space."
s2_name, s2_desc = "Random Forest", "Supervised filter that distinguishes true attacks from noisy anomalies."
s3_name, s3_desc = "Random Forest (Multi-class)", "Classifies confirmed attacks into categories: DoS, Exploits, Fuzzers, or Generic."
fp_red = "88.6%"
s3_status = "Active" if s3_model is not None else "Not trained yet — run `python scripts/train_stage3.py`"
s3_categories = ", ".join(s3_classes) if s3_classes else "DoS, Exploits, Fuzzers, Generic"

with col1:
    st.markdown(f"### 🔍 Stage 1: {s1_name}")
    st.markdown(f"""
    {s1_desc}
    - **Current Status**: Active
    - **Threshold Type**: {'Negative Score' if 'Isolation' in s1_name else 'MSE (90th Pctl)'}
    """)

with col2:
    st.markdown(f"### 🎯 Stage 2: {s2_name}")
    st.markdown(f"""
    {s2_desc}
    - **FP Reduction**: **{fp_red}**
    - **Current Status**: Active
    """)

with col3:
    st.markdown(f"### 🏷️ Stage 3: {s3_name}")
    st.markdown(f"""
    {s3_desc}
    - **Categories**: {s3_categories}
    - **Current Status**: {s3_status}
    """)

st.markdown("---")

if FIG_DIR.exists() and (FIG_DIR / "anomaly_scores.png").exists():
    try:
        st.markdown("### Model Comparison: Training Distribution")
        image = Image.open(FIG_DIR / "anomaly_scores.png")
        st.image(image, use_column_width=True)
    except:
        pass

st.markdown("---")

st.markdown("### 📊 Dataset: UNSW-NB15")
st.markdown(f"""
- Window Size: **{WINDOW_SIZE} flows** per aggregation
- Normal Library: {len(df_normal):,} baseline windows
- Testing Pool: {len(df_windowed):,} mixed test windows
""")

st.markdown("---")

st.info("""
**Navigate using the sidebar:**
- **Replay Monitor**: Live simulation of detection on the test set.
- **Live Monitor**: Real-time packet capture monitoring.
- **Comparative Analysis**: Side-by-side model comparison.
- **Model Performance**: Detailed metrics for all three stages.
- **Attack Category**: Stage 3 category breakdown and analysis.
""")
