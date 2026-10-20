"""
Stage 3: Attack Category Analysis
Dedicated dashboard page for attack category classification results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scripts.config import PROCESSED_DIR, ATTACK_CATEGORIES_STAGE3, CATEGORY_COLORS

st.set_page_config(page_title="Attack Category", page_icon="🏷️", layout="wide")
st.markdown("""
<style>
    .main { background-color: #FFFFFF; }
    .stMetric { background-color: #F8F9FA; padding: 10px; border-radius: 8px; border: 1px solid #E9ECEF; }
    h1, h2, h3 { color: #212529; }
</style>
""", unsafe_allow_html=True)

st.title("🏷️ Stage 3 — Attack Category Classification")
st.markdown("Multi-class classifier that identifies the type of confirmed cyber attacks.")

# ── Pipeline description ──────────────────────────────────────────────────
with st.expander("ℹ️ How Stage 3 works", expanded=False):
    st.markdown("""
    **Stage 3** receives windows that Stage 2 has confirmed as attacks and classifies them into one of
    four attack categories based on traffic feature patterns:

    | Category | Signature |
    |---|---|
    | **DoS** | High packet rate, large byte volumes, many interrupted connections |
    | **Exploits** | Targets specific services: HTTP, FTP, SSH, SSL |
    | **Fuzzers** | Many different protocols, erratic connection durations |
    | **Generic** | Broad-signature attacks — remaining traffic patterns |

    The classifier is a **Random Forest** with `class_weight='balanced'` to handle imbalanced categories.
    It is trained only on confirmed attack windows using rule-derived labels from UNSW-NB15 signatures.
    """)

st.markdown("---")

s3_file = PROCESSED_DIR / 'stage3_results.csv'

if not s3_file.exists():
    st.warning("""
    **Stage 3 model has not been trained yet.**

    Train it by running the following command from your project directory:
    ```bash
    python scripts/train_stage3.py
    ```
    Then refresh this page.
    """)
    st.stop()

# ── Load results ─────────────────────────────────────────────────────────
df = pd.read_csv(s3_file)

# ── Top-level metrics ─────────────────────────────────────────────────────
total   = len(df)
correct = (df['attack_cat'] == df['predicted_cat']).sum()
acc     = correct / total if total > 0 else 0
cats    = df['predicted_cat'].nunique()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Attack Windows",     total)
col2.metric("Correctly Labelled", correct)
col3.metric("Overall Accuracy",   f"{acc:.1%}")
col4.metric("Categories Found",   cats)

st.markdown("---")

# ── Charts row ────────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("### Predicted Category Distribution")
    cat_dist = df['predicted_cat'].value_counts().reset_index()
    cat_dist.columns = ['Category', 'Count']

    fig_donut = px.pie(
        cat_dist, names='Category', values='Count',
        hole=0.5,
        color='Category',
        color_discrete_map=CATEGORY_COLORS,
        title="Attack Type Breakdown",
    )
    fig_donut.update_traces(textposition='outside', textinfo='percent+label')
    fig_donut.update_layout(
        height=380, paper_bgcolor='white',
        showlegend=True,
        legend=dict(orientation='h', y=-0.1)
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with chart_col2:
    st.markdown("### Per-Category Accuracy")
    per_cat = df.groupby('attack_cat').apply(
        lambda g: pd.Series({
            'total':   len(g),
            'correct': (g['attack_cat'] == g['predicted_cat']).sum(),
            'accuracy': (g['attack_cat'] == g['predicted_cat']).mean()
        })
    ).reset_index()

    fig_bar = px.bar(
        per_cat, x='attack_cat', y='accuracy',
        color='attack_cat',
        color_discrete_map=CATEGORY_COLORS,
        title="Classification Accuracy per Category",
        labels={'attack_cat': 'Category', 'accuracy': 'Accuracy'},
        text_auto='.1%',
    )
    fig_bar.update_layout(
        yaxis_range=[0, 1], height=380,
        plot_bgcolor='white', paper_bgcolor='white',
        showlegend=False,
        xaxis_title='Attack Category',
        yaxis_title='Accuracy'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ── Confusion matrix ──────────────────────────────────────────────────────
st.markdown("### Confusion Matrix")

all_cats = sorted(df['attack_cat'].unique())
cm_data  = pd.crosstab(df['attack_cat'], df['predicted_cat'], rownames=['True'], colnames=['Predicted'])

# Ensure all categories appear in both axes
for cat in all_cats:
    if cat not in cm_data.columns:
        cm_data[cat] = 0
    if cat not in cm_data.index:
        cm_data.loc[cat] = 0
cm_data = cm_data.reindex(index=all_cats, columns=all_cats, fill_value=0)

fig_cm = px.imshow(
    cm_data,
    text_auto=True,
    color_continuous_scale='Blues',
    title="Stage 3 Confusion Matrix (True vs Predicted Category)",
    labels=dict(x="Predicted Category", y="True Category", color="Count"),
    aspect='auto',
)
fig_cm.update_layout(height=380, paper_bgcolor='white')
st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")

# ── Category detail cards ─────────────────────────────────────────────────
st.markdown("### Category Details")
cols = st.columns(len(all_cats))

for col_widget, cat in zip(cols, all_cats):
    cat_df  = df[df['attack_cat'] == cat]
    cat_acc = (cat_df['attack_cat'] == cat_df['predicted_cat']).mean()
    count   = len(cat_df)
    emoji   = "🟢" if cat_acc > 0.9 else "🟡" if cat_acc > 0.7 else "🔴"

    with col_widget:
        st.markdown(f"#### {emoji} {cat}")
        st.metric("Windows",  count)
        st.metric("Accuracy", f"{cat_acc:.1%}")

st.markdown("---")

# ── Raw data table ────────────────────────────────────────────────────────
with st.expander("📋 View Raw Classification Results"):
    st.dataframe(df, use_container_width=True, hide_index=True)
