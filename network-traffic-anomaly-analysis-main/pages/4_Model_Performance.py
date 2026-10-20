"""
Model Performance page: detailed metrics and evaluation results for all 3 stages.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from scripts.config import PROCESSED_DIR

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.markdown("""
<style>
    .main { background-color: #FFFFFF; }
    .stMetric { background-color: #F8F9FA; padding: 10px; border-radius: 8px; border: 1px solid #E9ECEF; }
    h1, h2, h3 { color: #212529; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Model Performance Metrics")
st.markdown("Detailed performance evaluation of the three-stage detection pipeline.")

tab1, tab2, tab3 = st.tabs(["Overall Performance", "Category Breakdown", "Stage 3 — Attack Classifier"])

with tab1:
    st.markdown("### Isolation Forest + Random Forest Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stage 1 Recall", "~97%")
    with col2:
        st.metric("FP Reduction", "88.6%")
    with col3:
        st.metric("Final Recall", "~97")
    with col4:
        st.metric("Final FP Rate", "<3%")

    st.markdown("---")

    eval_file = PROCESSED_DIR / 'eval_results.csv'
    if eval_file.exists():
        st.markdown("### Evaluation Results")
        df_eval = pd.read_csv(eval_file)
        st.dataframe(df_eval, use_container_width=True)
    else:
        st.info("`eval_results.csv` not found.")

with tab2:
    st.markdown("### Attack Category Detection Rates (Stage 1 + 2 pipeline)")

    attack_file = PROCESSED_DIR / 'attack_category_results.csv'
    if attack_file.exists():
        df_attack = pd.read_csv(attack_file)

        st.dataframe(df_attack, use_container_width=True)
        col1, col2 = st.columns(2)

        with col1:
            try:
                plot_df = df_attack.copy()
                if 'dominant_cat' in plot_df.columns:
                    plot_df = plot_df.rename(columns={'dominant_cat': 'category'})

                available_metrics = [c for c in ['detection_rate', 'precision', 'recall', 'f1_score'] if c in plot_df.columns]

                fig = px.bar(
                    plot_df,
                    x='category' if 'category' in plot_df.columns else plot_df.columns[0],
                    y=available_metrics if available_metrics else plot_df.columns[1:],
                    barmode='group',
                    title='Detection Metrics by Category',
                    labels={'value': 'Score', 'variable': 'Metric'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render chart: {e}")

        with col2:
            st.markdown("### Difficulty Ranking")
            sort_col = 'detection_rate' if 'detection_rate' in df_attack.columns else (
                'recall' if 'recall' in df_attack.columns else df_attack.columns[0]
            )
            df_sorted = df_attack.sort_values(sort_col, ascending=False)

            for idx, row in df_sorted.head(5).iterrows():
                metric = row.get('detection_rate', row.get('recall', row.get('f1_score', 0)))
                if metric > 1.0:
                    display_metric = metric / 100.0
                    metric_val = metric
                else:
                    display_metric = metric
                    metric_val = metric * 100.0

                emoji = "🟢" if display_metric > 0.9 else "🟡" if display_metric > 0.7 else "🔴"
                category = row.get('dominant_cat', row.get('attack_category', 'Unknown'))

                st.metric(f"{emoji} {category}", f"{display_metric:.1%}")
    else:
        st.info("`attack_category_results.csv` not found")

with tab3:
    st.markdown("### Stage 3: Attack Category Classifier Performance")

    s3_file = PROCESSED_DIR / 'stage3_results.csv'
    if s3_file.exists():
        df_s3 = pd.read_csv(s3_file)

        # Summary metrics
        correct = (df_s3['attack_cat'] == df_s3['predicted_cat']).sum()
        total   = len(df_s3)
        acc     = correct / total if total > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Windows Classified", total)
        col2.metric("Correctly Labelled", correct)
        col3.metric("Overall Accuracy", f"{acc:.1%}")

        st.markdown("---")

        # Per-category breakdown
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Predicted Category Distribution")
            cat_dist = df_s3['predicted_cat'].value_counts().reset_index()
            cat_dist.columns = ['Category', 'Count']
            fig_pie = px.pie(
                cat_dist, names='Category', values='Count',
                hole=0.4, title="Stage 3 Predictions",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_pie.update_layout(height=350, paper_bgcolor='white')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            st.markdown("#### True vs Predicted Labels")
            st.dataframe(
                df_s3[['window_id', 'attack_cat', 'predicted_cat']].head(30),
                use_container_width=True, hide_index=True
            )

        st.markdown("---")
        st.markdown("#### Per-Category Accuracy")

        per_cat = df_s3.groupby('attack_cat').apply(
            lambda g: (g['attack_cat'] == g['predicted_cat']).mean()
        ).reset_index()
        per_cat.columns = ['Category', 'Accuracy']

        fig_bar = px.bar(
            per_cat, x='Category', y='Accuracy',
            color='Category',
            title="Per-Category Classification Accuracy",
            color_discrete_sequence=px.colors.qualitative.Bold,
            text_auto='.1%'
        )
        fig_bar.update_layout(
            yaxis_range=[0, 1], height=350,
            plot_bgcolor='white', paper_bgcolor='white',
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info("""
        Stage 3 model has not been trained yet.

        Run the following command from the project root to train it:
        ```bash
        python scripts/train_stage3.py
        ```
        """)