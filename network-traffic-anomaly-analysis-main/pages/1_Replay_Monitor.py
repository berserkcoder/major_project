import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import load_pipeline, load_data, predict_window
from scripts.config import CATEGORY_COLORS

st.set_page_config(page_title="Replay Monitor", page_icon="📈", layout="wide")

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #FFFFFF; }
    .stMetric { background-color: #F8F9FA; padding: 10px; border-radius: 8px; border: 1px solid #E9ECEF; }
    h1, h2, h3 { color: #212529; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Replay Monitor")
st.markdown("Replaying UNSW-NB15 test set through the selected two-stage pipeline.")

# Load resources
s1_model, scaler, s2_model, model_threshold, s3_model, s3_scaler, s3_classes = load_pipeline()
df_windowed, _, _, expected_cols = load_data()
feature_cols = [col for col in expected_cols if col not in ['window_id', 'window_attack']]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Windows",  len(df_windowed))
col2.metric("Actual Attacks", int(df_windowed['window_attack'].sum()))
s1_placeholder = col3.empty()
s2_placeholder = col4.empty()

st.markdown("---")

c1, c2, c3 = st.columns([1, 1, 2])
speed     = c1.slider("Replay Speed", 1, 40, 15)
n_display = c2.slider("History Window", 20, 100, 50)
start     = c3.button("Start Replay Simulation", type="primary", use_container_width=True)

st.markdown("---")

chart_col, right_col = st.columns([3, 2])
with chart_col:
    chart_placeholder = st.empty()
with right_col:
    table_placeholder = st.empty()
    pie_placeholder   = st.empty()

status_placeholder = st.empty()

if start:
    history       = []
    s1_flags      = 0
    s2_alerts     = 0
    cat_counts    = {cat: 0 for cat in (s3_classes or ['DoS', 'Exploits', 'Fuzzers', 'Generic'])}

    for i, row in df_windowed.iterrows():
        window_row = df_windowed.iloc[[i]]
        score, threshold, s1, s2, category = predict_window(
            window_row, feature_cols, scaler, s1_model, s2_model,
            s3_model, s3_scaler, s3_classes
        )

        actual = int(row['window_attack'])

        # Determine verdict for display
        if s2 == 1 and actual == 1:
            verdict = f'Attack ({category})' if category else 'Attack (Detected)'
            s2_alerts += 1
            s1_flags  += 1
            if category in cat_counts:
                cat_counts[category] += 1
        elif s2 == 0 and actual == 0:
            verdict = 'Normal'
        elif s1 == 1 and s2 == 0:
            verdict = 'False Alarm (Blocked)'
            s1_flags += 1
        elif s1 == 0 and actual == 1:
            verdict = 'Missed'
        else:
            verdict = 'Normal'

        history.append({
            'window_id': int(row['window_id']),
            'score':     round(score, 4),
            'stage1':    s1,
            'final':     s2,
            'actual':    actual,
            'category':  category,
            'verdict':   verdict
        })

        display    = history[-n_display:]
        df_display = pd.DataFrame(display)

        # ── Plotly suspicion-score chart ──────────────────────────────
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_display['window_id'], y=df_display['score'],
            mode='lines', name='Suspicion Score',
            line=dict(color='#007BFF', width=2)
        ))
        fig.add_hline(
            y=threshold, line_dash='dash', line_color='#FFC107',
            annotation_text='Threshold', annotation_position="top left"
        )

        # Mark confirmed attacks — colour by category
        for cat, col_hex in CATEGORY_COLORS.items():
            cat_alerts = df_display[(df_display['final'] == 1) & (df_display['category'] == cat)]
            if len(cat_alerts) > 0:
                fig.add_trace(go.Scatter(
                    x=cat_alerts['window_id'], y=cat_alerts['score'],
                    mode='markers', name=cat,
                    marker=dict(color=col_hex, size=10, symbol='circle')
                ))

        # Mark unclassified alerts (stage 3 not available)
        uncat_alerts = df_display[(df_display['final'] == 1) & (df_display['category'] == '')]
        if len(uncat_alerts) > 0:
            fig.add_trace(go.Scatter(
                x=uncat_alerts['window_id'], y=uncat_alerts['score'],
                mode='markers', name='Attack',
                marker=dict(color='#DC3545', size=10, symbol='circle')
            ))

        # Mark FP Reductions
        fp_red_df = df_display[(df_display['stage1'] == 1) & (df_display['final'] == 0)]
        fig.add_trace(go.Scatter(
            x=fp_red_df['window_id'], y=fp_red_df['score'],
            mode='markers', name='FP Blocked',
            marker=dict(color='#28A745', size=12, symbol='star')
        ))

        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation='h', y=-0.2),
            xaxis_title='Window ID',
            yaxis_title='Suspicion Score',
            plot_bgcolor='white',
            paper_bgcolor='white',
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        # ── Result table ──────────────────────────────────────────────
        display_cols = ['window_id', 'score', 'category', 'verdict']
        table_placeholder.dataframe(
            df_display[display_cols]
            .tail(15).sort_values('window_id', ascending=False)
            .reset_index(drop=True),
            use_container_width=True, hide_index=True
        )

        # ── Live category pie chart ───────────────────────────────────
        active_cats = {k: v for k, v in cat_counts.items() if v > 0}
        if active_cats:
            pie_fig = px.pie(
                names=list(active_cats.keys()),
                values=list(active_cats.values()),
                color=list(active_cats.keys()),
                color_discrete_map=CATEGORY_COLORS,
                title="Attack Category Breakdown",
                hole=0.4,
            )
            pie_fig.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=True,
                paper_bgcolor='white',
            )
            pie_placeholder.plotly_chart(pie_fig, use_container_width=True)

        # ── Update metrics ────────────────────────────────────────────
        s1_placeholder.metric("Stage 1 Flags", s1_flags)
        s2_placeholder.metric("Final Alerts",  s2_alerts)

        time.sleep(1 / speed)

    status_placeholder.success(
        f"Replay complete! Detected {s2_alerts} attacks and blocked {s1_flags - s2_alerts} false alarms."
    )
