import time
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scripts.config import PROCESSED_DIR
from utils import load_pipeline, load_data, predict_window

st.set_page_config(page_title="Live Monitor", page_icon="📡", layout="wide")

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #FFFFFF; }
    .stMetric { background-color: #F8F9FA; padding: 10px; border-radius: 8px; border: 1px solid #E9ECEF; }
    h1, h2, h3 { color: #212529; }
</style>
""", unsafe_allow_html=True)

st.title("📡 Live Monitor")
st.markdown("Scoring real network traffic captured from your network interface.")

# Load resources
s1_model, scaler, s2_model, model_threshold, s3_model, s3_scaler, s3_classes = load_pipeline()
_, _, _, expected_cols = load_data()
feature_cols = [col for col in expected_cols if col not in ['window_id', 'window_attack']]

BUFFER_FILE = PROCESSED_DIR / "live_buffer.json"

st.info(
    "Run `python scripts/live_monitor.py` as Administrator before connecting.\n\n"
    "Anomalies will be flagged based on the selected pipeline's logic."
)

start = st.button("Connect to Live Stream", type="primary", use_container_width=True)
st.markdown("---")

m1, m2, m3 = st.columns(3)
m_rate    = m1.empty()
m_flags   = m2.empty()
m_attacks = m3.empty()

chart_col, table_col = st.columns([3, 2])
with chart_col:
    chart_placeholder = st.empty()
with table_col:
    table_placeholder = st.empty()

def read_buffer():
    if not BUFFER_FILE.exists():
        return []
    try:
        with open(BUFFER_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

if start:
    last_id      = -1
    history      = []
    total_flags  = 0
    total_attacks = 0

    while True:
        buffer = read_buffer()

        if buffer and buffer[-1]['window_id'] > last_id:
            new_windows = [w for w in buffer if w['window_id'] > last_id]

            for w in new_windows:
                df_row = pd.DataFrame([w])
                score, threshold, s1, s2, category = predict_window(
                    df_row, feature_cols, scaler, s1_model, s2_model,
                    s3_model, s3_scaler, s3_classes
                )

                if s1 == 1: total_flags   += 1
                if s2 == 1: total_attacks += 1

                if s2 == 1:
                    verdict = f'Attack ({category})' if category else 'Attack (Detected)'
                elif s1 == 1 and s2 == 0:
                    verdict = 'FP Blocked'
                else:
                    verdict = 'Normal'

                history.append({
                    'window_id': w['window_id'],
                    'rate':      round(w.get('raw_rate', 0), 1),
                    'score':     round(score, 4),
                    'stage1':    s1,
                    'final':     s2,
                    'category':  category,
                    'verdict':   verdict
                })

            last_id = buffer[-1]['window_id']
            history = history[-50:]
            df_display = pd.DataFrame(history)

            current_rate = buffer[-1].get('raw_rate', 0)
            m_rate.metric("Traffic Rate",   f"{current_rate:.1f} pkts/s")
            m_flags.metric("Stage 1 Flags", total_flags)
            m_attacks.metric("Attacks Confirmed", total_attacks)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_display['window_id'], y=df_display['score'],
                mode='lines+markers', name='Suspicion Score',
                line=dict(color='#007BFF', width=2),
                marker=dict(size=4)
            ))
            fig.add_hline(
                y=threshold, line_dash='dash', line_color='#FFC107',
                annotation_text='Threshold', annotation_position="top left"
            )
            
            alerts_df = df_display[df_display['final'] == 1]
            if len(alerts_df) > 0:
                fig.add_trace(go.Scatter(
                    x=alerts_df['window_id'], y=alerts_df['score'],
                    mode='markers', name='Attack',
                    marker=dict(color='#DC3545', size=10, symbol='circle')
                ))
            
            fp_df = df_display[(df_display['stage1']==1) & (df_display['final']==0)]
            if len(fp_df) > 0:
                fig.add_trace(go.Scatter(
                    x=fp_df['window_id'], y=fp_df['score'],
                    mode='markers', name='FP Blocked',
                    marker=dict(color='#28A745', size=12, symbol='star')
                ))
            
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation='h', y=-0.2),
                xaxis_title='Time Step (Window ID)',
                yaxis_title='Suspicion Score',
                plot_bgcolor='white',
                paper_bgcolor='white',
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            table_placeholder.dataframe(
                df_display[['window_id', 'rate', 'score', 'category', 'verdict']]
                .tail(15).sort_values('window_id', ascending=False)
                .reset_index(drop=True),
                use_container_width=True, hide_index=True
            )

        time.sleep(1.0)