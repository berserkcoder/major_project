# Network Traffic Anomaly Analysis

A compact anomaly detection project for network traffic using the UNSW-NB15 dataset.
It combines a two-stage machine learning pipeline with an interactive Streamlit dashboard to detect attacks and reduce false positives.

---

## Goals achieved

* Applied window based aggregation on the flow-level dataset
* Implemented Isolation Forest to flag unusual traffic windows
* Achieved 88.6% false alert reduction using Random Forest classifier as stage 2
* Built a Streamlit-based dashboard for analysis and monitoring
* A live packet capetring script is implemented for real-time window creation and demonstration

---

## Project structure

```
network-traffic-anomaly-analysis/
├── config.py
├── Home.py
├── README.md
├── requirements.txt
├── utils.py
|
├── dataset/
│   ├── raw/            # Raw UNSW-NB15 dataset
│   └── processed/      # Stores proceesed data files
|
├── figures/ 
├── models/
|
├── notebooks/
│   ├── 01_data_understanding_and_windowing.ipynb
│   ├── 02_baseline_anomaly_detection.ipynb
│   ├── 03_feature_analysis_and_fp_reduction.ipynb
│   ├── 04_test_set_evaluation.ipynb
│   └── 05_attack_category_analysis.ipynb
|
└── scripts/
    ├── live_monitor.py
    └── simulate_attack.py
```

---

## Quick start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the dashboard:
   ```bash
   streamlit run Home.py
   ```
3. (Optional) Run live capture:
   ```bash
   python scripts/live_monitor.py
   ```

---

## Dataset

This project uses the **UNSW-NB15 dataset**, a popular dataset for intrusion detection research.
The raw CSV files are stored in `dataset/raw/`. The processed data files can be reconstructed by running the notebooks in order.

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Scapy (for live packet capture)

