"""
Train Stage 3: Attack Category Classifier
------------------------------------------
Assigns attack category labels to windowed attack records using
rule-based heuristics derived from known UNSW-NB15 traffic signatures,
then trains a balanced Random Forest multi-class classifier.

Input : dataset/processed/full_windowed.csv  (all 1754 windowed records)
Output: models/rf_stage3_category.pkl
        models/rf_stage3_category_scaler.pkl
        models/rf_stage3_category_classes.pkl
        dataset/processed/stage3_results.csv
"""

import json
import sys
import os

# Allow running as `python scripts/train_stage3.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path

# ── Paths (mirrors scripts/config.py) ─────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
MODEL_DIR     = PROJECT_ROOT / "models"
RANDOM_STATE  = 42

np.random.seed(RANDOM_STATE)


# ── 1. Load windowed data ──────────────────────────────────────────────────
df = pd.read_csv(PROCESSED_DIR / 'full_windowed.csv')

with open(PROCESSED_DIR / 'windowed_columns.json') as f:
    cols = json.load(f)

feature_cols = [c for c in cols if c not in ['window_id', 'window_attack']]

# Keep only confirmed attack windows for Stage 3 training
df_attacks = df[df['window_attack'] == 1].copy().reset_index(drop=True)
print(f"Total windowed records : {len(df)}")
print(f"Attack windows for S3  : {len(df_attacks)}")


# ── 2. Rule-based category labelling ─────────────────────────────────────
# Each rule maps known UNSW-NB15 attack signatures to window-level statistics.
# Priority order: DoS → Fuzzers → Exploits → Generic (catch-all).

def assign_category(row: pd.Series) -> str:
    """Assign an attack category using deterministic feature thresholds."""

    # DoS: flood-style attacks -- very high state_INT (interrupted connections)
    # combined with high packet rate. In this pre-windowed dataset sbytes_mean
    # is log-scaled (~4-8 range) and rate_mean tops out at ~12.
    if (row.get('rate_mean', 0) > 11.0 or
            (row.get('state_INT_mean', 0) > 0.85 and row.get('rate_mean', 0) > 10.0)):
        return 'DoS'

    # Fuzzers: probe random inputs — many different protocols, erratic durations
    if (row.get('proto_nunique', 0) >= 4 or
            (row.get('dur_std', 0) > 1.5 and row.get('proto_nunique', 0) >= 3)):
        return 'Fuzzers'

    # Exploits: target specific services (HTTP, FTP, SSH, SSL)
    exploit_services = (
        row.get('service_http_mean', 0) +
        row.get('service_ftp_mean', 0) +
        row.get('service_ftp-data_mean', 0) +
        row.get('service_ssh_mean', 0) +
        row.get('service_ssl_mean', 0)
    )
    if exploit_services > 0.25:
        return 'Exploits'

    # Generic: everything else (broad signature attacks — e.g., Generic TCP)
    return 'Generic'


df_attacks['attack_cat'] = df_attacks.apply(assign_category, axis=1)

print("\nCategory distribution (rule-based labels):")
print(df_attacks['attack_cat'].value_counts())


# ── 3. Train / Test split ─────────────────────────────────────────────────
X = df_attacks[feature_cols].values
y = df_attacks['attack_cat'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")


# ── 4. Scale features ─────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ── 5. Train Stage 3 classifier ───────────────────────────────────────────
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
clf.fit(X_train_scaled, y_train)
classes = list(clf.classes_)
print(f"\nClasses learned: {classes}")


# ── 6. Evaluate ───────────────────────────────────────────────────────────
y_pred = clf.predict(X_test_scaled)

print("\n-- Stage 3 Classification Report --")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=classes))


# ── 7. Save model files ───────────────────────────────────────────────────
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(clf,     MODEL_DIR / 'rf_stage3_category.pkl')
joblib.dump(scaler,  MODEL_DIR / 'rf_stage3_category_scaler.pkl')
joblib.dump(classes, MODEL_DIR / 'rf_stage3_category_classes.pkl')
print(f"\nStage 3 model files saved to {MODEL_DIR}")


# ── 8. Save per-window results ────────────────────────────────────────────
all_X_scaled = scaler.transform(df_attacks[feature_cols].values)
df_attacks['predicted_cat'] = clf.predict(all_X_scaled)

output_file = PROCESSED_DIR / 'stage3_results.csv'
df_attacks[['window_id', 'window_attack', 'attack_cat', 'predicted_cat']].to_csv(
    output_file, index=False
)
print(f"Stage 3 results saved to {output_file}")
