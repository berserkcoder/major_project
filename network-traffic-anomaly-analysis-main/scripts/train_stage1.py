"""
Train Stage 1 models: Isolation Forest, LOF, Autoencoder
Saves predictions to: dataset/processed/stage1_predictions.csv
"""

import pandas as pd
import numpy as np
import joblib
import json
from scripts.config import PROCESSED_DIR
from scripts.models import create_stage1_detector

df_train = pd.read_csv(PROCESSED_DIR / 'normal_windowed.csv')
df_test = pd.read_csv(PROCESSED_DIR / 'test_windowed_results.csv')

with open(PROCESSED_DIR / 'windowed_columns.json') as f:
    cols = json.load(f)

feature_cols = [c for c in cols if c not in ['window_id', 'window_attack']]

X_train = df_train[feature_cols].values
X_test = df_test[feature_cols].values
y_test = df_test['window_attack'].values

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")

results = {'window_id': df_test['window_id'].values, 'window_attack': y_test}

for model_name in ['isolation_forest', 'lof', 'autoencoder']:
    print(f"\nTraining {model_name}...")
    
    detector = create_stage1_detector(model_name)
    detector.fit(X_train)
    
    score = detector.score(X_test)
    flag = detector.predict(X_test)
    
    results[f'{model_name[:2]}_score'] = score
    results[f'{model_name[:2]}_flag'] = flag
    
    attacks_caught = (flag == 1).sum()
    total_attacks = (y_test == 1).sum()
    
    print(f"Flagged: {(flag==1).sum()}")
    print(f"Recall: {attacks_caught / total_attacks:.3f}")

df_results = pd.DataFrame(results)
output_file = PROCESSED_DIR / 'stage1_predictions.csv'
df_results.to_csv(output_file, index=False)

print(f"\nSaved to {output_file}")
print(f"Shape: {df_results.shape}")
