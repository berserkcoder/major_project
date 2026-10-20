"""
Train Stage 2 models: 3 Stage1 × 3 Stage2 = 9 combinations
Input: dataset/processed/stage1_predictions.csv
Output: dataset/processed/stage2_comparison.csv
"""

import pandas as pd
import numpy as np
import json
from scripts.config import PROCESSED_DIR
from scripts.models import create_stage1_detector, create_stage2_classifier, evaluate_model

df_train_normal = pd.read_csv(PROCESSED_DIR / 'normal_windowed.csv')
df_test = pd.read_csv(PROCESSED_DIR / 'test_windowed_results.csv')

with open(PROCESSED_DIR / 'windowed_columns.json') as f:
    cols = json.load(f)

feature_cols = [c for c in cols if c not in ['window_id', 'window_attack']]

X_train = df_train_normal[feature_cols].values
X_test = df_test[feature_cols].values
y_test = df_test['window_attack'].values

print(f"Data loaded - train: {X_train.shape}, test: {X_test.shape}\n")

stage1_models = ['isolation_forest', 'lof', 'autoencoder']
stage2_models = ['random_forest', 'svm', 'xgboost']

results = []

for s1_name in stage1_models:
    print(f"Stage 1: {s1_name}")
    
    detector = create_stage1_detector(s1_name)
    detector.fit(X_train)
    
    flags_train = detector.predict(X_train)
    flags_test = detector.predict(X_test)
    
    split_idx = len(X_test) // 2

    X_s2_train = X_test[:split_idx][flags_test[:split_idx] == 1]
    y_s2_train = y_test[:split_idx][flags_test[:split_idx] == 1]

    X_s2_test = X_test[split_idx:][flags_test[split_idx:] == 1]
    y_s2_test = y_test[split_idx:][flags_test[split_idx:] == 1]
    
    print(f"Flagged: train={len(X_s2_train)}, test={len(X_s2_test)}")
    
    if len(X_s2_train) == 0 or len(X_s2_test) == 0:
        print(f"Skipping - not enough flagged windows")
        continue
    
    for s2_name in stage2_models:
        classifier = create_stage2_classifier(s2_name)
        classifier.fit(X_s2_train, y_s2_train, scaler=detector.scaler)
    
        y_pred = classifier.predict(X_s2_test)
        metrics = evaluate_model(y_s2_test, y_pred)
        
        results.append({
            'stage1': s1_name,
            'stage2': s2_name,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
        })
        
        print(f" + {s2_name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

df_results = pd.DataFrame(results)
output_file = PROCESSED_DIR / 'stage2_comparison.csv'
df_results.to_csv(output_file, index=False)

print(f"\nSaved to {output_file}")
print(f"\nResults ({len(results)} combos):")
print(df_results)
