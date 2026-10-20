import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.models import Model, load_model as load_keras_model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from scripts.config import STAGE1_MODELS, STAGE2_MODELS, STAGE3_MODELS, RANDOM_STATE, MODEL_DIR

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

class Stage1Detector:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = None
    
    def fit(self, X_normal: np.ndarray, scaler: StandardScaler = None):
        if scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_normal)
        else:
            self.scaler = scaler
            X_scaled = self.scaler.transform(X_normal)
        
        self._fit_model(X_scaled)
        return self
    
    def _fit_model(self, X_scaled):
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, name: str):
        if not MODEL_DIR.exists():
            MODEL_DIR.mkdir(parents=True)
        
        if self.scaler:
            joblib.dump(self.scaler, MODEL_DIR / f"{name}_scaler.joblib")
        
        self._save_model(name)
        print(f"Model and scaler saved to {MODEL_DIR}")

    def _save_model(self, name: str):
        raise NotImplementedError

    def load(self, name: str):
        scaler_path = MODEL_DIR / f"{name}_scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        self._load_model(name)
        print(f"Model and scaler loaded from {MODEL_DIR}")

    def _load_model(self, name: str):
        raise NotImplementedError


class IsolationForestDetector(Stage1Detector):
    def _fit_model(self, X_scaled):
        params = STAGE1_MODELS['isolation_forest'].copy()
        self.model = IsolationForest(**params)
        self.model.fit(X_scaled)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
    
    def predict(self, X: np.ndarray, threshold: float = 0.0262) -> np.ndarray:
        scores = self.score(X)
        return (scores < threshold).astype(int)

    def _save_model(self, name: str):
        joblib.dump(self.model, MODEL_DIR / f"{name}_model.joblib")

    def _load_model(self, name: str):
        self.model = joblib.load(MODEL_DIR / f"{name}_model.joblib")


class LOFDetector(Stage1Detector):
    def _fit_model(self, X_scaled):
        params = STAGE1_MODELS['lof'].copy()
        self.model = LocalOutlierFactor(**params)
        self.model.fit(X_scaled)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return -self.model.decision_function(X_scaled)
    
    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        scores = self.score(X)
        # Use median as threshold if None
        if threshold is None:
            threshold = np.median(scores)
        return (scores > threshold).astype(int)

    def _save_model(self, name: str):
        joblib.dump(self.model, MODEL_DIR / f"{name}_model.joblib")

    def _load_model(self, name: str):
        self.model = joblib.load(MODEL_DIR / f"{name}_model.joblib")


class AutoencoderDetector(Stage1Detector): 
    def __init__(self, model_name: str = 'autoencoder'):
        super().__init__(model_name)
        self.encoder = None
        self.reconstruction_threshold = None
    
    def _fit_model(self, X_scaled):
        params = STAGE1_MODELS['autoencoder'].copy()
        reconstruction_percentile = params.pop('reconstruction_percentile')
        input_dim = X_scaled.shape[1]
        encoding_dim = params.pop('encoding_dim')
        
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='linear')(encoded)
        
        self.model = Model(input_layer, decoded)
        self.model.compile(optimizer=Adam(), loss='mse')
        
        self.model.fit(X_scaled, X_scaled, 
                      epochs=params['epochs'],
                      batch_size=params['batch_size'],
                      validation_split=params['validation_split'],
                      verbose=params['verbose'])
        
        train_mse = np.mean((self.model.predict(X_scaled, verbose=0) - X_scaled) ** 2, axis=1)
        self.reconstruction_threshold = np.percentile(train_mse, reconstruction_percentile)
    
    def score(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        mse = np.mean((predictions - X_scaled) ** 2, axis=1)
        return mse
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score(X)
        return (scores > self.reconstruction_threshold).astype(int)

    def _save_model(self, name: str):
        self.model.save(MODEL_DIR / f"{name}_model.keras")
        joblib.dump(self.reconstruction_threshold, MODEL_DIR / f"{name}_threshold.joblib")

    def _load_model(self, name: str):
        self.model = load_keras_model(MODEL_DIR / f"{name}_model.keras")
        self.reconstruction_threshold = joblib.load(MODEL_DIR / f"{name}_threshold.joblib")

class Stage2Classifier:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, scaler: StandardScaler = None):

        if scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_train)
        else:
            self.scaler = scaler
            X_scaled = self.scaler.transform(X_train)
        
        self._fit_model(X_scaled, y_train)
        return self
    
    def _fit_model(self, X_scaled, y_train):
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        return None

    def save(self, name: str):
        if not MODEL_DIR.exists():
            MODEL_DIR.mkdir(parents=True)

        if self.scaler:
            joblib.dump(self.scaler, MODEL_DIR / f"{name}_scaler.joblib")
        
        joblib.dump(self.model, MODEL_DIR / f"{name}_model.joblib")
        print(f"Model saved to {MODEL_DIR}")

    def load(self, name: str):
        scaler_path = MODEL_DIR / f"{name}_scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        self.model = joblib.load(MODEL_DIR / f"{name}_model.joblib")
        print(f"Model loaded from {MODEL_DIR}")


class RandomForestClassifier_Stage2(Stage2Classifier):
    def _fit_model(self, X_scaled, y_train):
        params = STAGE2_MODELS['random_forest'].copy()
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_scaled, y_train)


class SVMClassifier_Stage2(Stage2Classifier):
    def _fit_model(self, X_scaled, y_train):
        params = STAGE2_MODELS['svm'].copy()
        self.model = SVC(**params, probability=True)
        self.model.fit(X_scaled, y_train)


class XGBoostClassifier_Stage2(Stage2Classifier):  
    def _fit_model(self, X_scaled, y_train):
        params = STAGE2_MODELS['xgboost'].copy()
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_scaled, y_train)


def create_stage1_detector(model_name: str) -> Stage1Detector:
    detectors = {
        'isolation_forest': IsolationForestDetector,
        'lof': LOFDetector,
        'autoencoder': AutoencoderDetector,
    }
    if model_name not in detectors:
        raise ValueError(f"Unknown Stage 1 model: {model_name}. Choose from {list(detectors.keys())}")
    return detectors[model_name](model_name)


def create_stage2_classifier(model_name: str) -> Stage2Classifier:
    classifiers = {
        'random_forest': RandomForestClassifier_Stage2,
        'svm': SVMClassifier_Stage2,
        'xgboost': XGBoostClassifier_Stage2,
    }
    if model_name not in classifiers:
        raise ValueError(f"Unknown Stage 2 model: {model_name}. Choose from {list(classifiers.keys())}")
    return classifiers[model_name](model_name)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


class Stage3CategoryClassifier:
    """Multi-class classifier that identifies the attack category (Stage 3)."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.classes_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, scaler: StandardScaler = None):
        if scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_train)
        else:
            self.scaler = scaler
            X_scaled = self.scaler.transform(X_train)

        params = STAGE3_MODELS['random_forest'].copy()
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_scaled, y_train)
        self.classes_ = list(self.model.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, name: str = 'rf_stage3_category'):
        if not MODEL_DIR.exists():
            MODEL_DIR.mkdir(parents=True)
        joblib.dump(self.model, MODEL_DIR / f'{name}.pkl')
        joblib.dump(self.scaler, MODEL_DIR / f'{name}_scaler.pkl')
        joblib.dump(self.classes_, MODEL_DIR / f'{name}_classes.pkl')
        print(f"Stage 3 model saved to {MODEL_DIR}")

    def load(self, name: str = 'rf_stage3_category'):
        self.model = joblib.load(MODEL_DIR / f'{name}.pkl')
        self.scaler = joblib.load(MODEL_DIR / f'{name}_scaler.pkl')
        self.classes_ = joblib.load(MODEL_DIR / f'{name}_classes.pkl')
        print(f"Stage 3 model loaded from {MODEL_DIR}")
