from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "dataset/raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset/processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIG_DIR = PROJECT_ROOT / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

WINDOW_SIZE = 100
RANDOM_STATE = 42
BEST_THRESHOLD = 0.0262

STAGE1_MODELS = {
    'isolation_forest': {
        'n_estimators': 100,
        'contamination': 0.02,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    },
    'lof': {
        'n_neighbors': 20,
        'contamination': 0.02,
        'novelty': True,
        'n_jobs': -1,
    },
    'autoencoder': {
        'encoding_dim': 16,
        'epochs': 50,
        'batch_size': 32,
        'validation_split': 0.1,
        'verbose': 0,
        'reconstruction_percentile': 90,
    },
}

STAGE2_MODELS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    },
    'svm': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'scale_pos_weight': 1.0,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    },
}

STAGE3_MODELS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 12,
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    },
}

TOP_FEATURES = [
    'state_INT_mean',
    'proto_nunique',
    'sttl_mean',
    'sbytes_mean',
    'dbytes_mean',
    'service_dns_mean',
    'rate_mean',
]

ATTACK_CATEGORIES = [
    'Analysis', 'Backdoor', 'DoS', 'Exploit', 'Fuzzers',
    'Generic', 'Reconnaissance', 'Shellcode', 'Worms',
]

# Categories confirmed present in test set (used by Stage 3)
ATTACK_CATEGORIES_STAGE3 = ['DoS', 'Exploits', 'Fuzzers', 'Generic']

CATEGORY_COLORS = {
    'DoS':     '#DC3545',   # red
    'Exploits':'#FD7E14',   # orange
    'Fuzzers': '#6F42C1',   # purple
    'Generic': '#0D6EFD',   # blue
}