from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

from catboost import CatBoostClassifier

from src.preprocessing import CreditPreprocessor


RANDOM_STATE = 42
TARGET = 'SeriousDlqin2yrs'
FINAL_THRESHOLD = 0.175

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

df = pd.read_csv(
    DATA_DIR/'train.csv',
    index_col='Id'
)

X = df.drop(
    columns=TARGET
)

y = df[TARGET]


X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)


catboost_model = CatBoostClassifier(
    random_strength=2.0,
    learning_rate=0.02,
    l2_leaf_reg=5.0,
    iterations=500,
    depth=6,
    bagging_temperature=1.0,
    loss_function='Logloss',
    random_seed=RANDOM_STATE,
    verbose=False,
    allow_writing_files=False
)


final_pipeline = Pipeline([
    (
        'preprocessor',
        CreditPreprocessor()
    ),
    (
        'model',
        catboost_model
    )
])


final_pipeline.fit(
    X_train,
    y_train
)


valid_proba = final_pipeline.predict_proba(
    X_valid
)[:, 1]


roc_auc = roc_auc_score(
    y_valid,
    valid_proba
)

pr_auc = average_precision_score(
    y_valid,
    valid_proba
)

brier = brier_score_loss(
    y_valid,
    valid_proba
)


print(f'ROC-AUC: {roc_auc:.4f}')
print(f'PR-AUC: {pr_auc:.4f}')
print(f'Brier Score: {brier:.4f}')


MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


joblib.dump(
    final_pipeline,
    MODELS_DIR / 'credit_scoring_pipeline.joblib'
)


model_config = {
    'threshold': FINAL_THRESHOLD,
    'target': TARGET,
    'positive_class': 1
}

joblib.dump(
    model_config,
    MODELS_DIR / 'model_config.joblib'
)


print('\nModel artifacts saved successfully.')