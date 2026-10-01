import joblib
import numpy as np
import pandas as pd

from app.config import (
    MODEL_CONFIG_PATH,
    MODEL_PATH
)


model = joblib.load(
    MODEL_PATH
)

model_config = joblib.load(
    MODEL_CONFIG_PATH
)

THRESHOLD = float(
    model_config['threshold']
)


RAW_FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents'
]


def prepare_input(
    data: dict
    ) -> pd.DataFrame:
    row = {}

    for feature in RAW_FEATURES:
        value = data.get(
            feature,
            np.nan
        )

        row[feature] = (
            np.nan
            if value is None
            else value
        )

    return pd.DataFrame([row])


def get_missing_features(
    data: dict
    ) -> list[str]:
    return [
        feature
        for feature in RAW_FEATURES
        if data.get(feature) is None
    ]


def predict_risk(
    data: dict
    ) -> dict:
    dataframe = prepare_input(
        data
    )

    probability = model.predict_proba(
        dataframe
    )[0, 1]

    risk_class = (
        'high'
        if probability >= THRESHOLD
        else 'low'
    )

    return {
        'default_probability': float(probability),
        'risk_class': risk_class,
        'threshold': THRESHOLD
    }


def get_model_info() -> dict:
    return model_config