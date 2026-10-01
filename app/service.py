import numpy as np
import pandas as pd


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


class CreditScoringService:
    def __init__(
        self,
        model,
        model_config: dict
    ):
        self.model = model
        self.model_config = model_config

        self.threshold = float(
            model_config['threshold']
        )

    def prepare_input(
        self,
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
        self,
        data: dict
    ) -> list[str]:
        return [
            feature
            for feature in RAW_FEATURES
            if data.get(feature) is None
        ]

    def predict_risk(
        self,
        data: dict
    ) -> dict:
        dataframe = self.prepare_input(
            data
        )

        probability = self.model.predict_proba(
            dataframe
        )[0, 1]

        risk_class = (
            'high'
            if probability >= self.threshold
            else 'low'
        )

        return {
            'default_probability': float(probability),
            'risk_class': risk_class,
            'threshold': self.threshold
        }

    def get_model_info(
        self
    ) -> dict:
        return self.model_config