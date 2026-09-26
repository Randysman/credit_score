import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class CreditPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.late_payment_features = [
            'NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate'
        ]

        self.log_features = [
            'RevolvingUtilizationOfUnsecuredLines',
            'DebtRatio',
            'MonthlyIncome'
        ]

        self.output_features = [
            'RevolvingUtilizationOfUnsecuredLines_log',
            'age',
            'NumberOfTime30-59DaysPastDueNotWorse',
            'DebtRatio_log',
            'MonthlyIncome_log',
            'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate',
            'NumberRealEstateLoansOrLines',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfDependents',
            'HasUnknownDelinquencyHistory'
        ]

        self.imputer = SimpleImputer(
            strategy='median'
        )

    def fit(self, X, y=None):
        X = self._prepare_special_values(X)

        self.imputer.fit(X)

        return self

    def transform(self, X):
        X = self._prepare_special_values(X)

        transformed = self.imputer.transform(X)

        X = pd.DataFrame(
            transformed,
            columns=X.columns,
            index=X.index
        )

        for feature in self.log_features:
            X[f'{feature}_log'] = np.log1p(
                X[feature]
            )

        return X[self.output_features]

    def _prepare_special_values(self, X):
        X = X.copy()

        X.loc[
            X['age'] == 0,
            'age'
        ] = np.nan

        X['HasUnknownDelinquencyHistory'] = (
            X[self.late_payment_features]
            .eq(98)
            .all(axis=1)
            .astype(int)
        )

        for feature in self.late_payment_features:
            X.loc[
                X[feature] == 98,
                feature
            ] = np.nan

        return X