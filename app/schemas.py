from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QuickCreditApplication(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra='forbid'
    )

    age: int = Field(
        ge=18,
        le=120,
        description='Age of the borrower'
    )

    MonthlyIncome: float | None = Field(
        default=None,
        ge=0,
        description='Monthly income'
    )

    DebtRatio: float | None = Field(
        default=None,
        ge=0,
        description='Debt ratio'
    )

    RevolvingUtilizationOfUnsecuredLines: float | None = Field(
        default=None,
        ge=0,
        description='Utilization of unsecured credit lines'
    )


class CreditApplication(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra='forbid'
    )

    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ge=0,
        description='Utilization of unsecured credit lines'
    )

    age: int = Field(
        ge=18,
        le=120,
        description='Age of the borrower'
    )

    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        alias='NumberOfTime30-59DaysPastDueNotWorse',
        ge=0,
        description='Number of 30-59 day past due events'
    )

    DebtRatio: float = Field(
        ge=0,
        description='Debt ratio'
    )

    MonthlyIncome: float = Field(
        ge=0,
        description='Monthly income'
    )

    NumberOfOpenCreditLinesAndLoans: int = Field(
        ge=0,
        description='Number of open credit lines and loans'
    )

    NumberOfTimes90DaysLate: int = Field(
        ge=0,
        description='Number of 90+ day late events'
    )

    NumberRealEstateLoansOrLines: int = Field(
        ge=0,
        description='Number of real estate loans or credit lines'
    )

    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        alias='NumberOfTime60-89DaysPastDueNotWorse',
        ge=0,
        description='Number of 60-89 day past due events'
    )

    NumberOfDependents: int = Field(
        ge=0,
        description='Number of dependents'
    )


class PredictionResponse(BaseModel):
    default_probability: float
    risk_class: Literal['low', 'high']
    threshold: float


class QuickPredictionResponse(PredictionResponse):
    is_preliminary: bool = True
    provided_features: int
    total_features: int
    missing_features: list[str]