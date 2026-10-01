from pydantic import BaseModel, ConfigDict, Field


class QuickCreditApplication(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True
    )

    age: int = Field(
        ge=18,
        le=120
    )

    MonthlyIncome: float | None = Field(
        default=None,
        ge=0
    )

    DebtRatio: float | None = Field(
        default=None,
        ge=0
    )

    RevolvingUtilizationOfUnsecuredLines: float | None = Field(
        default=None,
        ge=0
    )


class CreditApplication(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True
    )

    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ge=0
    )

    age: int = Field(
        ge=18,
        le=120
    )

    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        alias='NumberOfTime30-59DaysPastDueNotWorse',
        ge=0
    )

    DebtRatio: float = Field(
        ge=0
    )

    MonthlyIncome: float = Field(
        ge=0
    )

    NumberOfOpenCreditLinesAndLoans: int = Field(
        ge=0
    )

    NumberOfTimes90DaysLate: int = Field(
        ge=0
    )

    NumberRealEstateLoansOrLines: int = Field(
        ge=0
    )

    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        alias='NumberOfTime60-89DaysPastDueNotWorse',
        ge=0
    )
    NumberOfDependents: int = Field(
        ge=0
    )


class PredictionResponse(BaseModel):
    default_probability: float
    risk_class: str
    threshold: float


class QuickPredictionResponse(PredictionResponse):
    is_preliminary: bool = True