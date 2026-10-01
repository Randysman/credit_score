from fastapi import FastAPI

from app.schemas import (
    CreditApplication,
    PredictionResponse,
    QuickCreditApplication,
    QuickPredictionResponse
)
from app.service import (
    RAW_FEATURES,
    get_missing_features,
    predict_risk
)


app = FastAPI(
    title='Credit Scoring API',
    description=(
        'API for estimating the probability '
        'of serious credit delinquency.'
    ),
    version='1.0.0'
)


@app.get(
    '/',
    summary='API information'
)
def root() -> dict:
    return {
        'service': 'Credit Scoring API',
        'version': '1.0.0',
        'docs': '/docs'
    }


@app.get(
    '/health',
    summary='Health check'
)
def health() -> dict:
    return {
        'status': 'ok'
    }


@app.post(
    '/predict/quick',
    response_model=QuickPredictionResponse,
    summary='Preliminary credit risk estimation'
)
def predict_quick(
    application: QuickCreditApplication
) -> QuickPredictionResponse:
    data = application.model_dump(
        by_alias=True,
        exclude_none=True
    )

    prediction = predict_risk(
        data
    )

    missing_features = get_missing_features(
        data
    )

    return QuickPredictionResponse(
        **prediction,
        is_preliminary=True,
        provided_features=(
            len(RAW_FEATURES)
            - len(missing_features)
        ),
        total_features=len(RAW_FEATURES),
        missing_features=missing_features
    )


@app.post(
    '/predict',
    response_model=PredictionResponse,
    summary='Full credit risk estimation'
)
def predict(
    application: CreditApplication
) -> PredictionResponse:
    data = application.model_dump(
        by_alias=True
    )

    prediction = predict_risk(
        data
    )

    return PredictionResponse(
        **prediction
    )