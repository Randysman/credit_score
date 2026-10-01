from fastapi import Depends, FastAPI

from app.config import API_VERSION
from app.schemas import (
    CreditApplication,
    ModelInfoResponse,
    PredictionResponse,
    QuickCreditApplication,
    QuickPredictionResponse
)
from app.security import verify_internal_api_key
from app.service import (
    RAW_FEATURES,
    get_missing_features,
    get_model_info,
    predict_risk
)


app = FastAPI(
    title='Credit Scoring API',
    description=(
        'API for estimating the probability '
        'of serious credit delinquency.'
    ),
    version=API_VERSION
)


@app.get(
    '/',
    summary='API information'
)
def root() -> dict:
    return {
        'service': 'Credit Scoring API',
        'version': API_VERSION,
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


@app.get(
    '/internal/model-info',
    response_model=ModelInfoResponse,
    dependencies=[
        Depends(verify_internal_api_key)
    ],
    summary='Model information'
)
def model_info() -> ModelInfoResponse:
    info = get_model_info()

    return ModelInfoResponse(
        **info
    )