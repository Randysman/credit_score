from contextlib import asynccontextmanager

import joblib

from fastapi import (
    Depends,
    FastAPI,
    Request
)

from app.config import (
    API_VERSION,
    MODEL_CONFIG_PATH,
    MODEL_PATH
)
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
    CreditScoringService
)


@asynccontextmanager
async def lifespan(
    app: FastAPI
):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f'Model file not found: {MODEL_PATH}'
        )

    if not MODEL_CONFIG_PATH.exists():
        raise RuntimeError(
            f'Model config file not found: '
            f'{MODEL_CONFIG_PATH}'
        )

    try:
        model = joblib.load(
            MODEL_PATH
        )

        model_config = joblib.load(
            MODEL_CONFIG_PATH
        )

    except Exception as error:
        raise RuntimeError(
            'Failed to load model artifacts'
        ) from error

    app.state.scoring_service = (
        CreditScoringService(
            model=model,
            model_config=model_config
        )
    )

    yield

    app.state.scoring_service = None


app = FastAPI(
    title='Credit Scoring API',
    description=(
        'API for estimating the probability '
        'of serious credit delinquency.'
    ),
    version=API_VERSION,
    lifespan=lifespan
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
    application: QuickCreditApplication,
    request: Request
) -> QuickPredictionResponse:
    data = application.model_dump(
        by_alias=True,
        exclude_none=True
    )

    service = (
        request.app.state.scoring_service
    )

    prediction = service.predict_risk(
        data
    )

    missing_features = (
        service.get_missing_features(
            data
        )
    )

    return QuickPredictionResponse(
        **prediction,
        is_preliminary=True,
        provided_features=(
            len(RAW_FEATURES)
            - len(missing_features)
        ),
        total_features=len(
            RAW_FEATURES
        ),
        missing_features=missing_features
    )


@app.post(
    '/predict',
    response_model=PredictionResponse,
    summary='Full credit risk estimation'
)
def predict(
    application: CreditApplication,
    request: Request
) -> PredictionResponse:
    data = application.model_dump(
        by_alias=True
    )

    service = (
        request.app.state.scoring_service
    )

    prediction = service.predict_risk(
        data
    )

    return PredictionResponse(
        **prediction
    )


@app.get(
    '/internal/model-info',
    response_model=ModelInfoResponse,
    dependencies=[
        Depends(
            verify_internal_api_key
        )
    ],
    summary='Model information'
)
def model_info(
    request: Request
) -> ModelInfoResponse:
    service = (
        request.app.state.scoring_service
    )

    info = service.get_model_info()

    return ModelInfoResponse(
        **info
    )