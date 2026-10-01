from fastapi import FastAPI

from app.schemas import (
    CreditApplication,
    PredictionResponse,
    QuickCreditApplication,
    QuickPredictionResponse
)
from app.service import predict_risk


app = FastAPI(
    title='Credit Scoring API',
    description=(
        'API for estimating of credis score '
    ),
    version='1.0.0'
)


@app.get('/health')
def health() -> dict:
    return {
        'status': 'ok'
    }


@app.post(
    '/predict/quick',
    response_model=QuickPredictionResponse
)
def predict_quick(
    application: QuickCreditApplication
) -> QuickPredictionResponse:
    data = application.model_dump(
        by_alias=True,
        exclude_none=True
    )

    prediction = predict_risk(data)

    return QuickPredictionResponse(
        **prediction,
        is_preliminary=True
    )


@app.post(
    '/predict',
    response_model=PredictionResponse
)
def predict(
    application: CreditApplication
) -> PredictionResponse:
    data = application.model_dump(
        by_alias=True
    )

    prediction = predict_risk(data)

    return PredictionResponse(
        **prediction
    )