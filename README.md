# Credit Scoring

A machine learning service for estimating the probability of serious credit delinquency, built with **CatBoost**, **Scikit-learn**, **FastAPI**, and **Docker**.

## Features

- Credit risk prediction
- Preliminary scoring with partial input data
- Full scoring with complete borrower data
- Data preprocessing pipeline
- CatBoost classification model
- Probability calibration analysis
- Custom classification threshold
- FastAPI REST API
- Pydantic validation
- Protected internal model information endpoint
- Liveness and readiness checks
- Docker support

## Tech Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- XGBoost
- LightGBM
- Jupyter

### Backend

- FastAPI
- Pydantic
- Joblib
- Python Dotenv

### DevOps

- Docker
- Git

## Model

The final model is a tuned **CatBoostClassifier**.

Validation metrics:

- ROC-AUC: `0.8671`
- PR-AUC: `0.4072`
- Brier Score: `0.0485`
- Classification threshold: `0.175`

The model estimates the probability of serious credit delinquency and does not make a final lending decision.

## API

Available endpoints:

- `GET /`
- `GET /live`
- `GET /health`
- `POST /predict/quick`
- `POST /predict`
- `GET /internal/model-info`

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Installation
```bash
git clone https://github.com/Randysman/credit_score.git
cd credit_score

python -m venv venv
pip install -r requirements-api.txt

uvicorn app.main:app --reload
```

## Docker
```bash
docker build -t credit-scoring-api .
docker run -p 8000:8000 -e INTERNAL_API_KEY=your-secret-key credit-scoring-api
```

Visit **http://127.0.0.1:8000/docs**

## Author

**Danil**

GitHub: https://github.com/Randysman