import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(
    BASE_DIR / '.env'
)

MODELS_DIR = BASE_DIR / 'models'

MODEL_PATH = (
    MODELS_DIR
    / 'credit_scoring_pipeline.joblib'
)

MODEL_CONFIG_PATH = (
    MODELS_DIR
    / 'model_config.joblib'
)

API_VERSION = '1.0.0'

INTERNAL_API_KEY = os.getenv(
    'INTERNAL_API_KEY'
)