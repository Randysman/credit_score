import secrets

from fastapi import Header, HTTPException, status

from app.config import INTERNAL_API_KEY


def verify_internal_api_key(
    x_api_key: str = Header(...)
) -> None:
    if INTERNAL_API_KEY is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Internal API key is not configured'
        )

    if not secrets.compare_digest(
        x_api_key,
        INTERNAL_API_KEY
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid API key'
        )