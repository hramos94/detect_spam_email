"""Centralised settings loader using *pydantic* v2.
Automatically reads from environment variables and (if present) a `.env` file
in the project root.

Usage
-----
>>> from app.core.config import settings
>>> settings.openai_api_key
'sk-...'  # pulled from env
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


# Path to project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class _Settings(BaseSettings):

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    environment: str = Field(default="development", alias="ENV") 

    # Pydantic v2 settings
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> _Settings:  # pragma: no cover
    """Return a cached instance so we don't parse env vars repeatedly."""
    try:
        return _Settings()  # type: ignore [call-arg]  # mypy
    except ValidationError as exc:
        # Provide a clear message if mandatory vars are missing
        missing = [e["loc"][0] for e in exc.errors()]
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}") from exc


# Shortcut: module-level singleton for convenience
settings = get_settings()