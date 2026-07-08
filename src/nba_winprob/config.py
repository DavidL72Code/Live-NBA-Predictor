"""Central runtime configuration.

Every setting comes from the environment (or a local ``.env`` file), prefixed
``NBA_WINPROB_``. Nothing security-sensitive is ever hardcoded: connection
strings and any future credentials belong in ``.env`` (gitignored) — see
``.env.example`` for the full list of knobs. Code should take config values
from ``get_settings()``, never from literals.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NBA_WINPROB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Phase 1: ingestion
    raw_data_dir: Path = Path("data/raw")
    min_request_interval: float = 1.0

    # Phase 2+: infrastructure (defaults point at local dev services;
    # real deployments override via environment/.env)
    kafka_bootstrap_servers: str = "localhost:9092"
    redis_url: str = "redis://localhost:6379/0"
    postgres_dsn: str | None = None
    mlflow_tracking_uri: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
