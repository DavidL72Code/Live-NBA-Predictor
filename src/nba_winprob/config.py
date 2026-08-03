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

    # Phase 2: streaming
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_events_topic: str = "nba.game.events"
    kafka_features_topic: str = "nba.game.features"
    kafka_consumer_group: str = "nba-winprob-processor"
    # How often the producer polls NBA live endpoints (seconds)
    live_poll_interval: float = 15.0

    # Phase 3+: feature store / ML (defaults point at local dev services)
    redis_url: str = "redis://localhost:6379/0"
    postgres_dsn: str | None = None
    mlflow_tracking_uri: str | None = None

    # Phase 4: LLM analyst
    gemini_api_key: str | None = None
    analyst_model: str = "gemini-3.1-flash-lite"
    analyst_mlflow_run_id: str | None = None

    # Web deployment
    # Comma-separated browser origins allowed to call the backend API.
    cors_allowed_origins: str = "http://127.0.0.1:8765,http://localhost:8765,null"
    # Optional same-schema proxy for NBA Stats requests from cloud hosts.
    stats_proxy_url: str | None = None
    stats_proxy_token: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
