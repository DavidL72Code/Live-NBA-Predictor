from pathlib import Path

from nba_winprob.config import Settings


def test_defaults_are_local_dev(monkeypatch):
    monkeypatch.delenv("NBA_WINPROB_RAW_DATA_DIR", raising=False)
    settings = Settings(_env_file=None)
    assert settings.raw_data_dir == Path("data/raw")
    assert settings.min_request_interval == 1.0
    assert settings.postgres_dsn is None  # no credentials baked in anywhere


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("NBA_WINPROB_MIN_REQUEST_INTERVAL", "2.5")
    monkeypatch.setenv("NBA_WINPROB_REDIS_URL", "redis://example:6380/1")
    settings = Settings(_env_file=None)
    assert settings.min_request_interval == 2.5
    assert settings.redis_url == "redis://example:6380/1"
