"""Redis online feature store.

Stores the *current* FeatureVector for each live game. The serving layer
reads from here at prediction time — must be sub-millisecond.

Key schema:  ``feature:<game_id>``  →  JSON-encoded FeatureVector
TTL:         8 hours (covers any NBA game + overtime; auto-expires stale keys)
"""

from __future__ import annotations

import logging

from nba_winprob.schemas import FeatureVector

logger = logging.getLogger(__name__)

_KEY_PREFIX = "feature:"
_TTL_SECONDS = 8 * 3600


class OnlineStore:
    """Thin wrapper around a Redis connection.

    Lazy import of ``redis`` so the rest of the codebase (features, ingestion,
    tests) never requires the Redis client to be installed.
    """

    def __init__(self, redis_url: str | None = None):
        from nba_winprob.config import get_settings

        self._url = redis_url or get_settings().redis_url

    def _client(self):
        import redis

        return redis.from_url(self._url, decode_responses=True)

    def write(self, feature: FeatureVector) -> None:
        """Upsert the current feature vector for a game."""
        client = self._client()
        key = f"{_KEY_PREFIX}{feature.game_id}"
        client.setex(key, _TTL_SECONDS, feature.model_dump_json())

    def read(self, game_id: str) -> FeatureVector | None:
        """Return the latest feature vector for a game, or None if not found."""
        client = self._client()
        raw = client.get(f"{_KEY_PREFIX}{game_id}")
        if raw is None:
            return None
        return FeatureVector.model_validate_json(raw)

    def delete(self, game_id: str) -> None:
        self._client().delete(f"{_KEY_PREFIX}{game_id}")

    def ping(self) -> bool:
        """Return True if the Redis connection is healthy."""
        try:
            return self._client().ping()
        except Exception:
            return False
