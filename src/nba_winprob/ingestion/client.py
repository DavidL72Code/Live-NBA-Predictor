"""Rate-limited wrapper around nba_api's stats endpoints.

stats.nba.com is unauthenticated and unofficial; per the project plan we are
deliberately polite: a minimum interval between requests and bounded retries
with exponential backoff. All nba_api imports are lazy so unit tests and
feature code never need network-capable dependencies loaded.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

DEFAULT_MIN_REQUEST_INTERVAL = 1.0  # seconds between requests
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30


class NBAStatsClient:
    def __init__(
        self,
        min_request_interval: float = DEFAULT_MIN_REQUEST_INTERVAL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.min_request_interval = min_request_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        wait = self.min_request_interval - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _call(self, description: str, make_request):
        """Run one endpoint request with throttling and backoff."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                return make_request()
            except Exception as exc:  # nba_api raises requests errors and JSON errors
                last_error = exc
                backoff = 2**attempt
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %ds",
                    description,
                    attempt,
                    self.max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
        raise RuntimeError(
            f"{description} failed after {self.max_retries} attempts"
        ) from last_error

    def get_season_game_ids(
        self, season: str, season_type: str = "Regular Season"
    ) -> list[str]:
        """All game IDs for a season (e.g. season='2023-24'), in date order.

        LeagueGameLog lists each game once per team; duplicates are removed
        preserving first appearance.
        """
        from nba_api.stats.endpoints import leaguegamelog

        def make_request():
            return leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                timeout=self.timeout,
            ).get_dict()

        payload = self._call(f"LeagueGameLog({season}, {season_type})", make_request)
        result_set = payload["resultSets"][0]
        headers = result_set["headers"]
        game_id_idx = headers.index("GAME_ID")
        seen: dict[str, None] = {}
        for row in result_set["rowSet"]:
            seen.setdefault(str(row[game_id_idx]), None)
        return list(seen)

    def get_play_by_play_raw(self, game_id: str) -> dict:
        """Raw PlayByPlayV3 payload for one game (normalize separately).

        V3, not V2: as of mid-2025 the V2 endpoint returns empty payloads.
        """
        from nba_api.stats.endpoints import playbyplayv3

        def make_request():
            return playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=self.timeout).get_dict()

        return self._call(f"PlayByPlayV3({game_id})", make_request)
