"""Live game producer.

Polls the NBA CDN live scoreboard for games in progress, then continuously
fetches new play-by-play actions for each live game and publishes them to
the ``nba.game.events`` Kafka topic.

Design:
- One ``kafka-python`` KafkaProducer shared across all games.
- Per-game cursor: tracks the highest ``actionNumber`` already published so
  each poll only sends genuinely new events.
- Resilient: CDN returns empty/blocked responses outside game windows; those
  are logged at DEBUG and retried on the next poll cycle.
- Polite: a configurable poll interval (default 15s from config) avoids
  hammering NBA CDN.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from nba_winprob.config import get_settings
from nba_winprob.ingestion.normalize import SchemaDriftError, normalize_playbyplay
from nba_winprob.streaming.serde import serialize_event

logger = logging.getLogger(__name__)


@dataclass
class _GameCursor:
    game_id: str
    last_action_num: int = -1


class LiveProducer:
    """Polls NBA live endpoints and publishes new GameEvents to Kafka.

    Usage::

        producer = LiveProducer()
        producer.run()   # blocks; Ctrl-C to stop
    """

    def __init__(self, poll_interval: float | None = None):
        settings = get_settings()
        self._bootstrap = settings.kafka_bootstrap_servers
        self._topic = settings.kafka_events_topic
        self._poll_interval = poll_interval or settings.live_poll_interval
        self._cursors: dict[str, _GameCursor] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        from kafka import KafkaProducer  # lazy import: kafka-python not needed for tests

        producer = KafkaProducer(bootstrap_servers=self._bootstrap)
        logger.info(
            "producer connected to %s; polling every %.0fs",
            self._bootstrap, self._poll_interval,
        )
        try:
            while True:
                self._poll_once(producer)
                time.sleep(self._poll_interval)
        except KeyboardInterrupt:
            logger.info("producer shutting down")
        finally:
            producer.flush()
            producer.close()

    # ------------------------------------------------------------------
    # Internal helpers (testable without a real Kafka connection)
    # ------------------------------------------------------------------

    def poll_new_events(self, game_id: str, raw_pbp: dict) -> list:
        """Given a raw PlayByPlayV3 payload, return only the new GameEvents.

        Updates the internal cursor for this game. Raises ``SchemaDriftError``
        if the payload no longer matches the expected schema.
        """
        events = normalize_playbyplay(raw_pbp)
        cursor = self._cursors.setdefault(game_id, _GameCursor(game_id))
        new = [e for e in events if e.event_num > cursor.last_action_num]
        if new:
            cursor.last_action_num = new[-1].event_num
        return new

    def get_live_game_ids(self) -> list[str]:
        """Return game IDs currently in progress (gameStatus == 2)."""
        try:
            from nba_api.live.nba.endpoints.scoreboard import ScoreBoard
            data = ScoreBoard().get_dict()
            games = data.get("scoreboard", {}).get("games", [])
            return [g["gameId"] for g in games if g.get("gameStatus") == 2]
        except Exception as exc:
            logger.debug("scoreboard poll failed (no live games?): %s", exc)
            return []

    def get_raw_pbp(self, game_id: str) -> dict | None:
        """Fetch current play-by-play for one game. Returns None on failure."""
        try:
            from nba_api.live.nba.endpoints.playbyplay import PlayByPlay
            return PlayByPlay(game_id=game_id).get_dict()
        except Exception as exc:
            logger.debug("play-by-play fetch failed for %s: %s", game_id, exc)
            return None

    def _poll_once(self, producer) -> None:
        game_ids = self.get_live_game_ids()
        if not game_ids:
            logger.debug("no live games found")
            return
        for game_id in game_ids:
            raw = self.get_raw_pbp(game_id)
            if raw is None:
                continue
            try:
                new_events = self.poll_new_events(game_id, raw)
            except SchemaDriftError as exc:
                logger.error("schema drift on game %s: %s — skipping", game_id, exc)
                continue
            for event in new_events:
                key, value = serialize_event(event)
                producer.send(self._topic, key=key, value=value)
            if new_events:
                logger.info("game %s: published %d new events (cursor → %d)",
                            game_id, len(new_events), self._cursors[game_id].last_action_num)
