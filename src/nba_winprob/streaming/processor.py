"""Stream processor.

Consumes ``GameEvent`` messages from the ``nba.game.events`` topic, feeds
each one through its per-game ``GameState`` accumulator, publishes the
resulting ``FeatureVector`` to the ``nba.game.features`` topic, and writes
it to the Redis online store so the serving layer can read it immediately.

Design:
- One ``GameState`` per game, lazily created on the first event.
- Out-of-order events within a game are tolerated by the ``GameState``
  monotonic-clock clamping added in Phase 1.
- At-least-once delivery: Kafka offsets are committed after both the
  Kafka publish and the Redis write complete.
- Redis is optional at startup: if NBA_WINPROB_REDIS_URL is unreachable,
  the processor logs a warning and continues (features still flow through
  Kafka; serving degrades gracefully).
"""

from __future__ import annotations

import logging
import signal

from nba_winprob.config import get_settings
from nba_winprob.features import GameState
from nba_winprob.schemas import FeatureVector
from nba_winprob.streaming.serde import deserialize_event, serialize_feature

logger = logging.getLogger(__name__)


def process_message(
    raw_value: bytes,
    states: dict[str, GameState],
) -> tuple[bytes, bytes, FeatureVector] | None:
    """Pure processing step — no Kafka or Redis I/O.

    Deserializes one raw Kafka value, routes it through the correct GameState,
    and returns (key, value, feature) or None on parse error.
    Extracted so unit tests can call it without a broker or Redis.
    """
    try:
        event = deserialize_event(raw_value)
    except Exception as exc:
        logger.warning("failed to deserialize event: %s", exc)
        return None

    if event.game_id not in states:
        states[event.game_id] = GameState(event.game_id)
        logger.info("new game started: %s", event.game_id)

    feature = states[event.game_id].update(event)
    key, value = serialize_feature(feature)
    return key, value, feature


class StreamProcessor:
    """Consumes game events from Kafka, emits feature vectors.

    Usage::

        StreamProcessor().run()   # blocks; Ctrl-C or SIGTERM to stop
    """

    def __init__(self):
        settings = get_settings()
        self._bootstrap = settings.kafka_bootstrap_servers
        self._events_topic = settings.kafka_events_topic
        self._features_topic = settings.kafka_features_topic
        self._group_id = settings.kafka_consumer_group
        self._states: dict[str, GameState] = {}
        self._running = False
        self._online_store = None

    def _init_online_store(self):
        from nba_winprob.store.online import OnlineStore

        store = OnlineStore()
        if store.ping():
            logger.info("Redis online store connected")
            self._online_store = store
        else:
            logger.warning("Redis unreachable — features will not be written to online store")

    def run(self) -> None:
        from kafka import KafkaConsumer, KafkaProducer  # lazy: not needed in tests

        self._init_online_store()
        consumer = KafkaConsumer(
            self._events_topic,
            bootstrap_servers=self._bootstrap,
            group_id=self._group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=False,   # manual commit after all downstream writes
        )
        producer = KafkaProducer(bootstrap_servers=self._bootstrap)

        self._running = True
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        logger.info("processor connected; consuming %s → %s",
                    self._events_topic, self._features_topic)
        try:
            for msg in consumer:
                if not self._running:
                    break
                result = process_message(msg.value, self._states)
                if result is not None:
                    key, value, feature = result
                    producer.send(self._features_topic, key=key, value=value)
                    producer.flush()
                    if self._online_store is not None:
                        try:
                            self._online_store.write(feature)
                        except Exception as exc:
                            logger.warning("Redis write failed: %s", exc)
                consumer.commit()
        finally:
            consumer.close()
            producer.close()
            logger.info("processor shut down cleanly")

    def _handle_sigterm(self, signum, frame) -> None:
        logger.info("SIGTERM received — finishing current message then stopping")
        self._running = False
