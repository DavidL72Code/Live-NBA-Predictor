"""Stream processor.

Consumes ``GameEvent`` messages from the ``nba.game.events`` topic, feeds
each one through its per-game ``GameState`` accumulator, and publishes the
resulting ``FeatureVector`` to the ``nba.game.features`` topic.

This is where the training-serving consistency guarantee pays off: the same
``GameState.update()`` call used by the offline ``build-features`` CLI is
what runs here, live, one event at a time.

Design:
- One ``GameState`` per game, lazily created on the first event.
- Out-of-order events within a game are tolerated by the ``GameState``
  monotonic-clock clamping added in Phase 1.
- At-least-once delivery: Kafka offsets are committed after the feature
  is published downstream, so a crash between consume and publish
  replays the event. Features are idempotent (same event → same vector).
"""

from __future__ import annotations

import logging
import signal

from nba_winprob.config import get_settings
from nba_winprob.features import GameState
from nba_winprob.streaming.serde import deserialize_event, serialize_feature

logger = logging.getLogger(__name__)


def process_message(
    raw_value: bytes,
    states: dict[str, GameState],
) -> tuple[bytes, bytes] | None:
    """Pure processing step — no Kafka I/O.

    Deserializes one raw Kafka value, routes it through the correct GameState,
    and returns (key, value) bytes ready to produce, or None on parse error.
    Extracted so unit tests can call it without a broker.
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
    return serialize_feature(feature)


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

    def run(self) -> None:
        from kafka import KafkaConsumer, KafkaProducer  # lazy: not needed in tests

        consumer = KafkaConsumer(
            self._events_topic,
            bootstrap_servers=self._bootstrap,
            group_id=self._group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=False,   # manual commit after downstream publish
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
                    key, value = result
                    producer.send(self._features_topic, key=key, value=value)
                    producer.flush()
                consumer.commit()
        finally:
            consumer.close()
            producer.close()
            logger.info("processor shut down cleanly")

    def _handle_sigterm(self, signum, frame) -> None:
        logger.info("SIGTERM received — finishing current message then stopping")
        self._running = False
