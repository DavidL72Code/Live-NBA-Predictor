"""Kafka message serialization.

All messages are JSON-encoded pydantic models. The topic key is always the
``game_id`` bytes so all events for one game land on the same partition
(preserving order — required by the stream processor's per-game GameState).

Keeping it JSON for now (human-readable, zero schema registry dependency).
When throughput becomes a concern, swap to msgpack or protobuf here without
touching any other module.
"""

from __future__ import annotations

from nba_winprob.schemas import FeatureVector, GameEvent


def serialize_event(event: GameEvent) -> tuple[bytes, bytes]:
    """Return (key, value) bytes for a Kafka message."""
    return event.game_id.encode(), event.model_dump_json().encode()


def deserialize_event(value: bytes) -> GameEvent:
    return GameEvent.model_validate_json(value)


def serialize_feature(feature: FeatureVector) -> tuple[bytes, bytes]:
    return feature.game_id.encode(), feature.model_dump_json().encode()


def deserialize_feature(value: bytes) -> FeatureVector:
    return FeatureVector.model_validate_json(value)
