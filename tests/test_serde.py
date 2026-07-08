"""Serialization round-trips — no Kafka broker needed."""

from tests.conftest import make_event

from nba_winprob.schemas import EventType, FeatureVector, GameEvent
from nba_winprob.streaming.serde import (
    deserialize_event,
    deserialize_feature,
    serialize_event,
    serialize_feature,
)


def test_event_round_trip():
    event = make_event(1, period=2, clock_seconds=480, home=14, away=10)
    key, value = serialize_event(event)
    assert key == event.game_id.encode()
    assert deserialize_event(value) == event


def test_event_preserves_all_fields():
    event = GameEvent(
        game_id="0022399999",
        event_num=42,
        event_type=EventType.FREE_THROW,
        period=3,
        clock_seconds=62.5,
        home_score=88,
        away_score=90,
        description="Curry Free Throw 1 of 2",
    )
    _, value = serialize_event(event)
    recovered = deserialize_event(value)
    assert recovered.description == "Curry Free Throw 1 of 2"
    assert recovered.clock_seconds == 62.5
    assert recovered.event_type == EventType.FREE_THROW


def test_feature_round_trip():
    import math

    vec = FeatureVector(
        game_id="0022399999",
        event_num=7,
        period=4,
        seconds_remaining=90.0,
        seconds_elapsed=2790.0,
        home_score=105,
        away_score=102,
        score_diff=3,
        score_diff_norm=3 / math.sqrt(91),
        run_home=8,
        run_away=5,
        run_diff=3,
        is_overtime=False,
    )
    key, value = serialize_feature(vec)
    assert key == vec.game_id.encode()
    recovered = deserialize_feature(value)
    assert recovered == vec


def test_key_is_game_id_bytes():
    event = make_event(1, 1, 720, 0, 0, game_id="0021234567")
    key, _ = serialize_event(event)
    assert key == b"0021234567"
