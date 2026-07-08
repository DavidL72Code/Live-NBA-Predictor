"""Stream processor unit tests — no broker needed.

Tests call ``process_message`` directly with raw bytes, which is the pure
processing step extracted from the consumer loop. Kafka I/O is never touched.
"""


from tests.conftest import make_event

from nba_winprob.features import GameState
from nba_winprob.schemas import FeatureVector
from nba_winprob.streaming.processor import process_message
from nba_winprob.streaming.serde import deserialize_feature, serialize_event


def _raw(event_num, period, clock_seconds, home, away, game_id="g1"):
    event = make_event(event_num, period, clock_seconds, home, away, game_id=game_id)
    _, value = serialize_event(event)
    return value


class TestProcessMessage:
    def test_returns_key_and_value_bytes(self):
        states = {}
        result = process_message(_raw(1, 1, 720, 0, 0), states)
        assert result is not None
        key, value = result
        assert key == b"g1"
        vec = deserialize_feature(value)
        assert isinstance(vec, FeatureVector)

    def test_creates_game_state_on_first_event(self):
        states = {}
        process_message(_raw(1, 1, 720, 0, 0), states)
        assert "g1" in states
        assert isinstance(states["g1"], GameState)

    def test_reuses_state_across_events(self):
        states = {}
        process_message(_raw(1, 1, 600, 3, 0), states)
        _, v2 = process_message(_raw(2, 1, 540, 3, 2), states)
        vec = deserialize_feature(v2)
        # run_away should reflect both events being in the window
        assert vec.run_away == 2

    def test_multiple_games_get_independent_states(self):
        states = {}
        process_message(_raw(1, 1, 600, 10, 5, game_id="g1"), states)
        process_message(_raw(1, 1, 600, 3, 0, game_id="g2"), states)
        assert "g1" in states and "g2" in states
        # Sanity: g1 and g2 don't bleed into each other
        _, raw_g1 = process_message(_raw(2, 1, 540, 10, 7, game_id="g1"), states)
        vec_g1 = deserialize_feature(raw_g1)
        assert vec_g1.away_score == 7

    def test_bad_bytes_returns_none(self):
        states = {}
        result = process_message(b"not json at all", states)
        assert result is None

    def test_feature_values_match_direct_gamestate(self):
        """process_message must produce the same FeatureVector as calling GameState directly."""
        from nba_winprob.features import GameState as GS

        event = make_event(1, 2, 360, 55, 50, game_id="g99")
        direct_state = GS("g99")
        expected = direct_state.update(event)

        states = {}
        _, value = process_message(serialize_event(event)[1], states)
        got = deserialize_feature(value)
        assert got == expected

    def test_streaming_consistency_across_game(self):
        """Replaying a full game through process_message must match compute_game_features."""
        import json
        from pathlib import Path

        from nba_winprob.features import compute_game_features
        from nba_winprob.ingestion.normalize import normalize_playbyplay
        from nba_winprob.streaming.serde import serialize_event

        raw = json.loads(
            (Path(__file__).parent / "fixtures" / "pbp_v3_0022300001.json").read_text()
        )
        events = normalize_playbyplay(raw)
        batch_vectors = compute_game_features(events)

        states = {}
        streaming_vectors = []
        for event in events:
            _, value = process_message(serialize_event(event)[1], states)
            streaming_vectors.append(deserialize_feature(value))

        assert streaming_vectors == batch_vectors
