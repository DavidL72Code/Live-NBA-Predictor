import math

import pytest
from tests.conftest import make_event

from nba_winprob.features import GameState, compute_game_features, game_label
from nba_winprob.ingestion.normalize import normalize_playbyplay


class TestTrainingServingConsistency:
    """The core guarantee: streaming updates and batch replay must agree exactly."""

    def test_streaming_equals_batch_on_normalized_game(self, raw_pbp_payload):
        events = normalize_playbyplay(raw_pbp_payload)

        batch = compute_game_features(events)

        state = GameState(events[0].game_id)  # "online" path: one event at a time
        streaming = [state.update(e) for e in events]

        assert streaming == batch


class TestFeatureValues:
    def test_basic_vector(self):
        state = GameState("g1")
        event = make_event(1, period=1, clock_seconds=600, home=10, away=4, game_id="g1")
        vec = state.update(event)
        assert vec.score_diff == 6
        assert vec.seconds_remaining == 3 * 720 + 600
        assert vec.score_diff_norm == pytest.approx(6 / math.sqrt(vec.seconds_remaining + 1))
        assert not vec.is_overtime

    def test_scoring_run_window_prunes_old_baskets(self):
        state = GameState("g1", run_window_seconds=180)
        # Home basket early in Q1 (elapsed 60s)
        state.update(make_event(1, 1, 660, home=2, away=0, game_id="g1"))
        # Away basket at elapsed 120s — both inside any 180s window so far
        vec = state.update(make_event(2, 1, 600, home=2, away=2, game_id="g1"))
        assert (vec.run_home, vec.run_away, vec.run_diff) == (2, 2, 0)
        # Home basket at elapsed 300s: the first two (60s, 120s) fall outside 180s window
        vec = state.update(make_event(3, 1, 420, home=4, away=2, game_id="g1"))
        assert (vec.run_home, vec.run_away, vec.run_diff) == (2, 0, 2)

    def test_run_window_spans_period_boundary(self):
        state = GameState("g1", run_window_seconds=180)
        # Basket with 30s left in Q1 (elapsed 690s)
        state.update(make_event(1, 1, 30, home=3, away=0, game_id="g1"))
        # Basket 60s into Q2 (elapsed 780s) — 90s apart, same window
        vec = state.update(make_event(2, 2, 660, home=3, away=2, game_id="g1"))
        assert (vec.run_home, vec.run_away) == (3, 2)

    def test_overtime_flag_and_remaining(self):
        state = GameState("g1")
        event = make_event(1, period=5, clock_seconds=120, home=100, away=100, game_id="g1")
        vec = state.update(event)
        assert vec.is_overtime
        assert vec.seconds_remaining == 120

    def test_rejects_event_from_other_game(self):
        state = GameState("g1")
        with pytest.raises(ValueError, match="g2"):
            state.update(make_event(1, 1, 700, home=0, away=0, game_id="g2"))


class TestGameLabel:
    def test_home_win(self, raw_pbp_payload):
        events = normalize_playbyplay(raw_pbp_payload)
        assert game_label(events) == 1  # fixture game ends home 5 - away 4

    def test_away_win(self):
        events = [make_event(1, 4, 0, home=98, away=101)]
        assert game_label(events) == 0

    def test_tie_rejected_as_incomplete(self):
        events = [make_event(1, 4, 0, home=100, away=100)]
        with pytest.raises(ValueError, match="tied"):
            game_label(events)

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            game_label([])
