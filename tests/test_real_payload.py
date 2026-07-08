"""Regression test against a real captured PlayByPlayV3 payload.

Fixture: game 0022300001 (CLE @ IND, 2023-24, home IND won 121-116), captured
from stats.nba.com in July 2025. If nba_api or the endpoint changes shape,
re-capture and the drift will show up here before it hits the pipeline. The
payload also preserves real feed quirks the pipeline must tolerate: blank
scores on most actions and late-recorded events whose clock runs backward.
"""

import json
from pathlib import Path

import pytest

from nba_winprob.features import compute_game_features, game_label
from nba_winprob.ingestion.normalize import normalize_playbyplay

FIXTURE = Path(__file__).parent / "fixtures" / "pbp_v3_0022300001.json"


@pytest.fixture(scope="module")
def real_events():
    return normalize_playbyplay(json.loads(FIXTURE.read_text()))


def test_normalizes_full_game(real_events):
    assert len(real_events) == 504
    # Home team (DEN) won 121-116
    assert (real_events[-1].home_score, real_events[-1].away_score) == (121, 116)
    assert game_label(real_events) == 1


def test_scores_monotonically_nondecreasing(real_events):
    for prev, cur in zip(real_events, real_events[1:], strict=False):
        assert cur.home_score >= prev.home_score
        assert cur.away_score >= prev.away_score


def test_features_computed_for_every_event(real_events):
    vectors = compute_game_features(real_events)
    assert len(vectors) == len(real_events)
    final = vectors[-1]
    assert final.seconds_remaining == 0
    assert final.score_diff == 5
    # This feed contains two late-recorded events whose raw clock runs
    # backward; GameState must clamp so emitted game time stays monotonic.
    elapsed = [v.seconds_elapsed for v in vectors]
    assert elapsed == sorted(elapsed)
    remaining = [v.seconds_remaining for v in vectors]
    assert remaining == sorted(remaining, reverse=True)  # no OT in this game
