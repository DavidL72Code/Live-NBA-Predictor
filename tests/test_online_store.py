"""Online store unit tests — uses fakeredis, no real Redis needed."""

import math

import pytest

from nba_winprob.schemas import FeatureVector
from nba_winprob.store.online import _KEY_PREFIX, _TTL_SECONDS, OnlineStore


@pytest.fixture
def fake_store(monkeypatch):
    """OnlineStore backed by fakeredis (in-memory, no server)."""
    fakeredis = pytest.importorskip("fakeredis")

    fake_client = fakeredis.FakeRedis(decode_responses=True)
    store = OnlineStore.__new__(OnlineStore)
    store._url = "redis://fake"
    monkeypatch.setattr(store, "_client", lambda: fake_client)
    return store, fake_client


def _make_feature(game_id="g1", event_num=42, score_diff=5) -> FeatureVector:
    remaining = 90.0
    return FeatureVector(
        game_id=game_id,
        event_num=event_num,
        period=4,
        seconds_remaining=remaining,
        seconds_elapsed=2790.0,
        home_score=105,
        away_score=105 - score_diff,
        score_diff=score_diff,
        score_diff_norm=score_diff / math.sqrt(remaining + 1),
        run_home=6,
        run_away=4,
        run_diff=2,
        is_overtime=False,
    )


class TestOnlineStore:
    def test_write_and_read_roundtrip(self, fake_store):
        store, _ = fake_store
        vec = _make_feature()
        store.write(vec)
        got = store.read("g1")
        assert got == vec

    def test_read_missing_returns_none(self, fake_store):
        store, _ = fake_store
        assert store.read("nonexistent") is None

    def test_write_sets_ttl(self, fake_store):
        store, client = fake_store
        vec = _make_feature()
        store.write(vec)
        ttl = client.ttl(f"{_KEY_PREFIX}g1")
        assert 0 < ttl <= _TTL_SECONDS

    def test_write_overwrites_previous_value(self, fake_store):
        store, _ = fake_store
        store.write(_make_feature(event_num=1, score_diff=3))
        store.write(_make_feature(event_num=2, score_diff=7))
        got = store.read("g1")
        assert got.score_diff == 7
        assert got.event_num == 2

    def test_multiple_games_independent(self, fake_store):
        store, _ = fake_store
        store.write(_make_feature(game_id="g1", score_diff=3))
        store.write(_make_feature(game_id="g2", score_diff=-5))
        assert store.read("g1").score_diff == 3
        assert store.read("g2").score_diff == -5

    def test_delete_removes_key(self, fake_store):
        store, _ = fake_store
        store.write(_make_feature())
        store.delete("g1")
        assert store.read("g1") is None
