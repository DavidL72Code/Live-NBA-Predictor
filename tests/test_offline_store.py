"""Offline store unit tests — uses an in-memory SQLite stand-in.

psycopg2 speaks PostgreSQL syntax; for unit tests we swap the connection to
use sqlite3 (via a tiny adapter) so no real Postgres is needed. The SQL used
is intentionally ANSI-compatible so this works.
"""

import math
import sqlite3

import pytest

from nba_winprob.schemas import FeatureVector
from nba_winprob.store.offline import OfflineStore, _season_to_game_id_prefix

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature(game_id="0022300001", event_num=1, score_diff=5) -> FeatureVector:
    remaining = 600.0
    return FeatureVector(
        game_id=game_id,
        event_num=event_num,
        period=2,
        seconds_remaining=remaining,
        seconds_elapsed=1080.0,
        home_score=50 + score_diff,
        away_score=50,
        score_diff=score_diff,
        score_diff_norm=score_diff / math.sqrt(remaining + 1),
        run_home=4,
        run_away=2,
        run_diff=2,
        is_overtime=False,
    )


class _SqliteConn:
    """Minimal psycopg2-style context manager around sqlite3."""

    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._conn.commit()

    def cursor(self):
        return _SqliteCursor(self._conn.cursor())


class _SqliteCursor:
    def __init__(self, cur):
        self._cur = cur

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params=None):
        import re

        sql = sql.replace("%s", "?")
        sql = re.sub(r"%\((\w+)\)s", r":\1", sql)
        # sqlite3 doesn't allow multiple statements per execute(); split on ";"
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            self._cur.execute(stmt, params or ())

    def fetchall(self):
        return self._cur.fetchall()

    def fetchone(self):
        return self._cur.fetchone()

    def description(self):
        return self._cur.description


@pytest.fixture
def sqlite_store(tmp_path):
    """OfflineStore whose _connect() returns a SQLite connection."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    store = OfflineStore.__new__(OfflineStore)
    store._dsn = "sqlite:///:memory:"
    store._connect = lambda: _SqliteConn(conn)
    store.ensure_schema()
    return store, conn


class TestOfflineStore:
    def test_write_and_read_back(self, sqlite_store):
        store, conn = sqlite_store
        vecs = [_make_feature(event_num=i) for i in range(1, 4)]
        store.write_batch(vecs, home_win=1)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM features")
        assert cur.fetchone()[0] == 3

    def test_duplicate_insert_is_skipped(self, sqlite_store):
        store, conn = sqlite_store
        vec = _make_feature(event_num=1)
        store.write_batch([vec], home_win=1)
        store.write_batch([vec], home_win=1)   # same primary key
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM features")
        assert cur.fetchone()[0] == 1

    def test_home_win_stored_correctly(self, sqlite_store):
        store, conn = sqlite_store
        store.write_batch([_make_feature(event_num=1)], home_win=0)
        cur = conn.cursor()
        cur.execute("SELECT home_win FROM features")
        assert cur.fetchone()[0] == 0

    def test_in_progress_game_has_null_label(self, sqlite_store):
        store, conn = sqlite_store
        store.write_batch([_make_feature(event_num=1)], home_win=None)
        cur = conn.cursor()
        cur.execute("SELECT home_win FROM features")
        assert cur.fetchone()[0] is None

    def test_write_batch_returns_row_count(self, sqlite_store):
        store, _ = sqlite_store
        vecs = [_make_feature(event_num=i) for i in range(5)]
        assert store.write_batch(vecs, home_win=1) == 5


class TestSeasonPrefix:
    def test_regular_season_prefix(self):
        assert _season_to_game_id_prefix("2023-24") == "0022300"

    def test_different_season(self):
        assert _season_to_game_id_prefix("2022-23") == "0022200"
