"""Postgres offline feature store.

Stores every FeatureVector ever computed — one row per (game_id, event_num).
Used for:
  - Training: bulk read all rows for a season into a DataFrame
  - Debugging: inspect feature values at any point in any historical game
  - Drift monitoring (Phase 5): compare live distributions to training-era

Schema is created on first use (``ensure_schema()``), so there is no separate
migration step for local dev.

The offline store is intentionally append-only: rows are never updated or
deleted. Re-running the backfill for a game that already exists is safe —
the INSERT ... ON CONFLICT DO NOTHING skips duplicates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS features (
    game_id             TEXT        NOT NULL,
    event_num           INTEGER     NOT NULL,
    period              SMALLINT    NOT NULL,
    seconds_remaining   REAL        NOT NULL,
    seconds_elapsed     REAL        NOT NULL,
    home_score          SMALLINT    NOT NULL,
    away_score          SMALLINT    NOT NULL,
    score_diff          SMALLINT    NOT NULL,
    score_diff_norm     REAL        NOT NULL,
    run_home            SMALLINT    NOT NULL,
    run_away            SMALLINT    NOT NULL,
    run_diff            SMALLINT    NOT NULL,
    is_overtime         BOOLEAN     NOT NULL,
    home_win            SMALLINT,   -- NULL for in-progress games
    PRIMARY KEY (game_id, event_num)
);
CREATE INDEX IF NOT EXISTS features_game_id_idx ON features (game_id);
"""

# Migration for pre-game team context columns added in phase 3.
# ADD COLUMN IF NOT EXISTS is idempotent — safe to run on existing tables.
_MIGRATE_TEAM_CONTEXT = """
ALTER TABLE features ADD COLUMN home_win_pct    REAL     NOT NULL DEFAULT 0.5;
ALTER TABLE features ADD COLUMN home_avg_margin REAL     NOT NULL DEFAULT 0.0;
ALTER TABLE features ADD COLUMN home_streak     SMALLINT NOT NULL DEFAULT 0;
ALTER TABLE features ADD COLUMN away_win_pct    REAL     NOT NULL DEFAULT 0.5;
ALTER TABLE features ADD COLUMN away_avg_margin REAL     NOT NULL DEFAULT 0.0;
ALTER TABLE features ADD COLUMN away_streak     SMALLINT NOT NULL DEFAULT 0;
ALTER TABLE features ADD COLUMN home_venue_win_pct REAL NOT NULL DEFAULT 0.5;
ALTER TABLE features ADD COLUMN home_venue_avg_margin REAL NOT NULL DEFAULT 0.0;
ALTER TABLE features ADD COLUMN away_venue_win_pct REAL NOT NULL DEFAULT 0.5;
ALTER TABLE features ADD COLUMN away_venue_avg_margin REAL NOT NULL DEFAULT 0.0;
ALTER TABLE features ADD COLUMN home_elo_rating REAL NOT NULL DEFAULT 1500.0;
ALTER TABLE features ADD COLUMN away_elo_rating REAL NOT NULL DEFAULT 1500.0;
"""

_INSERT = """
INSERT INTO features (
    game_id, event_num, period, seconds_remaining, seconds_elapsed,
    home_score, away_score, score_diff, score_diff_norm,
    run_home, run_away, run_diff, is_overtime, home_win,
    home_win_pct, home_avg_margin, home_streak,
    away_win_pct, away_avg_margin, away_streak,
    home_venue_win_pct, home_venue_avg_margin,
    away_venue_win_pct, away_venue_avg_margin,
    home_elo_rating, away_elo_rating
) VALUES (
    %(game_id)s, %(event_num)s, %(period)s, %(seconds_remaining)s,
    %(seconds_elapsed)s, %(home_score)s, %(away_score)s, %(score_diff)s,
    %(score_diff_norm)s, %(run_home)s, %(run_away)s, %(run_diff)s,
    %(is_overtime)s, %(home_win)s,
    %(home_win_pct)s, %(home_avg_margin)s, %(home_streak)s,
    %(away_win_pct)s, %(away_avg_margin)s, %(away_streak)s,
    %(home_venue_win_pct)s, %(home_venue_avg_margin)s,
    %(away_venue_win_pct)s, %(away_venue_avg_margin)s,
    %(home_elo_rating)s, %(away_elo_rating)s
) ON CONFLICT (game_id, event_num) DO NOTHING;
"""


class OfflineStore:
    def __init__(self, dsn: str | None = None):
        from nba_winprob.config import get_settings

        self._dsn = dsn or get_settings().postgres_dsn
        if not self._dsn:
            raise RuntimeError(
                "NBA_WINPROB_POSTGRES_DSN is not set — add it to .env"
            )

    def _connect(self):
        import psycopg2

        return psycopg2.connect(self._dsn)

    def ensure_schema(self) -> None:
        """Create the features table and apply any pending column migrations."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(_CREATE_TABLE)
            for stmt in _MIGRATE_TEAM_CONTEXT.strip().splitlines():
                stmt = stmt.strip()
                if not stmt:
                    continue
                try:
                    cur.execute(stmt)
                except Exception as exc:
                    msg = str(exc).lower()
                    if "already exists" in msg or "already has a column" in msg:
                        pass  # idempotent: column was added in a prior run
                    else:
                        raise

    def write_batch(
        self,
        vectors: list,
        home_win: int | None = None,
    ) -> int:
        """Insert a list of FeatureVectors. Returns count of rows inserted.

        ``home_win`` is 1/0 for completed games, None for in-progress games.
        Duplicate (game_id, event_num) pairs are silently skipped.
        """
        rows = [
            {**v.model_dump(), "home_win": home_win, "is_overtime": bool(v.is_overtime)}
            for v in vectors
        ]
        with self._connect() as conn, conn.cursor() as cur:
            for row in rows:
                cur.execute(_INSERT, row)
        return len(rows)

    def read_training_data(self, seasons: list[str] | None = None) -> pd.DataFrame:
        """Read all labeled rows (home_win IS NOT NULL) as a DataFrame.

        ``seasons`` filters by game_id prefix (e.g. '2023-24' → game IDs
        starting with '0022300'). Pass None to read everything.
        """
        import pandas as pd

        where = "WHERE home_win IS NOT NULL"
        params: list = []
        if seasons:
            # NBA game IDs encode the season in the first 4 digits after '00':
            # 002 23 XXXXX = 2023-24 regular season
            placeholders = ",".join(["%s"] * len(seasons))
            prefixes = [_season_to_game_id_prefix(s) for s in seasons]
            where += f" AND LEFT(game_id, 7) IN ({placeholders})"
            params = prefixes

        with self._connect() as conn:
            return pd.read_sql(f"SELECT * FROM features {where}", conn, params=params or None)

    def ping(self) -> bool:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False


def _season_to_game_id_prefix(season: str) -> str:
    """'2023-24' → '0022300' (regular season game ID prefix).

    NBA game IDs: 00 2 YY XXXXX where YY = last two digits of start year.
    e.g. 2023-24 → '0022300...' (7-char prefix).
    """
    year = season.split("-")[0]
    short = year[2:]   # '2023' → '23'
    return f"002{short}00"[:7]
