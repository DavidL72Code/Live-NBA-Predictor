"""Canonical event and feature models.

Every layer of the pipeline (ingestion, stream processor, offline batch,
serving) speaks these types. Raw nba_api payloads are normalized into
``GameEvent`` exactly once, at the ingestion boundary — nothing downstream
ever touches a raw stats.nba.com row.
"""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, Field


class EventType(IntEnum):
    """Canonical event types (values follow the legacy EVENTMSGTYPE codes;
    the V3 actionType-string mapping lives in ``ingestion.normalize``)."""

    FIELD_GOAL_MADE = 1
    FIELD_GOAL_MISSED = 2
    FREE_THROW = 3
    REBOUND = 4
    TURNOVER = 5
    FOUL = 6
    VIOLATION = 7
    SUBSTITUTION = 8
    TIMEOUT = 9
    JUMP_BALL = 10
    EJECTION = 11
    PERIOD_START = 12
    PERIOD_END = 13
    UNKNOWN = -1


class GameEvent(BaseModel):
    """One normalized play-by-play event.

    Scores are forward-filled: every event carries the current score even if
    the raw row only populates it on scoring plays.
    """

    game_id: str
    event_num: int
    event_type: EventType
    period: int = Field(ge=1)
    clock_seconds: float = Field(ge=0, description="Seconds left on the period clock")
    home_score: int = Field(ge=0)
    away_score: int = Field(ge=0)
    description: str | None = None
    team_id: str | None = None
    team_tricode: str | None = None
    person_id: str | None = None
    player_name: str | None = None
    action_type: str | None = None
    sub_type: str | None = None
    shot_result: str | None = None
    shot_value: int | None = None

    @property
    def score_diff(self) -> int:
        """Home minus away."""
        return self.home_score - self.away_score


class TeamGameContext(BaseModel):
    """Pre-game context for one team, computed from season game log.

    All values reflect the team's state *entering* the game (i.e. excluding the
    game itself). Season openers get the neutral prior: win_pct=0.5, others=0.
    """

    win_pct: float = Field(
        default=0.5, description="Season W% entering this game; 0.5 for season opener"
    )
    avg_margin: float = Field(
        default=0.0, description="Average point differential per game entering this game"
    )
    streak: int = Field(
        default=0, description="+N = N-game win streak, -N = N-game losing streak"
    )
    venue_win_pct: float = Field(
        default=0.5, description="Win percentage at this team's venue entering the game"
    )
    venue_avg_margin: float = Field(
        default=0.0, description="Average margin at this team's venue entering the game"
    )
    elo_rating: float = Field(
        default=1500.0, description="Opponent-adjusted Elo-style rating entering the game"
    )


class FeatureVector(BaseModel):
    """Model-input features for one game state.

    Produced by ``features.compute.GameState`` — the same class in both the
    streaming and offline paths, which is what guarantees training-serving
    consistency.
    """

    game_id: str
    event_num: int
    period: int
    seconds_remaining: float
    seconds_elapsed: float
    home_score: int
    away_score: int
    score_diff: int
    score_diff_norm: float = Field(
        description="score_diff / sqrt(seconds_remaining + 1); classic win-prob feature"
    )
    run_home: int = Field(description="Home points scored inside the rolling window")
    run_away: int = Field(description="Away points scored inside the rolling window")
    run_diff: int = Field(description="run_home - run_away (recent scoring run)")
    is_overtime: bool
    # Pre-game team context (static per game; defaults give neutral prior when unavailable)
    home_win_pct: float = 0.5
    home_avg_margin: float = 0.0
    home_streak: int = 0
    away_win_pct: float = 0.5
    away_avg_margin: float = 0.0
    away_streak: int = 0
    home_venue_win_pct: float = 0.5
    home_venue_avg_margin: float = 0.0
    away_venue_win_pct: float = 0.5
    away_venue_avg_margin: float = 0.0
    home_elo_rating: float = 1500.0
    away_elo_rating: float = 1500.0
