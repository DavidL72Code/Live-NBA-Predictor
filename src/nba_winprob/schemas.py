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

    @property
    def score_diff(self) -> int:
        """Home minus away."""
        return self.home_score - self.away_score


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
