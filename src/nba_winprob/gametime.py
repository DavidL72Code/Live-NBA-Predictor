"""Game clock math shared by normalization and feature computation.

All functions take the period number (1-4 regulation, 5+ overtime) and the
seconds left on the period clock, and convert to game-level elapsed/remaining
seconds. Overtime convention: ``seconds_remaining`` counts only the clock of
the current period, since the total number of overtimes is unknowable live.
"""

from __future__ import annotations

REGULATION_PERIODS = 4
REGULATION_PERIOD_SECONDS = 720.0  # 12 minutes
OVERTIME_PERIOD_SECONDS = 300.0  # 5 minutes
REGULATION_TOTAL_SECONDS = REGULATION_PERIODS * REGULATION_PERIOD_SECONDS


def period_length_seconds(period: int) -> float:
    """Length of a period in seconds (12 min regulation, 5 min OT)."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return REGULATION_PERIOD_SECONDS if period <= REGULATION_PERIODS else OVERTIME_PERIOD_SECONDS


def seconds_elapsed(period: int, clock_seconds: float) -> float:
    """Total game seconds elapsed at a given period/clock.

    Monotonically increasing across the whole game including overtimes, so it
    is safe to use as the time axis for rolling-window features.
    """
    completed_regulation = min(period - 1, REGULATION_PERIODS)
    completed_overtimes = max(period - 1 - REGULATION_PERIODS, 0)
    completed = (
        completed_regulation * REGULATION_PERIOD_SECONDS
        + completed_overtimes * OVERTIME_PERIOD_SECONDS
    )
    return completed + (period_length_seconds(period) - clock_seconds)


def seconds_remaining(period: int, clock_seconds: float) -> float:
    """Game seconds remaining. In overtime, just the current period clock."""
    if period > REGULATION_PERIODS:
        return clock_seconds
    remaining_full_periods = REGULATION_PERIODS - period
    return remaining_full_periods * REGULATION_PERIOD_SECONDS + clock_seconds
