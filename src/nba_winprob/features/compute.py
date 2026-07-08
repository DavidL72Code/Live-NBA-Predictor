"""Feature computation — the single source of truth.

``GameState`` is an incremental accumulator: feed it one ``GameEvent`` at a
time and it emits the ``FeatureVector`` for that moment. The streaming path
(stream processor -> Redis) calls ``update()`` per live event; the offline
path (``compute_game_features``) replays historical events through the exact
same accumulator. There is intentionally no second implementation of any
feature — that is the training-serving consistency guarantee from the plan.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable

from nba_winprob import gametime
from nba_winprob.schemas import FeatureVector, GameEvent

DEFAULT_RUN_WINDOW_SECONDS = 180.0  # "recent scoring run" = last 3 minutes of game time


class GameState:
    """Incremental per-game feature state.

    Consumes events in order and maintains only what the rolling features
    need (a deque of recent scoring deltas), so memory stays bounded no
    matter how long the game runs.
    """

    def __init__(self, game_id: str, run_window_seconds: float = DEFAULT_RUN_WINDOW_SECONDS):
        self.game_id = game_id
        self.run_window_seconds = run_window_seconds
        self._last_home = 0
        self._last_away = 0
        # (elapsed_seconds, home_delta, away_delta) for scoring events in the window
        self._scoring: deque[tuple[float, int, int]] = deque()
        # Real NBA feeds contain late-recorded events whose clock runs backward
        # (e.g. a substitution logged after the plays that followed it). Game
        # time is clamped monotonic so rolling windows stay well-defined and,
        # crucially, streaming and batch replay stay identical in feed order.
        self._max_elapsed = 0.0
        self._current_period = 0
        self._min_remaining_in_period = float("inf")

    def update(self, event: GameEvent) -> FeatureVector:
        if event.game_id != self.game_id:
            raise ValueError(
                f"GameState for {self.game_id} received event from game {event.game_id}"
            )

        elapsed = max(
            gametime.seconds_elapsed(event.period, event.clock_seconds), self._max_elapsed
        )
        self._max_elapsed = elapsed
        if event.period != self._current_period:
            self._current_period = event.period
            self._min_remaining_in_period = float("inf")
        remaining = min(
            gametime.seconds_remaining(event.period, event.clock_seconds),
            self._min_remaining_in_period,
        )
        self._min_remaining_in_period = remaining

        home_delta = event.home_score - self._last_home
        away_delta = event.away_score - self._last_away
        self._last_home = event.home_score
        self._last_away = event.away_score
        if home_delta or away_delta:
            self._scoring.append((elapsed, home_delta, away_delta))

        cutoff = elapsed - self.run_window_seconds
        while self._scoring and self._scoring[0][0] <= cutoff:
            self._scoring.popleft()

        run_home = sum(h for _, h, _ in self._scoring)
        run_away = sum(a for _, _, a in self._scoring)
        score_diff = event.home_score - event.away_score

        return FeatureVector(
            game_id=event.game_id,
            event_num=event.event_num,
            period=event.period,
            seconds_remaining=remaining,
            seconds_elapsed=elapsed,
            home_score=event.home_score,
            away_score=event.away_score,
            score_diff=score_diff,
            score_diff_norm=score_diff / math.sqrt(remaining + 1.0),
            run_home=run_home,
            run_away=run_away,
            run_diff=run_home - run_away,
            is_overtime=event.period > gametime.REGULATION_PERIODS,
        )


def compute_game_features(
    events: Iterable[GameEvent],
    run_window_seconds: float = DEFAULT_RUN_WINDOW_SECONDS,
) -> list[FeatureVector]:
    """Offline/batch path: replay a game's events through ``GameState``.

    Events must be in game order (as emitted by the normalizer).
    """
    state: GameState | None = None
    vectors: list[FeatureVector] = []
    for event in events:
        if state is None:
            state = GameState(event.game_id, run_window_seconds=run_window_seconds)
        vectors.append(state.update(event))
    return vectors


def game_label(events: Iterable[GameEvent]) -> int:
    """Training label: 1 if the home team won, else 0.

    Uses the final score of the last event; caller is responsible for only
    passing completed games.
    """
    last: GameEvent | None = None
    for event in events:
        last = event
    if last is None:
        raise ValueError("cannot label a game with no events")
    if last.home_score == last.away_score:
        raise ValueError(
            f"game {last.game_id} ends tied {last.home_score}-{last.away_score}; "
            "incomplete play-by-play?"
        )
    return int(last.home_score > last.away_score)
