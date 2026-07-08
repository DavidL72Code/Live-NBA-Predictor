"""Normalize raw nba_api PlayByPlayV3 payloads into canonical ``GameEvent``s.

This is the only module that knows the shape of stats.nba.com responses.
Because nba_api is unofficial and endpoints can drift, payload structure is
validated up front and any mismatch raises ``SchemaDriftError`` with the
missing fields named — tests assert on this so drift is caught in CI, not in
production. (This already happened once: PlayByPlayV2 went dark in mid-2025
and returns empty payloads, which is why this module targets V3.)
"""

from __future__ import annotations

import re

from nba_winprob.schemas import EventType, GameEvent

REQUIRED_ACTION_FIELDS = frozenset(
    {"actionNumber", "actionType", "clock", "period", "scoreHome", "scoreAway"}
)

# PlayByPlayV3 actionType strings (lowercased) -> canonical event type.
# The "period" type is split into start/end via subType.
ACTION_TYPE_MAP = {
    "made shot": EventType.FIELD_GOAL_MADE,
    "missed shot": EventType.FIELD_GOAL_MISSED,
    "free throw": EventType.FREE_THROW,
    "rebound": EventType.REBOUND,
    "turnover": EventType.TURNOVER,
    "foul": EventType.FOUL,
    "violation": EventType.VIOLATION,
    "substitution": EventType.SUBSTITUTION,
    "timeout": EventType.TIMEOUT,
    "jump ball": EventType.JUMP_BALL,
    "ejection": EventType.EJECTION,
}

# "11:43" / "0:03.2" (V2 style) or ISO-8601 duration "PT11M43.00S" (V3 style)
_CLOCK_RE = re.compile(r"^\s*(\d+):(\d+(?:\.\d+)?)\s*$")
_CLOCK_ISO_RE = re.compile(r"^PT(\d+)M(\d+(?:\.\d+)?)S$")


class SchemaDriftError(RuntimeError):
    """Raised when an nba_api payload no longer matches the expected schema."""


def parse_clock(value: str) -> float:
    """Parse a period clock string to seconds remaining in the period."""
    match = _CLOCK_ISO_RE.match(value) or _CLOCK_RE.match(value)
    if not match:
        raise SchemaDriftError(f"unparseable clock value: {value!r}")
    minutes, seconds = match.groups()
    return int(minutes) * 60 + float(seconds)


def parse_score_value(value) -> int | None:
    """Parse one scoreHome/scoreAway field; blank means 'unchanged' (forward-fill)."""
    if value is None or not str(value).strip():
        return None
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise SchemaDriftError(f"unparseable score value: {value!r}") from exc


def _event_type(action_type: str, sub_type: str) -> EventType:
    action_type = (action_type or "").strip().lower()
    if action_type == "period":
        sub_type = (sub_type or "").strip().lower()
        return EventType.PERIOD_START if sub_type == "start" else EventType.PERIOD_END
    return ACTION_TYPE_MAP.get(action_type, EventType.UNKNOWN)


def normalize_playbyplay(raw: dict) -> list[GameEvent]:
    """Convert one game's raw PlayByPlayV3 payload into ordered ``GameEvent``s.

    Scores are forward-filled from 0-0: the raw feed leaves scoreHome/scoreAway
    blank on most non-scoring actions.
    """
    game = raw.get("game")
    if not isinstance(game, dict):
        raise SchemaDriftError(f"payload has no 'game' object; top-level keys: {list(raw)}")
    game_id = game.get("gameId")
    actions = game.get("actions")
    if not game_id or not isinstance(actions, list):
        raise SchemaDriftError(f"'game' object missing gameId/actions; keys: {list(game)}")
    if actions:
        missing = REQUIRED_ACTION_FIELDS - actions[0].keys()
        if missing:
            raise SchemaDriftError(f"actions missing fields {sorted(missing)}")

    events: list[GameEvent] = []
    home_score = 0
    away_score = 0
    for action in sorted(actions, key=lambda a: a["actionNumber"]):
        home = parse_score_value(action["scoreHome"])
        away = parse_score_value(action["scoreAway"])
        if home is not None:
            home_score = home
        if away is not None:
            away_score = away

        description = str(action.get("description") or "").strip()
        events.append(
            GameEvent(
                game_id=str(game_id),
                event_num=int(action["actionNumber"]),
                event_type=_event_type(action["actionType"], action.get("subType", "")),
                period=int(action["period"]),
                clock_seconds=parse_clock(action["clock"]),
                home_score=home_score,
                away_score=away_score,
                description=description or None,
            )
        )
    return events
