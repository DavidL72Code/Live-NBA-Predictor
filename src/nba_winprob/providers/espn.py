"""ESPN NBA adapter for the canonical game-event contract.

ESPN IDs are intentionally namespaced as ``espn:<event_id>`` so they cannot be
mistaken for NBA Stats game IDs elsewhere in the application.
"""

from __future__ import annotations

import re

import requests

from nba_winprob.schemas import EventType, GameEvent

_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
_CLOCK_RE = re.compile(r"^(\d+):(\d+(?:\.\d+)?)$")


def _clock_seconds(value: str) -> float:
    match = _CLOCK_RE.match(str(value or "12:00").strip())
    if not match:
        return 0.0
    return int(match.group(1)) * 60 + float(match.group(2))


def _event_type(text: str, type_text: str) -> EventType:
    value = f"{type_text} {text}".lower()
    if "period" in value and "end" in value:
        return EventType.PERIOD_END
    if "period" in value and "start" in value:
        return EventType.PERIOD_START
    if "free throw" in value:
        return EventType.FREE_THROW
    if "rebound" in value:
        return EventType.REBOUND
    if "turnover" in value:
        return EventType.TURNOVER
    if "foul" in value:
        return EventType.FOUL
    if "substitution" in value:
        return EventType.SUBSTITUTION
    if "timeout" in value:
        return EventType.TIMEOUT
    if "jumpball" in value or "jump ball" in value:
        return EventType.JUMP_BALL
    if "violation" in value:
        return EventType.VIOLATION
    if "makes" in value or "made" in value:
        return EventType.FIELD_GOAL_MADE
    if "misses" in value or "missed" in value:
        return EventType.FIELD_GOAL_MISSED
    return EventType.UNKNOWN


def _shot_value(text: str, event_type: EventType) -> int | None:
    value = text.lower()
    if event_type == EventType.FREE_THROW:
        return 1
    if event_type not in {EventType.FIELD_GOAL_MADE, EventType.FIELD_GOAL_MISSED}:
        return None
    if "three point" in value or "3-point" in value or "three-pointer" in value:
        return 3
    return 2


def _summary(event_id: str) -> dict:
    response = requests.get(
        _SUMMARY_URL,
        params={"event": event_id},
        headers={"Accept": "application/json", "User-Agent": "SwooshAI/1.0"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload.get("plays"), list):
        raise RuntimeError("ESPN summary did not contain a plays list")
    return payload


def normalize_summary(event_id: str, payload: dict | None = None) -> list[GameEvent]:
    """Convert ESPN summary plays into the app's canonical GameEvent list."""
    payload = payload or _summary(event_id)
    competition = (payload.get("header", {}).get("competitions") or [{}])[0]
    competitors = competition.get("competitors") or []
    team_tricode = {
        str(team.get("id")): str(team.get("team", {}).get("abbreviation") or "")
        for team in competitors
    }
    events: list[GameEvent] = []
    for index, play in enumerate(payload.get("plays", []), start=1):
        text = str(play.get("text") or "")
        type_text = str(play.get("type", {}).get("text") or "")
        event_type = _event_type(text, type_text)
        period = int(play.get("period", {}).get("number") or 1)
        participant = (play.get("participants") or [{}])[0]
        participant_id = str(participant.get("athlete", {}).get("id") or "") or None
        events.append(GameEvent(
            game_id=f"espn:{event_id}",
            event_num=index,
            event_type=event_type,
            period=max(period, 1),
            clock_seconds=_clock_seconds(play.get("clock", {}).get("displayValue")),
            home_score=int(play.get("homeScore") or 0),
            away_score=int(play.get("awayScore") or 0),
            description=text or None,
            team_id=str(play.get("team", {}).get("id") or "") or None,
            team_tricode=team_tricode.get(str(play.get("team", {}).get("id") or "")),
            person_id=participant_id,
            action_type=type_text or None,
            shot_value=_shot_value(text, event_type),
        ))
    return events


def fetch_events(game_id: str) -> list[GameEvent]:
    if not game_id.startswith("espn:"):
        raise ValueError(f"not an ESPN game ID: {game_id}")
    return normalize_summary(game_id.removeprefix("espn:"))
