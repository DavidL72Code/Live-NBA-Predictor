from __future__ import annotations

import pytest

from nba_winprob.schemas import EventType, GameEvent


def pbp_action(
    action_number: int,
    action_type: str,
    period: int,
    clock: str,
    score_home: str = "",
    score_away: str = "",
    sub_type: str = "",
    description: str = "",
) -> dict:
    """One PlayByPlayV3-shaped action (subset of the real fields)."""
    return {
        "actionNumber": action_number,
        "actionType": action_type,
        "subType": sub_type,
        "clock": clock,
        "period": period,
        "scoreHome": score_home,
        "scoreAway": score_away,
        "teamId": 0,
        "teamTricode": "",
        "personId": 0,
        "playerName": "",
        "description": description,
        "isFieldGoal": 0,
        "shotResult": "",
    }


@pytest.fixture
def raw_pbp_payload() -> dict:
    """Fabricated PlayByPlayV3 payload: a short game where the home team

    trails early and wins. Like the real endpoint, scoreHome/scoreAway are
    blank on most non-scoring actions and clocks are ISO-8601 durations.
    """
    actions = [
        pbp_action(1, "period", 1, "PT12M00.00S", "0", "0", sub_type="start",
                   description="Start of 1st Period"),
        pbp_action(2, "Jump Ball", 1, "PT12M00.00S", description="Jump Ball"),
        pbp_action(3, "Made Shot", 1, "PT11M40.00S", "0", "2", description="Away Layup"),
        pbp_action(4, "Missed Shot", 1, "PT11M15.00S", description="MISS Home 3PT"),
        pbp_action(5, "Made Shot", 1, "PT10M58.00S", "3", "2", description="Home 3PT Shot"),
        pbp_action(6, "period", 1, "PT00M00.00S", sub_type="end",
                   description="End of 1st Period"),
        pbp_action(7, "period", 4, "PT12M00.00S", sub_type="start",
                   description="Start of 4th Period"),
        pbp_action(8, "Made Shot", 4, "PT00M30.00S", "3", "4", description="Away Dunk"),
        pbp_action(9, "Made Shot", 4, "PT00M02.50S", "5", "4", description="Home 3PT Shot"),
        pbp_action(10, "period", 4, "PT00M00.00S", sub_type="end",
                   description="End of 4th Period"),
    ]
    return {
        "meta": {"version": 1, "code": 200},
        "game": {"gameId": "0022300001", "videoAvailable": 1, "actions": actions},
    }


def make_event(
    event_num: int,
    period: int,
    clock_seconds: float,
    home: int,
    away: int,
    game_id: str = "0022300001",
    event_type: EventType = EventType.FIELD_GOAL_MADE,
) -> GameEvent:
    return GameEvent(
        game_id=game_id,
        event_num=event_num,
        event_type=event_type,
        period=period,
        clock_seconds=clock_seconds,
        home_score=home,
        away_score=away,
    )
