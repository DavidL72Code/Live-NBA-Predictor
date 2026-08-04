from nba_winprob.providers.espn import normalize_summary
from nba_winprob.schemas import EventType


def test_normalize_summary_preserves_provider_id_and_event_contract():
    payload = {
        "header": {
            "competitions": [{
                "competitors": [
                    {"id": "2", "homeAway": "home", "team": {"abbreviation": "BOS"}},
                    {"id": "19", "homeAway": "away", "team": {"abbreviation": "ORL"}},
                ]
            }]
        },
        "plays": [
            {
                "id": "1",
                "clock": {"displayValue": "12:00"},
                "period": {"number": 1},
                "homeScore": 0,
                "awayScore": 0,
                "text": "Tip-off",
                "type": {"text": "Jumpball"},
                "team": {"id": "2"},
                "participants": [{"athlete": {"id": "123"}}],
            },
            {
                "id": "2",
                "clock": {"displayValue": "11:42"},
                "period": {"number": 1},
                "homeScore": 3,
                "awayScore": 0,
                "text": "Player makes 25-foot three point jumper",
                "type": {"text": "Jump Shot"},
                "team": {"id": "2"},
                "participants": [{"athlete": {"id": "456"}}],
            },
        ],
    }

    events = normalize_summary("401811041", payload)

    assert events[0].game_id == "espn:401811041"
    assert events[0].event_type is EventType.JUMP_BALL
    assert events[0].team_tricode == "BOS"
    assert events[1].event_type is EventType.FIELD_GOAL_MADE
    assert events[1].home_score == 3
    assert events[1].shot_value == 3
