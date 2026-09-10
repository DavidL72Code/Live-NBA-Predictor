from nba_winprob.providers.espn import normalize_summary, normalize_tricode
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


def test_normalize_tricode_maps_espn_only_codes_to_nba_tricodes():
    """ESPN ships six abbreviations the rest of the app does not recognise.

    Unmapped, these teams drop out of the team catalog and lose their logos,
    so pin the mapping rather than trusting the feed to stay consistent.
    """
    assert normalize_tricode("NY") == "NYK"
    assert normalize_tricode("SA") == "SAS"
    assert normalize_tricode("GS") == "GSW"
    assert normalize_tricode("NO") == "NOP"
    assert normalize_tricode("UTAH") == "UTA"
    assert normalize_tricode("WSH") == "WAS"


def test_normalize_tricode_passes_through_agreeing_and_empty_codes():
    assert normalize_tricode("BOS") == "BOS"
    assert normalize_tricode("phx") == "PHX"
    assert normalize_tricode(None) == ""


def test_normalize_summary_applies_tricode_mapping():
    payload = {
        "header": {
            "competitions": [{
                "competitors": [
                    {"id": "18", "homeAway": "home", "team": {"abbreviation": "NO"}},
                    {"id": "24", "homeAway": "away", "team": {"abbreviation": "UTAH"}},
                ]
            }]
        },
        "plays": [
            {"text": "Jump ball", "type": {"text": "Jump Ball"}, "period": {"number": 1},
             "clock": {"displayValue": "12:00"}, "homeScore": 0, "awayScore": 0,
             "team": {"id": "18"}},
            {"text": "Made 3", "type": {"text": "Three Point Jumper"}, "period": {"number": 1},
             "clock": {"displayValue": "11:40"}, "homeScore": 0, "awayScore": 3,
             "team": {"id": "24"}},
        ],
    }
    events = normalize_summary("401", payload)
    assert [event.team_tricode for event in events] == ["NOP", "UTA"]
