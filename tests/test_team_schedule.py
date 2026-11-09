"""Tests for the team-catalog schedule fetch."""

from unittest.mock import MagicMock, patch

from nba_winprob.api.server import _fetch_team_schedule


def _payload():
    return {"events": [{
        "id": "401",
        "name": "New York Knicks at Boston Celtics",
        "seasonType": {"type": 2},
        "competitions": [{
            "status": {"type": {"state": "post"}},
            "competitors": [
                {"homeAway": "home", "team": {"abbreviation": "BOS"}, "score": "100"},
                {"homeAway": "away", "team": {"abbreviation": "NY"}, "score": "98"},
            ],
        }],
    }]}


def test_schedule_requests_espn_month_form_not_a_date_range():
    """ESPN 400s on dates=YYYYMMDD-YYYYMMDD; only the YYYYMM form works.

    A range here breaks the catalog for all 30 teams at once, so assert the
    exact parameter rather than trusting the call to stay correct.
    """
    response = MagicMock(ok=True)
    response.json.return_value = _payload()
    with patch("requests.get", return_value=response) as get:
        _fetch_team_schedule.__wrapped__("NYK", "2026-27")
    sent = [call.kwargs["params"]["dates"] for call in get.call_args_list]
    assert sent, "no ESPN request was made"
    for value in sent:
        assert "-" not in value, f"date range form still in use: {value}"
        assert len(value) == 6 and value.isdigit(), f"not a YYYYMM month: {value}"
    assert "202609" in sent and "202706" in sent


def test_schedule_normalizes_espn_tricodes():
    response = MagicMock(ok=True)
    response.json.return_value = _payload()
    with patch("requests.get", return_value=response):
        games = _fetch_team_schedule.__wrapped__("NYK", "2026-27")
    assert games, "NYK should match its own game once NY is normalized"
    assert games[0]["away_team"] == "NYK"
    assert games[0]["home_team"] == "BOS"
