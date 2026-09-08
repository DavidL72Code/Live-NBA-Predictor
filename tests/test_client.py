"""Tests for NBA endpoint normalization and lineup inference."""

from nba_winprob.ingestion.client import derive_opening_lineup


def test_derive_opening_lineup_uses_players_before_first_substitution():
    actions = []
    for number, player_id in enumerate(range(101, 106), start=1):
        actions.append({
            "actionNumber": number,
            "period": 1,
            "teamId": "10",
            "personId": player_id,
            "playerName": f"Home {player_id}",
            "actionType": "Made Shot",
        })
    for number, player_id in enumerate(range(201, 206), start=10):
        actions.append({
            "actionNumber": number,
            "period": 1,
            "teamId": "20",
            "personId": player_id,
            "playerName": f"Away {player_id}",
            "actionType": "Rebound",
        })
    actions.extend([
        {
            "actionNumber": 20,
            "period": 1,
            "teamId": "10",
            "personId": 301,
            "playerName": "Home bench",
            "actionType": "Substitution",
        },
        {
            "actionNumber": 21,
            "period": 1,
            "teamId": "20",
            "personId": 302,
            "playerName": "Away bench",
            "actionType": "Substitution",
        },
    ])

    lineup = derive_opening_lineup(actions, "10", "20")

    assert [player["player_id"] for player in lineup["home"]] == [str(i) for i in range(101, 106)]
    assert [player["player_id"] for player in lineup["away"]] == [str(i) for i in range(201, 206)]
