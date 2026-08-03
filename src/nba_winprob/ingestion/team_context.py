"""Leakage-safe pre-game team context, including venue splits and Elo strength."""

from __future__ import annotations

from collections import defaultdict

from nba_winprob.schemas import TeamGameContext


def build_season_context(rows: list[dict]) -> dict[str, dict[str, TeamGameContext]]:
    """Build overall, venue-specific, and opponent-adjusted context before each game."""
    games: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        games[str(row["GAME_ID"])].append(row)

    ordered = sorted(
        games.items(),
        key=lambda item: (min(str(r.get("GAME_DATE") or "") for r in item[1]), item[0]),
    )
    stats: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "wins": 0, "losses": 0, "margin": 0.0, "streak": 0,
            "home_wins": 0, "home_losses": 0, "home_margin": 0.0,
            "road_wins": 0, "road_losses": 0, "road_margin": 0.0,
            "elo": 1500.0,
        }
    )
    result: dict[str, dict[str, TeamGameContext]] = {}

    for game_id, game_rows in ordered:
        sides: dict[str, dict] = {}
        for row in game_rows:
            side = "home" if " vs. " in str(row.get("MATCHUP") or "") else "away"
            sides[side] = row
        if "home" not in sides or "away" not in sides:
            continue

        contexts: dict[str, TeamGameContext] = {}
        for side, row in sides.items():
            s = stats[str(row["TEAM_ID"])]
            games_played = int(s["wins"]) + int(s["losses"])
            if side == "home":
                venue_wins, venue_losses, venue_margin = s["home_wins"], s["home_losses"], s["home_margin"]
            else:
                venue_wins, venue_losses, venue_margin = s["road_wins"], s["road_losses"], s["road_margin"]
            venue_games = int(venue_wins) + int(venue_losses)
            contexts[side] = TeamGameContext(
                win_pct=int(s["wins"]) / games_played if games_played else 0.5,
                avg_margin=float(s["margin"]) / games_played if games_played else 0.0,
                streak=int(s["streak"]),
                venue_win_pct=int(venue_wins) / venue_games if venue_games else 0.5,
                venue_avg_margin=float(venue_margin) / venue_games if venue_games else 0.0,
                elo_rating=float(s["elo"]),
            )
        result[game_id] = contexts

        home, away = sides["home"], sides["away"]
        home_s = stats[str(home["TEAM_ID"])]
        away_s = stats[str(away["TEAM_ID"])]
        home_wl = str(home.get("WL") or "").upper().strip()
        away_wl = str(away.get("WL") or "").upper().strip()
        if home_wl not in {"W", "L"} or away_wl not in {"W", "L"}:
            continue

        home_margin = float(home.get("PLUS_MINUS") or 0)
        away_margin = float(away.get("PLUS_MINUS") or 0)
        for s, wl, margin in ((home_s, home_wl, home_margin), (away_s, away_wl, away_margin)):
            s["wins"] = int(s["wins"]) + (wl == "W")
            s["losses"] = int(s["losses"]) + (wl == "L")
            s["margin"] = float(s["margin"]) + margin
            streak = int(s["streak"])
            s["streak"] = streak + 1 if wl == "W" and streak > 0 else 1 if wl == "W" else streak - 1 if streak < 0 else -1
        home_s["home_wins"] = int(home_s["home_wins"]) + (home_wl == "W")
        home_s["home_losses"] = int(home_s["home_losses"]) + (home_wl == "L")
        home_s["home_margin"] = float(home_s["home_margin"]) + home_margin
        away_s["road_wins"] = int(away_s["road_wins"]) + (away_wl == "W")
        away_s["road_losses"] = int(away_s["road_losses"]) + (away_wl == "L")
        away_s["road_margin"] = float(away_s["road_margin"]) + away_margin

        expected_home = 1.0 / (1.0 + 10 ** ((float(away_s["elo"]) - float(home_s["elo"]) - 65.0) / 400.0))
        delta = 20.0 * ((1.0 if home_wl == "W" else 0.0) - expected_home)
        home_s["elo"] = float(home_s["elo"]) + delta
        away_s["elo"] = float(away_s["elo"]) - delta

    return result
