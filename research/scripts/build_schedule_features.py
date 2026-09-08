"""Download historical NBA schedules and build leakage-safe schedule features."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

# Approximate arena/city coordinates are sufficient for travel-distance buckets.
# These are static metadata, never derived from game outcomes.
TEAM_GEOGRAPHY: dict[str, tuple[float, float, int]] = {
    "1610612737": (33.7573, -84.3963, -5), "1610612738": (42.3662, -71.0621, -5),
    "1610612751": (40.6826, -73.9754, -5), "1610612766": (35.2251, -80.8392, -5),
    "1610612741": (41.8807, -87.6742, -6), "1610612739": (41.4965, -81.6882, -5),
    "1610612742": (32.7905, -96.8103, -6), "1610612743": (39.7487, -105.0077, -7),
    "1610612765": (42.3410, -83.0550, -5), "1610612744": (37.7680, -122.3877, -8),
    "1610612745": (29.7508, -95.3621, -6), "1610612754": (39.7640, -86.1555, -5),
    "1610612746": (34.0430, -118.2673, -8), "1610612747": (34.0430, -118.2673, -8),
    "1610612763": (35.1382, -90.0506, -6), "1610612748": (25.7814, -80.1870, -5),
    "1610612749": (43.0451, -87.9172, -6), "1610612750": (44.9795, -93.2760, -6),
    "1610612740": (29.9490, -90.0821, -6), "1610612752": (40.7505, -73.9934, -5),
    "1610612760": (35.4634, -97.5151, -6), "1610612753": (28.5392, -81.3839, -5),
    "1610612755": (39.9012, -75.1720, -5), "1610612756": (33.4457, -112.0712, -7),
    "1610612757": (45.5316, -122.6668, -8), "1610612758": (38.5802, -121.4997, -8),
    "1610612759": (29.4270, -98.4375, -6), "1610612761": (43.6435, -79.3791, -5),
    "1610612762": (40.7683, -111.9011, -7), "1610612764": (38.8981, -77.0209, -5),
}


def _distance_miles(first: tuple[float, float], second: tuple[float, float]) -> float:
    radius = 3958.8
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    d_lat, d_lon = lat2 - lat1, lon2 - lon1
    value = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(value))


def collect_schedules(seasons: list[str], output: Path) -> pd.DataFrame:
    from nba_winprob.ingestion.client import NBAStatsClient

    rows: list[dict] = []
    client = NBAStatsClient(min_request_interval=0.2, max_retries=3, timeout=30)
    for season in seasons:
        league_rows = client.get_league_game_log(season)
        by_game: dict[str, list[dict]] = defaultdict(list)
        for row in league_rows:
            by_game[str(row["GAME_ID"])].append(row)
        for game_id, game_rows in by_game.items():
            home = next((r for r in game_rows if " vs. " in str(r.get("MATCHUP") or "")), None)
            away = next((r for r in game_rows if " @ " in str(r.get("MATCHUP") or "")), None)
            if not home or not away:
                continue
            rows.append({
                "game_id": game_id,
                "season": season,
                "game_date": str(home["GAME_DATE"]),
                "home_team_id": str(home["TEAM_ID"]),
                "away_team_id": str(away["TEAM_ID"]),
            })
    result = pd.DataFrame(rows).drop_duplicates("game_id")
    result["game_date"] = pd.to_datetime(result["game_date"], utc=True)
    result = result.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return result


def add_schedule_features(features: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Add only pregame schedule information; update team history after each game."""
    required = {"game_id", "home_team_id", "away_team_id", "game_date"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"schedule missing columns: {sorted(missing)}")

    games = schedule.copy()
    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"], utc=True)
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    last_date: dict[str, pd.Timestamp] = {}
    recent_dates: dict[str, deque[pd.Timestamp]] = defaultdict(lambda: deque(maxlen=6))
    last_venue: dict[str, tuple[float, float, int]] = {}
    last_season: str | None = None
    generated: list[dict] = []

    def context(team_id: str, date: pd.Timestamp) -> dict[str, float]:
        previous = last_date.get(team_id)
        gap = float((date - previous).days - 1) if previous is not None else 7.0
        dates = recent_dates[team_id]
        games_4 = sum((date - value).days <= 3 for value in dates)
        games_7 = sum((date - value).days <= 6 for value in dates)
        return {
            "rest_days": max(0.0, gap),
            "is_back_to_back": float(gap == 0.0),
            "games_last_4_days": float(games_4),
            "games_last_7_days": float(games_7),
        }

    def travel(team_id: str, venue_id: str) -> tuple[float, float]:
        venue = TEAM_GEOGRAPHY[venue_id]
        previous = last_venue.get(team_id, TEAM_GEOGRAPHY[team_id])
        miles = _distance_miles(previous[:2], venue[:2])
        timezone_shift = float(venue[2] - previous[2])
        return miles, timezone_shift

    for row in games.itertuples(index=False):
        if getattr(row, "season", None) != last_season:
            last_date.clear()
            recent_dates.clear()
            last_venue.clear()
            last_season = getattr(row, "season", None)
        date = row.game_date
        home = context(str(row.home_team_id), date)
        away = context(str(row.away_team_id), date)
        home_miles, home_timezone = travel(str(row.home_team_id), str(row.home_team_id))
        away_miles, away_timezone = travel(str(row.away_team_id), str(row.home_team_id))
        generated.append({
            "game_id": str(row.game_id),
            "home_rest_days": home["rest_days"],
            "away_rest_days": away["rest_days"],
            "rest_diff": home["rest_days"] - away["rest_days"],
            "home_back_to_back": home["is_back_to_back"],
            "away_back_to_back": away["is_back_to_back"],
            "back_to_back_diff": home["is_back_to_back"] - away["is_back_to_back"],
            "home_games_last_4_days": home["games_last_4_days"],
            "away_games_last_4_days": away["games_last_4_days"],
            "home_games_last_7_days": home["games_last_7_days"],
            "away_games_last_7_days": away["games_last_7_days"],
            "schedule_density_diff": home["games_last_7_days"] - away["games_last_7_days"],
            "home_travel_miles": home_miles,
            "away_travel_miles": away_miles,
            "travel_miles_diff": home_miles - away_miles,
            "home_timezone_shift": home_timezone,
            "away_timezone_shift": away_timezone,
            "timezone_shift_diff": home_timezone - away_timezone,
        })
        for team_id in (str(row.home_team_id), str(row.away_team_id)):
            last_date[team_id] = date
            recent_dates[team_id].append(date)
            last_venue[team_id] = TEAM_GEOGRAPHY[str(row.home_team_id)]

    schedule_features = pd.DataFrame(generated)
    result = features.copy()
    result["game_id"] = result["game_id"].astype(str)
    result = result.merge(schedule_features, on="game_id", how="left", validate="many_to_one")
    new_cols = [c for c in schedule_features.columns if c != "game_id"]
    if result[new_cols].isna().any().any():
        missing_games = int(result.loc[result[new_cols].isna().any(axis=1), "game_id"].nunique())
        raise ValueError(f"schedule features missing for {missing_games} games")
    return result


def repair_missing_schedule_games(features: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Recover rare postponed/omitted games from the per-game summary endpoint."""
    missing = sorted(set(features["game_id"].astype(str)) - set(schedule["game_id"].astype(str)))
    if not missing:
        return schedule
    from nba_api.stats.endpoints import boxscoresummaryv2

    rows = [schedule]
    for game_id in missing:
        payload = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id, timeout=30).get_dict()
        result_set = next(rs for rs in payload["resultSets"] if rs.get("name") == "GameSummary")
        values = result_set["rowSet"][0]
        fields = dict(zip(result_set["headers"], values, strict=True))
        rows.append(pd.DataFrame([{
            "game_id": game_id,
            "season": f"20{game_id[3:5]}-{int(game_id[3:5]) + 1:02d}",
            "game_date": pd.Timestamp(fields["GAME_DATE_EST"], tz="UTC"),
            "home_team_id": str(fields["HOME_TEAM_ID"]),
            "away_team_id": str(fields["VISITOR_TEAM_ID"]),
        }]))
    repaired = pd.concat(rows, ignore_index=True).drop_duplicates("game_id")
    repaired["game_date"] = pd.to_datetime(repaired["game_date"], utc=True)
    return repaired.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", type=Path, default=Path("data/features/features_enhanced.parquet")
    )
    parser.add_argument(
        "--schedule", type=Path,
        default=Path("data/schedule/regular_season_schedule.parquet"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/features/features_schedule.parquet")
    )
    args = parser.parse_args()
    seasons = sorted(p.name for p in Path("data/raw").iterdir() if p.is_dir())
    if args.schedule.exists():
        schedule = pd.read_parquet(args.schedule)
    else:
        schedule = collect_schedules(seasons, args.schedule)
    schedule["season"] = schedule["game_id"].astype(str).map(
        lambda game_id: f"20{game_id[3:5]}-{int(game_id[3:5]) + 1:02d}"
    )
    features = pd.read_parquet(args.features)
    schedule = repair_missing_schedule_games(features, schedule)
    schedule.to_parquet(args.schedule, index=False)
    result = add_schedule_features(features, schedule)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(json.dumps({
        "seasons": seasons,
        "schedule_games": len(schedule),
        "feature_rows": len(result),
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
