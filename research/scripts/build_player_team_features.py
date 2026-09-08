"""Build prior-game-only shooting, rotation, and lineup-continuity features."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from nba_winprob import gametime
from nba_winprob.ingestion.client import derive_opening_lineup

SUB_RE = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+)$", re.IGNORECASE)


def _name_keys(value: str) -> set[str]:
    text = " ".join(str(value or "").replace(".", "").split()).lower()
    if not text:
        return set()
    return {text, text.split()[-1]}


def _player_id_by_name(actions: list[dict]) -> dict[str, str]:
    result: dict[str, str] = {}
    for action in actions:
        player_id = str(action.get("personId") or "")
        if not player_id or player_id == "0":
            continue
        for value in (action.get("playerName"), action.get("playerNameI")):
            for key in _name_keys(str(value or "")):
                result.setdefault(key, player_id)
    return result


def _game_summary(raw: dict, home_id: str, away_id: str) -> dict:
    actions = sorted(raw["game"]["actions"], key=lambda item: int(item.get("actionNumber") or 0))
    by_team: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    player_minutes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    name_to_id = _player_id_by_name(actions)
    opening = derive_opening_lineup(actions, home_id, away_id)
    lineups = {"home": set(item["player_id"] for item in opening["home"]),
               "away": set(item["player_id"] for item in opening["away"])}
    team_for_side = {home_id: "home", away_id: "away"}
    last_elapsed = 0.0
    last_period = 1

    def add_minutes(delta: float) -> None:
        for side, lineup in lineups.items():
            for player_id in lineup:
                player_minutes[side][player_id] += delta / 60.0

    for action in actions:
        period = int(action.get("period") or 1)
        clock = str(action.get("clock") or "PT00M00.00S")
        try:
            clock_seconds = float(clock[2:clock.index("M")]) * 60 + float(
                clock[clock.index("M") + 1:-1]
            )
            elapsed = gametime.seconds_elapsed(period, clock_seconds)
        except (ValueError, IndexError):
            elapsed = last_elapsed
        add_minutes(max(0.0, elapsed - last_elapsed))
        last_elapsed = max(last_elapsed, elapsed)
        last_period = max(last_period, period)

        team_id = str(action.get("teamId") or "")
        side = team_for_side.get(team_id)
        action_type = str(action.get("actionType") or "")
        if side:
            stats = by_team[side]
            if action_type in {"Made Shot", "Missed Shot"} and action.get("isFieldGoal"):
                stats["fga"] += 1
                stats["fgm"] += action_type == "Made Shot"
                stats["three_pa"] += int(action.get("shotValue") or 0) == 3
                stats["three_pm"] += (
                    int(action.get("shotValue") or 0) == 3
                    and action_type == "Made Shot"
                )
            elif (
                action_type == "Free Throw"
                and str(action.get("shotResult") or "").lower() in {"made", "missed"}
            ):
                stats["fta"] += 1
                stats["ftm"] += str(action.get("shotResult") or "").lower() == "made"

            if action_type == "Substitution":
                match = SUB_RE.search(str(action.get("description") or ""))
                if match:
                    incoming = name_to_id.get(match.group(1).strip().lower())
                    incoming = incoming or name_to_id.get(
                        match.group(1).strip().split()[-1].lower()
                    )
                    outgoing = str(action.get("personId") or "")
                    if outgoing in lineups[side]:
                        lineups[side].remove(outgoing)
                    if incoming and incoming != "0":
                        lineups[side].add(incoming)

    result = {}
    for side in ("home", "away"):
        stats = by_team[side]
        fga, fta = stats["fga"], stats["fta"]
        for metric in ("fga", "three_pa", "fta", "fgm", "three_pm", "ftm"):
            result[f"{side}_{metric}"] = float(stats[metric])
        result[f"{side}_efg_pct"] = (stats["fgm"] + 0.5 * stats["three_pm"]) / fga if fga else 0.5
        result[f"{side}_three_pct"] = (
            stats["three_pm"] / stats["three_pa"] if stats["three_pa"] else 0.35
        )
        result[f"{side}_free_throw_pct"] = stats["ftm"] / fta if fta else 0.75
        result[f"{side}_true_shooting_pct"] = (
            (stats["fgm"] * 2 + stats["three_pm"] + stats["ftm"])
            / (2 * (fga + 0.44 * fta))
            if (fga + 0.44 * fta)
            else 0.5
        )
        minutes = player_minutes[side]
        total = sum(minutes.values())
        top = sorted(minutes.values(), reverse=True)[:5]
        result[f"{side}_top5_minutes_share"] = sum(top) / total if total else 0.0
        result[f"{side}_rotation_size"] = float(sum(value >= 5.0 for value in minutes.values()))
        result[f"{side}_top5_players"] = sorted(minutes, key=minutes.get, reverse=True)[:5]
    return result


def build_features(schedule: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    summaries = []
    for row in schedule.sort_values(["game_date", "game_id"]).itertuples(index=False):
        path = raw_dir / str(row.season) / f"{row.game_id}.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        summary = _game_summary(raw, str(row.home_team_id), str(row.away_team_id))
        summaries.append({"game_id": str(row.game_id), **summary})
    per_game = pd.DataFrame(summaries)
    per_game["game_id"] = per_game["game_id"].astype(str)
    per_game = per_game.sort_values("game_id")
    history: dict[str, list[dict]] = defaultdict(list)
    last_season: str | None = None
    rows = []
    for row in schedule.sort_values(["game_date", "game_id"]).itertuples(index=False):
        if str(row.season) != last_season:
            history.clear()
            last_season = str(row.season)
        current = per_game[per_game.game_id == str(row.game_id)]
        if current.empty:
            continue
        summary = current.iloc[0].to_dict()
        output = {"game_id": str(row.game_id)}
        for side, team_id in (
            ("home", str(row.home_team_id)), ("away", str(row.away_team_id))
        ):
            prior = history[team_id][-5:]
            metrics = (
                "efg_pct", "three_pct", "free_throw_pct", "true_shooting_pct",
                "top5_minutes_share", "rotation_size",
            )
            for metric in metrics:
                values = [item[metric] for item in prior]
                output[f"{side}_rolling_{metric}"] = (
                    float(sum(values) / len(values)) if values else 0.5
                )
            totals = {
                metric: sum(item[f"{metric}"] for item in prior)
                for metric in ("fga", "three_pa", "fta", "fgm", "three_pm", "ftm")
            }
            attempts = max(totals["fga"], 1.0)
            three_attempts = max(totals["three_pa"], 1.0)
            free_attempts = max(totals["fta"], 1.0)
            output[f"{side}_rolling_fga"] = attempts / max(len(prior), 1)
            output[f"{side}_rolling_three_pa"] = three_attempts / max(len(prior), 1)
            output[f"{side}_rolling_fta"] = free_attempts / max(len(prior), 1)
            # Empirical-Bayes shrinkage toward league-average priors. The
            # pseudo-counts stabilize early-season percentages.
            output[f"{side}_shrunk_efg_pct"] = (
                (totals["fgm"] + 0.5 * totals["three_pm"] + 2.6)
                / (attempts + 5.0)
            )
            output[f"{side}_shrunk_three_pct"] = (
                (totals["three_pm"] + 1.8) / (three_attempts + 5.0)
            )
            output[f"{side}_shrunk_free_throw_pct"] = (
                (totals["ftm"] + 3.75) / (free_attempts + 5.0)
            )
            if len(prior) >= 2:
                output[f"{side}_lineup_continuity"] = len(
                    set(prior[-1]["top5_players"]) & set(prior[-2]["top5_players"])
                ) / 5.0
            else:
                output[f"{side}_lineup_continuity"] = 0.0
            history[team_id].append(
                {
                    metric: summary[f"{side}_{metric}"] for metric in metrics
                }
                | {
                    metric: summary[f"{side}_{metric}"]
                    for metric in ("fga", "three_pa", "fta", "fgm", "three_pm", "ftm")
                }
                | {"top5_players": summary[f"{side}_top5_players"]}
            )
        for metric in (
            "rolling_efg_pct", "rolling_three_pct", "rolling_free_throw_pct",
            "rolling_true_shooting_pct", "shrunk_efg_pct", "shrunk_three_pct",
            "shrunk_free_throw_pct", "rolling_fga", "rolling_three_pa", "rolling_fta",
        ):
            output[f"shooting_diff_{metric}"] = (
                output[f"home_{metric}"] - output[f"away_{metric}"]
            )
        rows.append(output)
    return pd.DataFrame(rows)


def main() -> None:
    schedule = pd.read_parquet("data/schedule/regular_season_schedule.parquet")
    features = build_features(schedule, Path("data/raw"))
    base = pd.read_parquet("data/features/features_schedule.parquet")
    output = base.merge(features, on="game_id", how="left", validate="many_to_one")
    new_cols = [column for column in features.columns if column != "game_id"]
    if output[new_cols].isna().any().any():
        raise ValueError("missing player/team features")
    output.to_parquet("data/features/features_schedule_player.parquet", index=False)
    print(json.dumps({
        "games": int(features.game_id.nunique()),
        "rows": len(output),
        "features": new_cols,
    }))


if __name__ == "__main__":
    main()
