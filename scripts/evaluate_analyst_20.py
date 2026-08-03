"""Compare the opening XGBoost probability with the Gemini analyst on 20 games.

This is intentionally a small paired evaluation, not a replacement for the
untouched production test set. Both systems receive the same opening feature
state; the analyst also receives the official context that the product sends
to Gemini. Results are written to artifacts/analyst_eval_20.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from nba_winprob.api import server
from nba_winprob.analyst.analyst import LLMAnalyst
from nba_winprob.analyst.context import GameContext
from nba_winprob.config import get_settings
from nba_winprob.ingestion.news import fetch_team_news


def _log_loss(probability: float, outcome: int) -> float:
    p = min(max(probability, 1e-6), 1 - 1e-6)
    return -(outcome * math.log(p) + (1 - outcome) * math.log(1 - p))


def _metrics(rows: list[dict], probability_key: str) -> dict:
    brier = sum((row[probability_key] - row["home_win"]) ** 2 for row in rows) / len(rows)
    logloss = sum(_log_loss(row[probability_key], row["home_win"]) for row in rows) / len(rows)
    correct = sum((row[probability_key] >= 0.5) == bool(row["home_win"]) for row in rows)
    return {
        "brier": round(brier, 6),
        "log_loss": round(logloss, 6),
        "accuracy": round(correct / len(rows), 4),
        "correct": correct,
        "games": len(rows),
    }


def _cutoff(game_time_utc: str | None) -> datetime | None:
    if not game_time_utc:
        return None
    try:
        return datetime.fromisoformat(game_time_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _collect_games(end_date: date, target: int) -> list[dict]:
    games: list[dict] = []
    cursor = end_date
    while len(games) < target:
        slate = server._fetch_scoreboard(cursor.isoformat())
        games.extend(game for game in slate if game.get("status") == "Final")
        cursor -= timedelta(days=1)
        if (end_date - cursor).days > 45:
            break
    return games[:target]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--output", default="artifacts/analyst_eval_20.json")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.gemini_api_key:
        raise SystemExit("NBA_WINPROB_GEMINI_API_KEY is not configured")
    if not settings.analyst_mlflow_run_id:
        raise SystemExit("NBA_WINPROB_ANALYST_MLFLOW_RUN_ID is not configured")

    end_date = date.fromisoformat(args.end_date)
    games = _collect_games(end_date, args.games)
    if len(games) < args.games:
        raise SystemExit(f"Only found {len(games)} completed games in the search window")

    analyst = LLMAnalyst(api_key=settings.gemini_api_key, model=settings.analyst_model)
    rows: list[dict] = []
    failures: list[dict] = []

    for index, game in enumerate(games, start=1):
        game_id = game["game_id"]
        try:
            events = server._normalized_events(game_id)
            if not events:
                raise RuntimeError("no normalized play-by-play events")
            feature, recent_plays = server._replay_to_event(game_id, events[0].event_num)
            model_probability = server._predict(feature)
            if model_probability is None:
                raise RuntimeError("model probability unavailable")
            cutoff = _cutoff(game.get("game_time_utc"))
            home_team = game.get("home_team") or "Home"
            away_team = game.get("away_team") or "Away"
            # Do not call build_context here: for a completed game it fetches
            # the final boxscore, which would leak postgame player stats into
            # an opening-state evaluation.
            context = GameContext(
                feature=feature,
                model_prob=float(model_probability),
                home_team=home_team,
                away_team=away_team,
                recent_plays=recent_plays,
                news_items=fetch_team_news((home_team, away_team), cutoff),
            )
            output = analyst.analyze(context)
            home_win = int(game["home_score"] > game["away_score"])
            row = {
                "game_id": game_id,
                "game_date": game.get("game_date"),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "home_score": game.get("home_score"),
                "away_score": game.get("away_score"),
                "home_win": home_win,
                "model_probability": round(float(model_probability), 6),
                "analyst_probability": round(float(output.analyst_probability), 6),
                "analyst_direction": output.direction,
                "analyst_confidence": output.confidence,
                "analyst_headline": output.headline,
                "model_abs_error": round(abs(float(model_probability) - home_win), 6),
                "analyst_abs_error": round(abs(float(output.analyst_probability) - home_win), 6),
            }
            rows.append(row)
            print(
                f"[{index:02d}/{len(games)}] {row['away_team']} at {row['home_team']} "
                f"model={row['model_probability']:.1%} analyst={row['analyst_probability']:.1%} "
                f"actual={'HOME' if home_win else 'AWAY'}",
                flush=True,
            )
            # Keep the direct evaluation polite even though it bypasses HTTP rate limits.
            time.sleep(1.0)
        except Exception as exc:
            failures.append({"game_id": game_id, "error": str(exc)})
            print(f"[{index:02d}/{len(games)}] {game_id} FAILED: {exc}", flush=True)

    if len(rows) < args.games:
        raise SystemExit(f"Only completed {len(rows)}/{args.games} analyst evaluations")

    model_metrics = _metrics(rows, "model_probability")
    analyst_metrics = _metrics(rows, "analyst_probability")
    analyst_better = sum(row["analyst_abs_error"] < row["model_abs_error"] for row in rows)
    model_better = sum(row["model_abs_error"] < row["analyst_abs_error"] for row in rows)
    ties = len(rows) - analyst_better - model_better
    result = {
        "evaluation": "20-game paired opening-state comparison",
        "end_date": args.end_date,
        "analyst_model": settings.analyst_model,
        "note": "Small diagnostic sample; not a replacement for the untouched production test set.",
        "model": model_metrics,
        "analyst": analyst_metrics,
        "paired_absolute_error": {
            "analyst_better": analyst_better,
            "model_better": model_better,
            "tie": ties,
        },
        "games": rows,
        "failures": failures,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\nMODEL", json.dumps(model_metrics))
    print("ANALYST", json.dumps(analyst_metrics))
    print("PAIRED", json.dumps(result["paired_absolute_error"]))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
