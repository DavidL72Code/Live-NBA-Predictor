"""Historical backfill: pull raw play-by-play for whole seasons to disk.

Raw payloads are stored verbatim (one JSON per game under
``<output>/<season>/<game_id>.json``) so normalization can be re-run later
without re-hitting NBA.com. Backfill is resume-safe: existing files are
skipped, so an interrupted run just picks up where it left off.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from nba_winprob.ingestion.client import NBAStatsClient
from nba_winprob.ingestion.team_context import build_season_context

logger = logging.getLogger(__name__)


def backfill_season(
    client: NBAStatsClient,
    season: str,
    output_dir: Path,
    season_type: str = "Regular Season",
    skip_existing: bool = True,
) -> dict[str, int]:
    """Download raw play-by-play for every game in a season.

    Returns counts: {"downloaded": n, "skipped": n, "failed": n}.
    Individual game failures are logged and skipped rather than aborting the
    whole season; rerun to retry them.
    """
    season_dir = output_dir / season.replace("/", "-")
    season_dir.mkdir(parents=True, exist_ok=True)

    game_ids = client.get_season_game_ids(season, season_type=season_type)
    logger.info("%s %s: %d games", season, season_type, len(game_ids))

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    for i, game_id in enumerate(game_ids, 1):
        path = season_dir / f"{game_id}.json"
        if skip_existing and path.exists():
            counts["skipped"] += 1
            continue
        try:
            payload = client.get_play_by_play_raw(game_id)
        except RuntimeError:
            logger.exception("giving up on game %s", game_id)
            counts["failed"] += 1
            continue
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload))
        tmp_path.rename(path)  # atomic: no half-written files if interrupted
        counts["downloaded"] += 1
        if i % 50 == 0 or i == len(game_ids):
            logger.info("%s: %d/%d games processed %s", season, i, len(game_ids), counts)
    return counts


def backfill_team_context(
    client: NBAStatsClient,
    season: str,
    output_dir: Path,
    season_type: str = "Regular Season",
) -> int:
    """Fetch league game log and save pre-game team context for a season.

    Writes ``<output_dir>/<season>/team_context.json``.  Existing files are
    overwritten so the context is always up-to-date with the full season.
    Returns the number of games with complete (home + away) context.
    """
    season_dir = output_dir / season.replace("/", "-")
    season_dir.mkdir(parents=True, exist_ok=True)
    context_path = season_dir / "team_context.json"

    rows = client.get_league_game_log(season, season_type=season_type)
    context = build_season_context(rows)

    serializable = {
        game_id: {role: ctx.model_dump() for role, ctx in roles.items()}
        for game_id, roles in context.items()
    }
    context_path.write_text(json.dumps(serializable))
    complete = sum(1 for v in context.values() if "home" in v and "away" in v)
    logger.info(
        "%s: wrote team context for %d games (%d complete) to %s",
        season, len(context), complete, context_path,
    )
    return complete


def iter_raw_game_files(raw_dir: Path) -> list[Path]:
    """All raw game JSON files under a backfill directory, sorted for determinism."""
    return sorted(p for p in raw_dir.rglob("*.json") if p.name != "team_context.json")
