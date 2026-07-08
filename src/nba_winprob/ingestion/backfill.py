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


def iter_raw_game_files(raw_dir: Path) -> list[Path]:
    """All raw game JSON files under a backfill directory, sorted for determinism."""
    return sorted(raw_dir.rglob("*.json"))
