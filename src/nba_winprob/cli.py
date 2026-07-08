"""Command-line entry points.

    nba-winprob backfill --seasons 2022-23 2023-24
    nba-winprob build-features --raw-dir data/raw --output data/features/features.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def cmd_backfill(args: argparse.Namespace) -> int:
    from nba_winprob.ingestion.backfill import backfill_season
    from nba_winprob.ingestion.client import NBAStatsClient

    client = NBAStatsClient(min_request_interval=args.interval)
    for season in args.seasons:
        counts = backfill_season(
            client,
            season,
            output_dir=Path(args.output),
            season_type=args.season_type,
        )
        print(f"{season}: {counts}")
    return 0


def cmd_build_features(args: argparse.Namespace) -> int:
    """Normalize raw payloads and compute the offline feature table.

    Uses the same GameState accumulator as the (future) streaming path, so
    this parquet is by construction consistent with online features.
    """
    import pandas as pd

    from nba_winprob.features import compute_game_features, game_label
    from nba_winprob.ingestion.backfill import iter_raw_game_files
    from nba_winprob.ingestion.normalize import SchemaDriftError, normalize_playbyplay

    raw_files = iter_raw_game_files(Path(args.raw_dir))
    if not raw_files:
        print(f"no raw game files under {args.raw_dir}", file=sys.stderr)
        return 1

    frames = []
    skipped = 0
    for path in raw_files:
        try:
            events = normalize_playbyplay(json.loads(path.read_text()))
            vectors = compute_game_features(events, run_window_seconds=args.run_window)
            label = game_label(events)
        except (SchemaDriftError, ValueError) as exc:
            logger.warning("skipping %s: %s", path.name, exc)
            skipped += 1
            continue
        frame = pd.DataFrame([v.model_dump() for v in vectors])
        frame["home_win"] = label
        frames.append(frame)

    if not frames:
        print("all games failed to normalize", file=sys.stderr)
        return 1

    table = pd.concat(frames, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    print(
        f"wrote {len(table)} feature rows from {len(frames)} games to {output}"
        + (f" ({skipped} games skipped)" if skipped else "")
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from nba_winprob.config import get_settings

    settings = get_settings()

    parser = argparse.ArgumentParser(prog="nba-winprob")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backfill = subparsers.add_parser("backfill", help="download raw play-by-play for seasons")
    backfill.add_argument("--seasons", nargs="+", required=True, metavar="YYYY-YY")
    backfill.add_argument("--output", default=str(settings.raw_data_dir))
    backfill.add_argument("--season-type", default="Regular Season")
    backfill.add_argument(
        "--interval",
        type=float,
        default=settings.min_request_interval,
        help="min seconds between NBA.com requests",
    )
    backfill.set_defaults(func=cmd_backfill)

    build = subparsers.add_parser(
        "build-features", help="normalize raw games and write the offline feature table"
    )
    build.add_argument("--raw-dir", default=str(settings.raw_data_dir))
    build.add_argument("--output", default="data/features/features.parquet")
    build.add_argument(
        "--run-window", type=float, default=180.0, help="scoring-run window in game seconds"
    )
    build.set_defaults(func=cmd_build_features)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
