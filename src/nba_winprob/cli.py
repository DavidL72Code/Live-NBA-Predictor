"""Command-line entry points.

    nba-winprob backfill --seasons 2022-23 2023-24
    nba-winprob build-features --raw-dir data/raw --output data/features/features.parquet
    nba-winprob produce        # live: poll NBA → publish events to Kafka
    nba-winprob process        # live: consume events → compute features → write Redis
    nba-winprob train          # train XGBoost model, log to MLflow
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def cmd_backfill(args: argparse.Namespace) -> int:
    from nba_winprob.ingestion.backfill import backfill_season, backfill_team_context
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
        n = backfill_team_context(
            client,
            season,
            output_dir=Path(args.output),
            season_type=args.season_type,
        )
        print(f"{season}: team context for {n} games")
    return 0


def cmd_build_features(args: argparse.Namespace) -> int:
    """Normalize raw payloads and compute the offline feature table.

    Writes to parquet always. Also writes to Postgres when
    NBA_WINPROB_POSTGRES_DSN is set (creates the table if needed).
    """
    import pandas as pd

    from nba_winprob.config import get_settings
    from nba_winprob.features import compute_game_features, game_label
    from nba_winprob.ingestion.backfill import iter_raw_game_files
    from nba_winprob.ingestion.normalize import SchemaDriftError, normalize_playbyplay

    raw_files = iter_raw_game_files(Path(args.raw_dir))
    if not raw_files:
        print(f"no raw game files under {args.raw_dir}", file=sys.stderr)
        return 1

    offline_store = None
    if get_settings().postgres_dsn:
        from nba_winprob.store.offline import OfflineStore

        offline_store = OfflineStore()
        offline_store.ensure_schema()
        logger.info("Postgres offline store connected")

    # Load team context files lazily per season directory
    from nba_winprob.schemas import TeamGameContext

    season_contexts: dict[str, dict] = {}

    def _load_season_context(season_dir: Path) -> dict:
        key = str(season_dir)
        if key not in season_contexts:
            ctx_file = season_dir / "team_context.json"
            season_contexts[key] = json.loads(ctx_file.read_text()) if ctx_file.exists() else {}
        return season_contexts[key]

    frames = []
    skipped = 0
    for path in raw_files:
        try:
            events = normalize_playbyplay(json.loads(path.read_text()))
            game_id = events[0].game_id if events else None
            game_ctx = _load_season_context(path.parent).get(game_id, {}) if game_id else {}
            home_context = TeamGameContext(**game_ctx["home"]) if "home" in game_ctx else None
            away_context = TeamGameContext(**game_ctx["away"]) if "away" in game_ctx else None
            vectors = compute_game_features(
                events,
                run_window_seconds=args.run_window,
                home_context=home_context,
                away_context=away_context,
            )
            label = game_label(events)
        except (SchemaDriftError, ValueError) as exc:
            logger.warning("skipping %s: %s", path.name, exc)
            skipped += 1
            continue
        if offline_store is not None:
            offline_store.write_batch(vectors, home_win=label)
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


def cmd_produce(args: argparse.Namespace) -> int:
    from nba_winprob.streaming.producer import LiveProducer

    LiveProducer(poll_interval=args.interval).run()
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    from nba_winprob.streaming.processor import StreamProcessor

    StreamProcessor().run()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"Dashboard running at http://{args.host}:{args.port}")
    uvicorn.run(
        "nba_winprob.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """One-shot LLM analyst: game_id → model prob → structured LLM judgment."""
    from nba_winprob.analyst.analyst import LLMAnalyst
    from nba_winprob.analyst.context import build_context
    from nba_winprob.analyst.serve import WinProbServer
    from nba_winprob.config import get_settings
    from nba_winprob.ingestion.client import NBAStatsClient

    settings = get_settings()
    game_id = args.game_id

    # ── 1. Feature vector ──────────────────────────────────────────────────
    feature = None
    try:
        from nba_winprob.store.online import OnlineStore

        feature = OnlineStore().read(game_id)
    except Exception as exc:
        logger.debug("Redis unavailable, will fall back to NBA API: %s", exc)

    if feature is None:
        logger.info("feature not in Redis — computing from live play-by-play")
        client_tmp = NBAStatsClient()
        from nba_winprob.features.compute import GameState
        from nba_winprob.ingestion.normalize import normalize_playbyplay

        raw = client_tmp.get_play_by_play_raw(game_id)
        events = normalize_playbyplay(raw)
        if not events:
            print(f"no events found for game {game_id}", file=sys.stderr)
            return 1
        state = GameState(game_id)
        for event in events:
            feature = state.update(event)

    if feature is None:
        print(f"could not obtain a feature vector for game {game_id}", file=sys.stderr)
        return 1

    # ── 2. Model ───────────────────────────────────────────────────────────
    run_id = args.run_id or settings.analyst_mlflow_run_id
    if run_id:
        server = WinProbServer.from_mlflow(run_id, tracking_uri=settings.mlflow_tracking_uri)
    elif args.model_path and args.calibrator_path:
        server = WinProbServer.from_paths(args.model_path, args.calibrator_path)
    else:
        print(
            "no model source — set --run-id, --model-path/--calibrator-path, "
            "or NBA_WINPROB_ANALYST_MLFLOW_RUN_ID",
            file=sys.stderr,
        )
        return 1

    prob = server.predict(feature)

    # ── 3. Boxscore + context ──────────────────────────────────────────────
    client = NBAStatsClient()
    context = build_context(game_id, feature, prob, client)

    # ── 4. LLM analysis ───────────────────────────────────────────────────
    api_key = args.api_key or settings.gemini_api_key
    if not api_key:
        print(
            "no Gemini API key — set --api-key or NBA_WINPROB_GEMINI_API_KEY",
            file=sys.stderr,
        )
        return 1

    analyst = LLMAnalyst(api_key=api_key, model=settings.analyst_model)
    output = analyst.analyze(context)

    # ── 5. Print ───────────────────────────────────────────────────────────
    sep = "=" * 60
    fv = feature
    period_label = f"Q{fv.period}" if fv.period <= 4 else f"OT{fv.period - 4}"
    mins_left = fv.seconds_remaining / 60.0

    print(f"\n{sep}")
    print(f"GAME  {context.home_team} {fv.home_score} – {fv.away_score} {context.away_team}")
    print(f"      {period_label}, {mins_left:.1f} min remaining")
    print(sep)
    print(f"\n{output.headline}\n")
    print(f"MODEL :   {output.model_probability:.1%}  (home win)")
    print(f"ANALYST:  {output.analyst_probability:.1%}  ({output.direction.upper()}, {output.confidence} confidence)\n")
    print("KEY FACTORS:")
    for factor in output.key_factors:
        print(f"  • {factor}")
    print(f"\nREASONING:\n{output.reasoning}\n")

    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from nba_winprob.training.train import load_parquet, load_postgres, train

    if args.source == "postgres":
        from nba_winprob.config import get_settings

        dsn = get_settings().postgres_dsn
        if not dsn:
            print("NBA_WINPROB_POSTGRES_DSN not set", file=sys.stderr)
            return 1
        df = load_postgres(dsn, seasons=args.seasons or None)
    else:
        df = load_parquet(args.parquet)

    run = train(
        df,
        experiment_name=args.experiment,
        run_name=args.run_name,
        test_size=args.test_size,
    )
    print(f"MLflow run: {run.info.run_id}")
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

    produce = subparsers.add_parser(
        "produce", help="poll live NBA games and publish events to Kafka"
    )
    produce.add_argument(
        "--interval",
        type=float,
        default=settings.live_poll_interval,
        help="seconds between live scoreboard polls",
    )
    produce.set_defaults(func=cmd_produce)

    process = subparsers.add_parser(
        "process", help="consume game events from Kafka and write feature vectors to Redis"
    )
    process.set_defaults(func=cmd_process)

    serve_cmd = subparsers.add_parser(
        "serve", help="start the web dashboard (FastAPI + Uvicorn)"
    )
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8765)
    serve_cmd.add_argument("--reload", action="store_true", default=False)
    serve_cmd.set_defaults(func=cmd_serve)

    analyze_cmd = subparsers.add_parser(
        "analyze",
        help="run the LLM analyst on a live game (model prob + qualitative judgment)",
    )
    analyze_cmd.add_argument("game_id", help="NBA game ID (e.g. 0022300001)")
    analyze_cmd.add_argument(
        "--run-id", default=None,
        help="MLflow run ID of the trained model (overrides NBA_WINPROB_ANALYST_MLFLOW_RUN_ID)",
    )
    analyze_cmd.add_argument("--model-path", default=None, help="path to XGBoost model file")
    analyze_cmd.add_argument(
        "--calibrator-path", default=None, help="path to isotonic calibrator pickle"
    )
    analyze_cmd.add_argument(
        "--api-key", default=None,
        help="Anthropic API key (overrides NBA_WINPROB_ANTHROPIC_API_KEY)",
    )
    analyze_cmd.set_defaults(func=cmd_analyze)

    train_cmd = subparsers.add_parser(
        "train", help="train XGBoost win-probability model and log to MLflow"
    )
    train_cmd.add_argument(
        "--source", choices=["parquet", "postgres"], default="parquet",
        help="where to read training data from",
    )
    train_cmd.add_argument(
        "--parquet", default="data/features/features.parquet",
        help="path to parquet file (used when --source=parquet)",
    )
    train_cmd.add_argument(
        "--seasons", nargs="+", metavar="YYYY-YY",
        help="filter by seasons (used when --source=postgres)",
    )
    train_cmd.add_argument("--experiment", default="nba-winprob")
    train_cmd.add_argument("--run-name", default=None)
    train_cmd.add_argument("--test-size", type=float, default=0.2)
    train_cmd.set_defaults(func=cmd_train)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
