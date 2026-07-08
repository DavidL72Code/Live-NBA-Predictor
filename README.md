# NBA Live Win Probability

A production-style streaming ML system that predicts live NBA win probability,
updating possession-by-possession. The point of the project is the
infrastructure around the model — streaming features, a feature store with
consistent online/offline paths, calibration monitoring, and CI/CD with
replay-based validation. Full design: [nba-win-probability-plan.md](nba-win-probability-plan.md).

## Status

**Phase 1 (historical data & backfill)** — in progress. The base currently includes:

- `nba_winprob.schemas` — canonical `GameEvent` / `FeatureVector` models
- `nba_winprob.gametime` — game clock math (elapsed/remaining, OT handling)
- `nba_winprob.features` — `GameState` incremental accumulator; the *same*
  class serves the streaming path and the offline batch path, which is what
  guarantees training-serving consistency
- `nba_winprob.ingestion` — rate-limited `nba_api` client, PlayByPlayV3
  normalizer with schema-drift detection, resume-safe season backfill.
  (V3, not V2: while building this we found the V2 endpoint now returns
  empty payloads — exactly the endpoint-drift risk the plan calls out.)
- `nba_winprob.cli` — `backfill` and `build-features` commands

Later phases (event bus, stream processor, Redis/Postgres feature store,
FastAPI serving, React dashboard, monitoring) build on this base — see the plan.

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration & secrets

All runtime config comes from environment variables prefixed `NBA_WINPROB_`,
loaded via [src/nba_winprob/config.py](src/nba_winprob/config.py) (which also
reads a local `.env`). Copy `.env.example` to `.env` and adjust as needed.

Rules:

- `.env` is gitignored; only `.env.example` (placeholders) is committed.
- Never hardcode connection strings or credentials — add a field to
  `Settings` and an entry to `.env.example` instead.
- Never paste real secret values into chats, issues, commits, or logs.
- Phase 1 needs no credentials at all (`nba_api` is unauthenticated); the
  Kafka/Redis/Postgres/MLflow entries are placeholders for later phases.

## Usage

```bash
# Download raw play-by-play for full seasons (resume-safe; polite 1 req/s)
nba-winprob backfill --seasons 2022-23 2023-24

# Normalize + compute the offline feature table (parquet)
nba-winprob build-features --raw-dir data/raw --output data/features/features.parquet
```

Raw payloads land in `data/raw/<season>/<game_id>.json` verbatim, so
normalization and feature changes can be re-run without re-hitting NBA.com.

## Tests

```bash
pytest
```

Notable tests:

- `tests/test_features.py::TestTrainingServingConsistency` — replaying a game
  event-by-event ("online") must produce byte-identical feature vectors to the
  batch path.
- `tests/test_normalize.py` — the normalizer raises `SchemaDriftError` naming
  missing columns when the unofficial stats.nba.com schema drifts, so CI
  catches endpoint changes early.

## Design rules

- **One feature implementation.** All feature logic lives in
  `nba_winprob/features/compute.py`. The streaming processor and the offline
  builder both import it; never fork the logic.
- **Normalize once, at the boundary.** Only `ingestion/normalize.py` knows
  raw nba_api shapes. Everything downstream consumes `GameEvent`.
- **Be polite to NBA.com.** All requests go through `NBAStatsClient`, which
  enforces a minimum request interval and bounded retries.
