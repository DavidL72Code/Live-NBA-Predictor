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

## Deployment split

The app can be deployed as a static Vercel frontend with a Hugging Face Spaces
backend.

### Vercel frontend

Vercel uses [vercel.json](vercel.json) to build the static UI into `dist/`.
Set this Vercel environment variable (the build also has this as its default):

```bash
NBA_WINPROB_PUBLIC_API_BASE=https://davidl72code-swoosh-ai.hf.space
```

That value is written into `dist/config.js`, and the browser uses it for every
`/api/...` fetch and live SSE stream. Local FastAPI testing still uses
same-origin `http://127.0.0.1:8765/`.

### Hugging Face backend

Create a **Docker Space** named `SWOOSH_AI` and deploy this repository. The included
[Dockerfile](Dockerfile) starts FastAPI on port `7860`, which is the Hugging
Face Spaces web port.

Set these Hugging Face Space secrets or variables:

```bash
NBA_WINPROB_GEMINI_API_KEY=...
NBA_WINPROB_ANALYST_MLFLOW_RUN_ID=...
NBA_WINPROB_MLFLOW_TRACKING_URI=sqlite:///mlflow.db
NBA_WINPROB_CORS_ALLOWED_ORIGINS=https://live-nba-predictor.vercel.app,http://127.0.0.1:8765,http://localhost:8765
```

Use the real Vercel app URL in `NBA_WINPROB_CORS_ALLOWED_ORIGINS`; otherwise
the browser will block frontend calls to the Hugging Face backend.

### Render alternative

Render can run the same Dockerfile as a Web Service. Create a Web Service from
the repository, choose the Free instance for testing, and leave the Dockerfile
as the runtime. The container now uses Render's `PORT` automatically.

Set these Render environment variables:

```bash
NBA_WINPROB_GEMINI_API_KEY=...
NBA_WINPROB_ANALYST_MLFLOW_RUN_ID=...
NBA_WINPROB_MLFLOW_TRACKING_URI=sqlite:///mlflow.db
NBA_WINPROB_CORS_ALLOWED_ORIGINS=https://live-nba-predictor.vercel.app
NBA_WINPROB_STATS_PROXY_URL=https://live-nba-predictor.vercel.app
NBA_WINPROB_STATS_PROXY_TOKEN=the_same_random_value_as_vercel
```

Set `NBA_STATS_PROXY_TOKEN` to the same random value in Vercel. The proxy is
restricted to the NBA endpoints used by this app and is not an open relay.

Free Render services have 512 MB RAM, 0.1 CPU, sleep after 15 minutes without
traffic, and lose local filesystem changes when they restart. That makes Free
Render suitable for a demo, but not reliable for the live polling pipeline or
SQLite/MLflow persistence. Keep the trained model in deployable artifact
storage and use an external Redis/MLflow store, or move to a paid instance for
the full live backend.

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
