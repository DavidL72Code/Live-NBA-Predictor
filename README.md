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

## Model

The live win probability comes from **logistic regression on a diffusion
basis** — a linear model fed ratios that describe a scoring margin as a random
walk.

A basketball margin behaves like a random walk, so the win probability at any
moment tracks the lead divided by the time still left to overturn it. A linear
model given raw `score_diff` and `seconds_remaining` as separate inputs cannot
represent that surface. Given the ratios directly, it can.
[`features/basis.py`](src/nba_winprob/features/basis.py) builds them and
[`training/logistic.py`](src/nba_winprob/training/logistic.py) wraps the whole
thing — basis expansion, scaling, and the estimator — in one scikit-learn
pipeline, so the served pickle carries its own feature engineering and the
serving layer keeps passing a plain `FEATURE_COLS` frame.

**21 inputs:** live game state, venue form, and Elo, plus seven derived terms
(`lead_over_t`, `log_t`, `lead_x_log_t`, `abs_lead_over_sqrt_t`, `sign_lead`,
`elo_diff_decayed`, `run_diff_over_sqrt_t`). Pre-game win/loss records are
dropped as redundant with Elo. **No calibrator** — fitted temperatures land at
~1.0 across folds, so there is nothing to correct, and isotonic makes it worse.

Trained on 6,149 games (2021-22 through 2025-26). Evaluated on rolling
chronological folds, 4,500 future games never seen in training:

| model | Brier | log loss | ROC-AUC |
|---|---|---|---|
| **logistic + diffusion basis** (shipped) | **0.1511** | **0.4515** | **0.861** |
| XGBoost, early-stopped | 0.1539 | 0.4593 | 0.856 |
| XGBoost, 400 trees + isotonic (previous) | 0.1656 | 0.4924 | 0.832 |

Paired game-level bootstrap vs the previous model: ΔBrier `[-0.0172, -0.0119]`.

### Why the previous model lost

It fitted 400 trees with no early stopping. Nested selection picks 66–106 trees
at every fold, so most of the gap was a hyperparameter rather than the
algorithm — early stopping alone recovers about three quarters of it. The
remaining quarter is the model class: at this data volume the extra flexibility
of a tree ensemble costs more in variance than it recovers in bias.

### Models tested

Same five-season protocol, directly comparable:

| candidate | Brier | verdict |
|---|---|---|
| logistic + diffusion basis | **0.1511** | **shipped** |
| logistic + basis + pre-game records | 0.1512 | tie, `[-0.00031, +0.00049]` |
| random forest on the basis | 0.1533 | worse, `[+0.00120, +0.00315]` |
| XGBoost, early-stopped | 0.1539 | worse |
| random forest, raw features | 0.1544 | worse, `[+0.00197, +0.00470]` |
| HistGradientBoosting, raw | 0.1562 | worse, `[+0.00351, +0.00678]` |
| HistGradientBoosting on the basis | 0.1563 | worse, `[+0.00367, +0.00677]` |
| XGBoost, 400 trees + isotonic | 0.1656 | worse (previous production) |

Random forest is the strongest tree model — averaging reduces variance, which is
the binding constraint at this data volume, whereas boosting accumulates it. It
is also the only tree model the diffusion basis helps (0.1544 → 0.1533); both
boosters are indifferent to it. Neither is enough to win.

Both boosting implementations independently chose **50–106 iterations** per fold
on game-disjoint holdouts. The retired production model used 400.

Tried against the basis on four seasons; none beat it:

| candidate | result |
|---|---|
| natural splines on the key ratios | tie |
| time-varying coefficients (every term × `log_t`) | tie |
| XGBoost residual boosting on the linear logit | tie; early stopping added **zero** trees in 2 of 6 folds |
| XGBoost trained *on* the basis features | worse — the parameterization does not rescue trees |
| isotonic / temperature calibration on top | worse |
| box-score features (shooting, rebounds, fouls, turnovers) | no help; team fouls significantly negative |

Earlier work benchmarked ExtraTrees, MLP, kNN, DART, monotonic XGBoost,
Poisson-Skellam, a negative-binomial margin model, Bayesian state models, and
per-horizon experts. All land in 0.149–0.160. Those runs used varying parquets and fold schemes, so
their absolute scores are **not** comparable to the tables above — see
`artifacts/*.json`.

### Data volume

A learning curve on identical validation blocks shows the logistic saturating
around 2,000 games while XGBoost keeps improving:

| training games | logistic | XGBoost | gap |
|---|---|---|---|
| 500 | 0.14882 | 0.16403 | +0.01521 |
| 2,000 | 0.14651 | 0.15057 | +0.00407 |
| 4,500 | 0.14635 | 0.14772 | +0.00137 |

Doubling from 2,000 to 4,500 games is worth 0.00016 to the logistic and 0.00285
to XGBoost. The gap has closed 91% but has not crossed. Extrapolation puts a
crossover around 9,000–14,000 games (8–11 seasons), and the nearest clean
seasons are 2018-19 and earlier — 2019-20 and 2020-21 are COVID-distorted, with
bubble and limited-crowd games that the venue and Elo features would learn from.

### Reproducing

```bash
.venv/bin/python scripts/validate_five_season.py      # the comparison table
.venv/bin/python scripts/benchmark_learning_curve.py  # the learning curve
.venv/bin/python scripts/benchmark_extra_features.py  # box-score feature test
.venv/bin/python scripts/benchmark_forest_gbm.py      # forest / gradient boosting
```

Results land in `artifacts/live_*.json`. A known caveat: an earlier feature
table left pre-game context at placeholder values for 25.9% of games. The
current builder fixes it, but artifacts predating
`artifacts/live_five_season_comparison.json` inherit the defect, so any older
conclusion involving pre-game features is unreliable.

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
NBA_WINPROB_ANALYST_MODEL_TYPE=logistic
NBA_WINPROB_ANALYST_LOGISTIC_MODEL_PATH=artifacts/live_logistic_model.pkl
NBA_WINPROB_ANALYST_MODEL_TYPE=logistic
NBA_WINPROB_ANALYST_LOGISTIC_MODEL_PATH=artifacts/live_logistic_model.pkl
NBA_WINPROB_ANALYST_MLFLOW_RUN_ID=...
NBA_WINPROB_MLFLOW_TRACKING_URI=sqlite:///mlflow.db
NBA_WINPROB_CORS_ALLOWED_ORIGINS=https://live-nba-predictor.vercel.app,http://127.0.0.1:8765,http://localhost:8765
```

Use the real Vercel app URL in `NBA_WINPROB_CORS_ALLOWED_ORIGINS`; otherwise
the browser will block frontend calls to the Hugging Face backend.

`ANALYST_MODEL_TYPE` defaults to `xgboost` in code. Without the two model
variables above, a deploy silently serves the old XGBoost stack (or no
predictions at all) instead of failing loudly.

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
