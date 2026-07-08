# NBA live win probability system — project plan

## Goal

Build a production-style streaming ML system that predicts live NBA win probability, updating possession-by-possession as a game unfolds. This is a portfolio piece aimed at demonstrating AI/ML engineering infrastructure skills (streaming, feature stores, training/serving consistency, CI/CD) rather than just a trained model in a notebook.

## Why this project

Most ML portfolios show a trained model. This project is meant to show the surrounding infrastructure that makes ML reliable in production:

- Real-time feature computation from a live event stream
- Training-serving consistency (same feature logic offline and online)
- A feature store with online (low-latency) and offline (historical) paths
- Monitoring for calibration and drift
- A CI/CD pipeline that validates against replayed real games before deploying

## Data source

- `nba_api` (Python, free, no auth required) — pulls play-by-play, boxscores, and live game data from stats.nba.com
- Known risk: unofficial/unofficial-adjacent package that scrapes NBA.com; endpoints can change without notice. Pin versions and write integration tests that catch schema drift early.
- Historical data (multiple seasons of play-by-play) is available now for backfill and training; live data is only testable when real games are being played.

## Architecture

Layers, in order of data flow:

1. **Ingestion** — Python producer polls `nba_api` live endpoints during games, publishes play-by-play events
2. **Event bus** — Kafka or Redpanda receives events as they happen
3. **Stream processor** — computes rolling, time-aware features: score differential adjusted for time remaining, recent scoring runs, possession efficiency, foul trouble / bonus state
4. **Feature store**
   - Online: Redis — current live feature vector per game, low-latency reads for serving
   - Offline: Postgres + Parquet — historical features for training, computed with the *same* logic as the online path to avoid training-serving skew
5. **Model serving** — FastAPI service, pulls current features from Redis, returns live win probability. Model: XGBoost/LightGBM initially, trained on historical games from the offline store, tracked with MLflow
6. **Frontend** — React dashboard showing live win probability as a chart during a game
7. **Monitoring** — calibration tracking (Brier score, "of all times model said X% win probability, did that side actually win X% of the time"), feature drift checks between training-era and live distributions

## Build roadmap (coding effort compressed via Claude Code)

| Phase | Scope | Estimate |
|---|---|---|
| 1. Historical data & backfill | `nba_api` ingestion scripts, pull multi-season play-by-play, store raw data | Days 1-4 |
| 2. Streaming pipeline | Kafka/Redpanda setup, stream processor computing rolling features | Days 5-10 |
| 3. Feature store & training | Redis online store, Postgres/Parquet offline store, XGBoost training pipeline with MLflow | Days 11-15 |
| 4. Model serving & dashboard | FastAPI serving layer, React live dashboard | Days 16-20 |
| 5. Monitoring & CI/CD | Calibration/drift monitoring, GitHub Actions pipeline | Days 21-25 |
| 6. Live validation gate | Validate end-to-end against real live games | Gated by NBA calendar, not code — see note below |

**Important constraint:** phases 1-5 are pure coding and compress heavily with AI-assisted development. Phase 6 cannot be accelerated the same way — it requires actual live NBA games to test against. As of this plan, NBA Summer League (July) is the nearest live-testing window; the regular season starts in October. Decide early whether to target Summer League for a faster (but lower-volume) validation pass, or wait for preseason/regular season for more testing volume.

## CI/CD pipeline

1. **Push code & CI checks** — GitHub Actions runs lint + unit tests on every push
2. **Build & push image** — Docker image built and pushed to a registry
3. **Deploy to staging** — staging environment replays historical games through the pipeline as integration tests (this is the key validation step, since live games aren't available on demand)
4. **Deploy to production** — blue-green release
5. **Monitor & rollback** — automatic rollback triggered by failed calibration/drift checks post-deploy

## Open decisions / things to revisit

- Exact feature set for the stream processor (start simple: score differential + time remaining + recent scoring run; expand later)
- Whether to add lineup-based features (on/off court impact) — likely a phase 2 extension, not v1
- Whether to target NBA Summer League for early live validation or wait for the regular season
- Model choice: starting with gradient-boosted trees (XGBoost/LightGBM) for interpretability and speed; a temporal model (e.g. simple RNN) could be a later iteration if time allows

## Notes for Claude Code

- This is a long-term, ongoing project (not a weekend build) intended as a job-search portfolio piece.
- Prioritize clean, well-tested infrastructure over model sophistication — the differentiator here is the pipeline, not squeezing out marginal accuracy.
- `nba_api` has no authentication; be mindful of polling frequency during live games to avoid hammering NBA.com endpoints.
- Keep feature computation logic in a single shared module/package that both the streaming path and the offline batch path import from, to guarantee training-serving consistency — don't duplicate feature logic between online and offline code paths.
