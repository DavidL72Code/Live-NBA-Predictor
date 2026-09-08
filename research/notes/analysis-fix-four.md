# Analysis Fix Four — Team-Specific Home/Road Splits and Elo Strength

## The Problem

The model had overall team record, average margin, and streak, but it did not
know whether a team was better at home or on the road. It also had no
opponent-adjusted strength signal. That made a league-wide home advantage easy
to learn, while team-specific venue behavior was invisible.

The AI analyst had the same limitation: it could discuss home court in generic
NBA terms, but it was not given a team-specific home/road statistic.

## The Fix

Pregame context now includes the following leakage-safe features:

- `home_venue_win_pct` — home team win rate in prior home games
- `home_venue_avg_margin` — home team average margin in prior home games
- `away_venue_win_pct` — away team win rate in prior road games
- `away_venue_avg_margin` — away team average margin in prior road games
- `home_elo_rating` — home team opponent-adjusted Elo-style rating
- `away_elo_rating` — away team opponent-adjusted Elo-style rating

Each value is recorded before the target game. Venue splits use home games for
the home team and road games for the away team. Elo starts at 1500, applies a
65-point home-court adjustment, and updates only after a completed game.

The same context is propagated through offline feature generation, historical
serving, and the analyst prompt so training and inference use the same fields.

## Benchmark Setup

The repository contains 4,919 games across four seasons. Baseline and enhanced
models used the same 80/20 game-level split with `random_state=42`.

The full-timeline comparison used the same raw XGBoost configuration for both
feature sets. The opening-state comparison selected the first feature row from
each game, which isolates the new pregame signals.

## Results

### Full timeline

| Metric | Existing features | Venue + Elo features | Change |
|---|---:|---:|---:|
| **Brier score** | 0.162001 | **0.161723** | **0.17% lower** |
| Log loss | 0.479683 | **0.479517** | 0.03% lower |
| ROC-AUC | 0.837701 | **0.838281** | +0.000580 |

### Opening state

| Metric | Existing features | Venue + Elo features | Change |
|---|---:|---:|---:|
| **Brier score** | 0.240892 | **0.239676** | **0.51% lower** |
| Log loss | 0.796948 | **0.771766** | 3.16% lower |
| ROC-AUC | 0.595509 | **0.623939** | +0.028430 |

The opening-state lift is materially larger than the full-timeline lift, which
is expected: once a game has a score and recent-run history, those live signals
dominate the prediction. Venue and Elo context matter most before the game has
produced much evidence.

## Trained Model

The enhanced feature table was retrained and logged to the local MLflow store
as run `cba0bbd311c643598e11bd480ac5b5b5` using the same calibrated training
pipeline and a 100-tree XGBoost candidate. The local backend configuration now
points at this run, and the server was restarted after activation.

## Comparison With Earlier Fixes

The earlier notes report a full-timeline raw Brier score of 0.1687. That result
was produced with a different training configuration and evaluation setup, so
it is not a direct apples-to-apples comparison with the 100-tree matched
benchmark above. The controlled baseline comparison is the relevant result:
adding venue splits and Elo improved Brier score, log loss, and ROC-AUC on the
same games and split.

## Limitations

This fix still does not include trades, signings, injuries, starting lineups,
or player availability. It measures team performance entering the game, not
the exact roster available that night. Those signals require a dated roster
and injury feed and a separate backtest to avoid future-information leakage.
