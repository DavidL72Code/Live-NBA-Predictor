"""Diffusion-informed basis expansion for the live win-probability model.

A basketball scoring margin behaves like a random walk, so the win probability
at any moment is approximately ``Phi(lead / sigma(t))`` where ``sigma(t)``
shrinks as the clock runs down. A linear model on raw ``score_diff`` and
``seconds_remaining`` cannot represent that surface; a linear model on the
ratios below can.

The expansion also drops the pre-game record features. They are strictly
redundant with Elo once the game is underway and measurably hurt out-of-sample
log loss (see ``artifacts/live_no_pregame_xgb_logistic.json``), while the Elo
edge is kept and explicitly decayed toward zero as time expires.

``expand_live_basis`` accepts any frame containing ``FEATURE_COLS`` and is used
as a ``FunctionTransformer`` step inside the served pipeline, so training and
serving share one definition of the model matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Live game state plus venue form and Elo — everything in FEATURE_COLS except
# the pre-game win/margin/streak block.
LIVE_BASE_COLS = [
    "seconds_remaining",
    "seconds_elapsed",
    "score_diff",
    "score_diff_norm",
    "run_home",
    "run_away",
    "run_diff",
    "is_overtime",
    "home_venue_win_pct",
    "home_venue_avg_margin",
    "away_venue_win_pct",
    "away_venue_avg_margin",
    "home_elo_rating",
    "away_elo_rating",
]

DERIVED_COLS = [
    "lead_over_t",
    "log_t",
    "lead_x_log_t",
    "abs_lead_over_sqrt_t",
    "sign_lead",
    "elo_diff_decayed",
    "run_diff_over_sqrt_t",
]

BASIS_COLS = [*LIVE_BASE_COLS, *DERIVED_COLS]


def expand_live_basis(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the model matrix for the live logistic win-probability model."""
    features = frame[LIVE_BASE_COLS].astype(float).copy()

    remaining = frame["seconds_remaining"].astype(float).to_numpy()
    lead = frame["score_diff"].astype(float).to_numpy()
    # +1 keeps every ratio finite at the final buzzer, where remaining is 0.
    sqrt_remaining = np.sqrt(remaining + 1.0)

    # score_diff_norm is already lead / sqrt(t); lead / t dominates it in the
    # last possessions, when a lead becomes close to insurmountable.
    features["lead_over_t"] = lead / (remaining + 1.0)
    features["log_t"] = np.log1p(remaining)
    features["lead_x_log_t"] = lead * np.log1p(remaining)
    # Magnitude and direction split apart so the model can learn a symmetric
    # confidence ramp without forcing it through a single signed coefficient.
    features["abs_lead_over_sqrt_t"] = np.abs(lead) / sqrt_remaining
    features["sign_lead"] = np.sign(lead)
    # A pre-game rating edge is worth less the less time remains to express it.
    elo_diff = (
        frame["home_elo_rating"].astype(float).to_numpy()
        - frame["away_elo_rating"].astype(float).to_numpy()
    )
    features["elo_diff_decayed"] = elo_diff * sqrt_remaining / 60.0
    features["run_diff_over_sqrt_t"] = (
        frame["run_diff"].astype(float).to_numpy() / sqrt_remaining
    )

    return features[BASIS_COLS]
