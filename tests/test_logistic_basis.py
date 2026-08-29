"""Tests for the diffusion basis and the logistic win-probability pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_winprob.analyst.serve import PROBABILITY_FLOOR, LogisticWinProbServer
from nba_winprob.features.basis import BASIS_COLS, expand_live_basis
from nba_winprob.schemas import FeatureVector
from nba_winprob.training.logistic import build_logistic_model, train_logistic
from nba_winprob.training.train import FEATURE_COLS, TARGET_COL


def _frame(rows: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    remaining = rng.uniform(0.0, 2880.0, rows)
    lead = rng.integers(-25, 26, rows).astype(float)
    frame = pd.DataFrame({
        "seconds_remaining": remaining,
        "seconds_elapsed": 2880.0 - remaining,
        "score_diff": lead,
        "score_diff_norm": lead / np.sqrt(remaining + 1.0),
        "run_home": rng.integers(0, 12, rows).astype(float),
        "run_away": rng.integers(0, 12, rows).astype(float),
        "is_overtime": np.zeros(rows),
        "home_win_pct": rng.uniform(0.2, 0.8, rows),
        "home_avg_margin": rng.normal(0, 5, rows),
        "home_streak": rng.integers(-5, 6, rows).astype(float),
        "away_win_pct": rng.uniform(0.2, 0.8, rows),
        "away_avg_margin": rng.normal(0, 5, rows),
        "away_streak": rng.integers(-5, 6, rows).astype(float),
        "home_venue_win_pct": rng.uniform(0.2, 0.8, rows),
        "home_venue_avg_margin": rng.normal(0, 5, rows),
        "away_venue_win_pct": rng.uniform(0.2, 0.8, rows),
        "away_venue_avg_margin": rng.normal(0, 5, rows),
        "home_elo_rating": rng.normal(1500, 60, rows),
        "away_elo_rating": rng.normal(1500, 60, rows),
    })
    frame["run_diff"] = frame["run_home"] - frame["run_away"]
    # Label the obvious signal so a fitted model is directionally sane.
    frame[TARGET_COL] = (frame["score_diff_norm"] + rng.normal(0, 0.2, rows) > 0).astype(int)
    return frame


def test_basis_is_finite_at_the_final_buzzer():
    frame = _frame()
    frame["seconds_remaining"] = 0.0
    expanded = expand_live_basis(frame)
    assert list(expanded.columns) == BASIS_COLS
    assert np.isfinite(expanded.to_numpy()).all()


def test_basis_drops_pregame_record_columns():
    expanded = expand_live_basis(_frame())
    assert not {"home_win_pct", "away_win_pct", "home_streak", "away_streak"} & set(
        expanded.columns
    )


def test_basis_needs_the_full_feature_contract():
    with pytest.raises(KeyError):
        expand_live_basis(_frame().drop(columns=["home_elo_rating"]))


def test_pipeline_round_trips_through_the_serving_layer(tmp_path):
    frame = _frame()
    path = tmp_path / "logistic.pkl"
    model = train_logistic(frame, path)

    server = LogisticWinProbServer.from_paths(path)
    features = [
        FeatureVector(
            game_id="0022400001",
            event_num=index,
            period=1,
            home_score=0,
            away_score=0,
            **{column: float(row[column]) for column in FEATURE_COLS},
        )
        for index, row in enumerate(frame.head(20).to_dict("records"))
    ]
    served = np.array(server.predict_batch(features))
    direct = model.predict_proba(frame.head(20))[:, 1]
    assert np.allclose(served, np.clip(direct, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR))


def test_served_probabilities_stay_off_the_bounds():
    frame = _frame()
    model = build_logistic_model()
    model.fit(frame, frame[TARGET_COL])
    server = LogisticWinProbServer.from_model(model)

    blowout = frame.head(5).copy()
    blowout["seconds_remaining"] = 0.0
    blowout["score_diff"] = 40.0
    blowout["score_diff_norm"] = 40.0
    features = [
        FeatureVector(
            game_id="0022400001",
            event_num=index,
            period=4,
            home_score=140,
            away_score=100,
            **{column: float(row[column]) for column in FEATURE_COLS},
        )
        for index, row in enumerate(blowout.to_dict("records"))
    ]
    probabilities = server.predict_batch(features)
    assert all(0.0 < p < 1.0 for p in probabilities)
