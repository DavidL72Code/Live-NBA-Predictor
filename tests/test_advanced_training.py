"""Small, dependency-light checks for the advanced benchmark utilities."""

import numpy as np
import pandas as pd

from nba_winprob.training.advanced import BetaCalibrator, walk_forward_folds
from nba_winprob.training.train import FEATURE_COLS


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for season_start in range(2021, 2025):
        for game_num in range(4):
            game_id = f"002{str(season_start)[-2:]}{game_num:07d}"
            label = game_num % 2
            for event_num in range(3):
                row = {column: float(rng.normal()) for column in FEATURE_COLS}
                row.update({
                    "game_id": game_id,
                    "event_num": event_num + 1,
                    "home_win": label,
                    "home_score": event_num + label * 2,
                    "away_score": event_num + (1 - label) * 2,
                })
                rows.append(row)
    return pd.DataFrame(rows)


def test_walk_forward_folds_keep_future_seasons_out_of_training():
    frame = _frame()
    folds = walk_forward_folds(frame, min_train_seasons=2)
    assert len(folds) == 2
    for train_idx, validation_idx, _, validation_season in folds:
        train_games = frame.iloc[train_idx]
        validation_games = frame.iloc[validation_idx]
        assert train_games.game_id.isin(validation_games.game_id).sum() == 0
        assert train_games.game_id.str[3:5].astype(int).max() < validation_season - 2000


def test_beta_calibrator_is_smooth_and_returns_probabilities():
    raw = np.linspace(0.05, 0.95, 40)
    labels = (raw > 0.52).astype(int)
    calibrator = BetaCalibrator().fit(raw, labels)
    calibrated = calibrator.predict(raw)
    assert np.all((calibrated > 0) & (calibrated < 1))
    assert np.all(np.diff(calibrated) >= 0)
