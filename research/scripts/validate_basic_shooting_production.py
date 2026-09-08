"""Production-shaped holdout validation for the basic shooting candidate."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nba_winprob.training.train import FEATURE_COLS, TARGET_COL

BASIC_SHOOTING_COLS = [
    "home_rolling_efg_pct", "away_rolling_efg_pct",
    "home_rolling_three_pct", "away_rolling_three_pct",
    "home_rolling_free_throw_pct", "away_rolling_free_throw_pct",
    "home_rolling_true_shooting_pct", "away_rolling_true_shooting_pct",
]


def _fit_predict(train, calibration, test, columns):
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    params = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 2,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(train[columns].astype(float), train[TARGET_COL].astype(int), verbose=False)
    calibration_raw = model.predict_proba(calibration[columns].astype(float))[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(calibration_raw, calibration[TARGET_COL].astype(int))
    raw = model.predict_proba(test[columns].astype(float))[:, 1]
    calibrated = calibrator.predict(raw)
    y = test[TARGET_COL].astype(int).to_numpy()
    return {
        "raw": {
            "brier": float(brier_score_loss(y, raw)),
            "log_loss": float(log_loss(y, raw, labels=[0, 1])),
            "roc_auc": float(roc_auc_score(y, raw)),
        },
        "isotonic": {
            "brier": float(brier_score_loss(y, calibrated)),
            "log_loss": float(log_loss(y, calibrated, labels=[0, 1])),
            "roc_auc": float(roc_auc_score(y, calibrated)),
        },
    }


def main() -> None:
    base = pd.read_parquet("artifacts/benchmark_live_25.parquet")
    shooting = pd.read_parquet(
        "data/features/features_schedule_player.parquet",
        columns=["game_id", *BASIC_SHOOTING_COLS],
    ).drop_duplicates("game_id")
    data = base.merge(shooting, on="game_id", how="left", validate="many_to_one")
    if data[BASIC_SHOOTING_COLS].isna().any().any():
        raise ValueError("basic shooting features contain missing values")

    # Final season is untouched. The prior season is calibration-only, and the
    # first three seasons are training-only, preserving chronological order.
    season_code = data["game_id"].astype(str).str[3:5].astype(int)
    train = data[season_code <= 23]
    calibration = data[season_code == 23]
    test = data[season_code == 24]
    # Remove calibration games from the model-training partition.
    train = data[season_code <= 22]
    if (
        train.game_id.nunique() == 0
        or calibration.game_id.nunique() == 0
        or test.game_id.nunique() == 0
    ):
        raise ValueError("chronological train/calibration/test split is incomplete")

    current = _fit_predict(train, calibration, test, FEATURE_COLS)
    shooting_result = _fit_predict(
        train, calibration, test, [*FEATURE_COLS, *BASIC_SHOOTING_COLS]
    )
    result = {
        "protocol": {
            "train_seasons": ["2021-22", "2022-23"],
            "calibration_season": "2023-24",
            "untouched_test_season": "2024-25",
            "train_games": int(train.game_id.nunique()),
            "calibration_games": int(calibration.game_id.nunique()),
            "test_games": int(test.game_id.nunique()),
            "production_xgb_settings": {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05},
        },
        "current_production_features": current,
        "basic_shooting_candidate": shooting_result,
        "isotonic_brier_delta_candidate_minus_current": (
            shooting_result["isotonic"]["brier"] - current["isotonic"]["brier"]
        ),
        "isotonic_logloss_delta_candidate_minus_current": (
            shooting_result["isotonic"]["log_loss"] - current["isotonic"]["log_loss"]
        ),
    }
    output = Path("artifacts/validate_basic_shooting_production.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
