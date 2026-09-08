"""Integrated, production-shaped research comparison for model improvements."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_winprob.training.train import FEATURE_COLS, TARGET_COL

BASIC_SHOOTING_COLS = [
    "home_rolling_efg_pct", "away_rolling_efg_pct",
    "home_rolling_three_pct", "away_rolling_three_pct",
    "home_rolling_free_throw_pct", "away_rolling_free_throw_pct",
    "home_rolling_true_shooting_pct", "away_rolling_true_shooting_pct",
]


def metrics(y, p) -> dict[str, float]:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)),
    }


def calibrate(raw_cal, y_cal, raw_test):
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(raw_cal, y_cal)
    return model.predict(raw_test)


def main() -> None:
    import xgboost as xgb
    from scipy.stats import norm
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    base = pd.read_parquet("artifacts/benchmark_live_25.parquet")
    margins = pd.read_parquet(
        "artifacts/benchmark_live_25_with_margin.parquet",
        columns=["game_id", "final_margin"],
    ).drop_duplicates("game_id")
    base = base.merge(
        margins,
        on=["game_id"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    shooting = pd.read_parquet(
        "data/features/features_schedule_player.parquet",
        columns=["game_id", *BASIC_SHOOTING_COLS],
    ).drop_duplicates("game_id")
    data = base.merge(shooting, on="game_id", how="left", validate="many_to_one")
    if data[BASIC_SHOOTING_COLS].isna().any().any():
        raise ValueError("missing lagged shooting features")

    season = data["game_id"].astype(str).str[3:5].astype(int)
    train = data[season <= 22]
    calibration = data[season == 23]
    test = data[season == 24]
    y_train = train[TARGET_COL].astype(int)
    y_cal = calibration[TARGET_COL].astype(int).to_numpy()
    y_test = test[TARGET_COL].astype(int).to_numpy()

    params = {
        "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "objective": "binary:logistic", "eval_metric": "logloss",
        "tree_method": "hist", "random_state": 42,
    }

    def xgb_candidate(columns):
        model = xgb.XGBClassifier(**params)
        model.fit(
            train[columns].astype(float), y_train,
            eval_set=[(
                calibration[columns].astype(float), y_cal
            )],
            verbose=False,
        )
        cal_raw = model.predict_proba(calibration[columns].astype(float))[:, 1]
        test_raw = model.predict_proba(test[columns].astype(float))[:, 1]
        test_cal = calibrate(cal_raw, y_cal, test_raw)
        return {"raw": metrics(y_test, test_raw), "isotonic": metrics(y_test, test_cal),
                "calibration_raw": cal_raw, "test_raw": test_raw, "test_cal": test_cal}

    current = xgb_candidate(FEATURE_COLS)
    shooting_result = xgb_candidate([*FEATURE_COLS, *BASIC_SHOOTING_COLS])

    logistic = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)
    )
    logistic.fit(train[FEATURE_COLS].astype(float), y_train)
    logistic_cal_raw = logistic.predict_proba(calibration[FEATURE_COLS].astype(float))[:, 1]
    logistic_test_raw = logistic.predict_proba(test[FEATURE_COLS].astype(float))[:, 1]
    logistic_test_cal = calibrate(logistic_cal_raw, y_cal, logistic_test_raw)
    logistic_result = {"raw": metrics(y_test, logistic_test_raw),
                       "isotonic": metrics(y_test, logistic_test_cal)}

    margin_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    margin_model.fit(train[FEATURE_COLS].astype(float), train["final_margin"].astype(float))
    margin_train_pred = margin_model.predict(train[FEATURE_COLS].astype(float))
    residual = train["final_margin"].to_numpy() - margin_train_pred
    scales = []
    for lower, upper in ((60.0, 300.0), (300.0, 720.0), (720.0, np.inf)):
        time = train["seconds_remaining"].to_numpy()
        values = residual[(time > lower) & (time <= upper)]
        scales.append(max(float(np.std(values, ddof=1)), 1.0))
    pred_margin = test["score_diff"].to_numpy() + margin_model.predict(
        test[FEATURE_COLS].astype(float)
    )
    remaining = test["seconds_remaining"].to_numpy()
    sigma = np.where(
        remaining <= 300.0, scales[0],
        np.where(remaining <= 720.0, scales[1], scales[2]),
    )
    margin_test_raw = norm.cdf(pred_margin / sigma)
    margin_cal_raw = norm.cdf(
        calibration["score_diff"].to_numpy()
        + margin_model.predict(calibration[FEATURE_COLS].astype(float))
    / np.where(
        calibration["seconds_remaining"].to_numpy() <= 300.0,
        scales[0],
        np.where(
            calibration["seconds_remaining"].to_numpy() <= 720.0,
            scales[1], scales[2],
        ),
    )
    )
    margin_test_cal = calibrate(margin_cal_raw, y_cal, margin_test_raw)
    margin_result = {"raw": metrics(y_test, margin_test_raw),
                     "isotonic": metrics(y_test, margin_test_cal), "sigma": scales}

    # Calibration-selected blend: weights are learned only on the calibration season.
    candidates = []
    xgb_cal = current["calibration_raw"]
    for weight in np.linspace(0.0, 1.0, 21):
        blend_cal = weight * xgb_cal + (1 - weight) * logistic_cal_raw
        score = float(np.mean(
            -(
                y_cal * np.log(np.clip(blend_cal, 1e-6, 1 - 1e-6))
                + (1 - y_cal) * np.log(np.clip(1 - blend_cal, 1e-6, 1 - 1e-6))
            )
        ))
        candidates.append((score, float(weight)))
    _, selected_weight = min(candidates)
    blend_test_raw = (
        selected_weight * current["test_raw"]
        + (1 - selected_weight) * logistic_test_raw
    )
    blend_cal_raw = selected_weight * xgb_cal + (1 - selected_weight) * logistic_cal_raw
    blend_test_cal = calibrate(blend_cal_raw, y_cal, blend_test_raw)
    blend_result = {"selected_xgb_weight": selected_weight,
                    "raw": metrics(y_test, blend_test_raw),
                    "isotonic": metrics(y_test, blend_test_cal)}

    def strip_arrays(value):
        return {
            key: strip_arrays(item) if isinstance(item, dict) else item
            for key, item in value.items()
            if not isinstance(item, np.ndarray)
        }

    result = {
        "protocol": {
            "train_seasons": ["2021-22", "2022-23"],
            "calibration_season": "2023-24",
            "untouched_test_season": "2024-25",
            "train_games": int(train.game_id.nunique()),
            "calibration_games": int(calibration.game_id.nunique()),
            "test_games": int(test.game_id.nunique()),
            "production_xgb": params,
        },
        "current_production": strip_arrays(current),
        "basic_shooting": strip_arrays(shooting_result),
        "logistic": logistic_result,
        "margin_distribution": margin_result,
        "calibration_selected_oof_style_blend": blend_result,
    }
    for _name, candidate in result.items():
        if isinstance(candidate, dict) and "isotonic" in candidate:
            candidate["isotonic_brier_delta_vs_current"] = (
                candidate["isotonic"]["brier"] - current["isotonic"]["brier"]
            )
            candidate["isotonic_logloss_delta_vs_current"] = (
                candidate["isotonic"]["log_loss"] - current["isotonic"]["log_loss"]
            )
    output = Path("artifacts/five_phase_production_research.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
