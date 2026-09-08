"""Multi-season expanding production-parity validation for XGB/logistic/blend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_winprob.training.train import FEATURE_COLS, TARGET_COL, apply_temperature, fit_temperature


def score(y, p) -> dict[str, float]:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)),
    }


def main() -> None:
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = pd.read_parquet("artifacts/benchmark_live_25.parquet")
    starts = data["game_id"].astype(str).str[3:5].astype(int)
    params = {
        "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "objective": "binary:logistic", "eval_metric": "logloss",
        "tree_method": "hist", "random_state": 42,
    }
    folds = []
    paired_rows = []
    for validation_season in (22, 23, 24):
        historical = data[starts < validation_season]
        validation = data[starts == validation_season]
        games = historical["game_id"].drop_duplicates().to_numpy()
        calibration_games = set(games[int(len(games) * 0.9):])
        train = historical[~historical["game_id"].isin(calibration_games)]
        calibration = historical[historical["game_id"].isin(calibration_games)]
        y_train = train[TARGET_COL].astype(int)
        y_cal = calibration[TARGET_COL].astype(int).to_numpy()
        y_test = validation[TARGET_COL].astype(int).to_numpy()

        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(
            train[FEATURE_COLS].astype(float), y_train,
            eval_set=[(calibration[FEATURE_COLS].astype(float), y_cal)],
            verbose=False,
        )
        xgb_cal_raw = xgb_model.predict_proba(calibration[FEATURE_COLS].astype(float))[:, 1]
        xgb_test_raw = xgb_model.predict_proba(validation[FEATURE_COLS].astype(float))[:, 1]

        log_model = make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)
        )
        log_model.fit(train[FEATURE_COLS].astype(float), y_train)
        log_cal_raw = log_model.predict_proba(calibration[FEATURE_COLS].astype(float))[:, 1]
        log_test_raw = log_model.predict_proba(validation[FEATURE_COLS].astype(float))[:, 1]

        xgb_iso = IsotonicRegression(out_of_bounds="clip").fit(xgb_cal_raw, y_cal)
        log_iso = IsotonicRegression(out_of_bounds="clip").fit(log_cal_raw, y_cal)
        temperature, _ = fit_temperature(log_cal_raw, y_cal)
        weights = []
        for weight in np.linspace(0.0, 1.0, 21):
            blend = weight * xgb_cal_raw + (1 - weight) * log_cal_raw
            clipped = np.clip(blend, 1e-6, 1 - 1e-6)
            loss = float(np.mean(-(y_cal * np.log(clipped) + (1 - y_cal) * np.log(1 - clipped))))
            weights.append((loss, float(weight)))
        _, selected_weight = min(weights)
        xgb_test_iso = xgb_iso.predict(xgb_test_raw)
        log_test_iso = log_iso.predict(log_test_raw)
        log_test_temperature = apply_temperature(log_test_raw, temperature)
        blend_test = selected_weight * xgb_test_raw + (1 - selected_weight) * log_test_raw
        blend_iso = IsotonicRegression(out_of_bounds="clip").fit(
            selected_weight * xgb_cal_raw + (1 - selected_weight) * log_cal_raw,
            y_cal,
        ).predict(blend_test)
        folds.append({
            "validation_season": f"20{validation_season}-",
            "train_games": int(train.game_id.nunique()),
            "calibration_games": int(calibration.game_id.nunique()),
            "validation_games": int(validation.game_id.nunique()),
            "selected_xgb_weight": selected_weight,
            "xgb": {"raw": score(y_test, xgb_test_raw), "isotonic": score(y_test, xgb_test_iso)},
            "logistic": {
                "raw": score(y_test, log_test_raw),
                "isotonic": score(y_test, log_test_iso),
                "temperature": score(y_test, log_test_temperature),
                "temperature_value": temperature,
            },
            "blend": {"raw": score(y_test, blend_test), "isotonic": score(y_test, blend_iso)},
        })
        game_frame = validation[["game_id"]].copy()
        game_frame["y"] = y_test
        game_frame["xgb_raw"] = xgb_test_raw
        game_frame["logistic_raw"] = log_test_raw
        paired_rows.append(game_frame.groupby("game_id", sort=False).agg({
            "y": "first", "xgb_raw": "mean", "logistic_raw": "mean",
        }).reset_index())

    paired = pd.concat(paired_rows, ignore_index=True)
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(2000):
        indices = rng.integers(0, len(paired), len(paired))
        sample = paired.iloc[indices]
        deltas.append(
            float(
                np.mean((sample.logistic_raw - sample.y) ** 2)
                - np.mean((sample.xgb_raw - sample.y) ** 2)
            )
        )
    paired_interval = {
        "mean_logistic_minus_xgb_brier": float(np.mean(deltas)),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "games": int(len(paired)),
    }

    output = Path("artifacts/expanding_production_folds.json")
    result = {"protocol": params, "folds": folds, "paired_game_bootstrap": paired_interval}
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "folds": folds,
        "paired_game_bootstrap": paired_interval,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
