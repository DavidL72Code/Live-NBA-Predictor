"""Train current/candidate models and compare them through serving interfaces."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from nba_winprob.analyst.serve import LogisticWinProbServer, WinProbServer
from nba_winprob.schemas import FeatureVector
from nba_winprob.training.train import FEATURE_COLS, TARGET_COL


def _features(frame: pd.DataFrame) -> list[FeatureVector]:
    fields = set(FeatureVector.model_fields)
    return [
        FeatureVector(**{key: row[key] for key in fields if key in row})
        for row in frame.to_dict("records")
    ]


def main() -> None:
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = pd.read_parquet("artifacts/benchmark_live_25.parquet")
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
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(
        train[FEATURE_COLS].astype(float), y_train,
        eval_set=[(calibration[FEATURE_COLS].astype(float), y_cal)],
        verbose=False,
    )
    xgb_cal_raw = xgb_model.predict_proba(calibration[FEATURE_COLS].astype(float))[:, 1]
    xgb_calibrator = IsotonicRegression(out_of_bounds="clip").fit(xgb_cal_raw, y_cal)

    logistic_model = make_pipeline(
        StandardScaler(), LogisticRegression(C=10.0, max_iter=1000)
    )
    logistic_model.fit(train[FEATURE_COLS].astype(float), y_train)
    model_path = Path("artifacts/shadow_logistic_model.pkl")
    with model_path.open("wb") as artifact:
        pickle.dump(logistic_model, artifact)

    current_server = WinProbServer(xgb_model, xgb_calibrator)
    candidate_server = LogisticWinProbServer.from_paths(model_path)
    features = _features(test)
    current_probabilities = current_server.predict_batch(features)
    candidate_probabilities = candidate_server.predict_batch(features)

    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    def metrics(probabilities):
        return {
            "brier": float(brier_score_loss(y_test, probabilities)),
            "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }

    result = {
        "protocol": {
            "train_games": int(train.game_id.nunique()),
            "calibration_games": int(calibration.game_id.nunique()),
            "shadow_games": int(test.game_id.nunique()),
            "feature_cols": FEATURE_COLS,
            "candidate_regularization_C": 10.0,
        },
        "current_xgb_server": metrics(current_probabilities),
        "candidate_logistic_server": metrics(candidate_probabilities),
        "max_probability_interface_error": float(max(
            abs(a - b) for a, b in zip(
                candidate_probabilities,
                logistic_model.predict_proba(test[FEATURE_COLS].astype(float))[:, 1],
                strict=True,
            )
        )),
        "brier_delta_candidate_minus_current": float(
            brier_score_loss(y_test, candidate_probabilities)
            - brier_score_loss(y_test, current_probabilities)
        ),
    }
    output = Path("artifacts/logistic_shadow_test.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
