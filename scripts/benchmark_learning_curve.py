"""How much is more training data actually worth?

Trains on the k games immediately preceding a fold boundary, for increasing k,
and scores the same validation block each time. A curve still descending at the
right edge means more history buys accuracy; a flat one means the data we have
already saturates the model and only new *information* will help.

Both model classes are measured, because trees are more data-hungry than linear
models: if the tree curve is steeper it may overtake the logistic given enough
seasons, which would change the recommendation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_winprob.training.advanced import _metrics
from nba_winprob.training.logistic import build_logistic_model
from nba_winprob.training.train import FEATURE_COLS, TARGET_COL

TRAIN_SIZES = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500)
VALIDATION_GAMES = 500
BOUNDARIES = (4500, 5000, 5500)
PARQUET = "artifacts/benchmark_live_25_5season.parquet"
CALIBRATION_FRACTION = 0.15
XGB_PARAMS = {
    "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "objective": "binary:logistic",
    "eval_metric": "logloss", "tree_method": "hist", "random_state": 42,
    "n_jobs": 4, "early_stopping_rounds": 30,
}


def main() -> None:
    import xgboost as xgb

    df = pd.read_parquet(PARQUET)
    games = np.array(sorted(df["game_id"].astype(str).unique()))
    by_game = {game: frame for game, frame in df.groupby(df["game_id"].astype(str))}

    def block(selected) -> pd.DataFrame:
        return pd.concat([by_game[game] for game in selected])

    curves: dict[str, dict[int, list[float]]] = {
        "logistic_basis": {size: [] for size in TRAIN_SIZES},
        "xgb_early_stopped": {size: [] for size in TRAIN_SIZES},
    }

    for boundary in BOUNDARIES:
        validation = block(games[boundary:boundary + VALIDATION_GAMES])
        y_validation = validation[TARGET_COL].astype(int).to_numpy()
        for size in TRAIN_SIZES:
            train_games = games[boundary - size:boundary]
            train = block(train_games)
            y_train = train[TARGET_COL].astype(int)

            model = build_logistic_model(1.0)
            model.fit(train, y_train)
            curves["logistic_basis"][size].append(
                _metrics(y_validation, model.predict_proba(validation)[:, 1])["brier"]
            )

            # Trees need a holdout for early stopping; carve it from the same block
            # so the tree model never sees more games than the logistic one.
            split = int(len(train_games) * (1 - CALIBRATION_FRACTION))
            fit = block(train_games[:split])
            calibration = block(train_games[split:])
            booster = xgb.XGBClassifier(**XGB_PARAMS)
            booster.fit(
                fit[FEATURE_COLS].astype(float),
                fit[TARGET_COL].astype(int),
                eval_set=[(
                    calibration[FEATURE_COLS].astype(float),
                    calibration[TARGET_COL].astype(int),
                )],
                verbose=False,
            )
            curves["xgb_early_stopped"][size].append(
                _metrics(
                    y_validation,
                    booster.predict_proba(validation[FEATURE_COLS].astype(float))[:, 1],
                )["brier"]
            )
            print(f"boundary {boundary} size {size} done", flush=True)

    results = {
        model: {
            str(size): {
                "brier_mean": float(np.mean(scores)),
                "brier_per_boundary": [float(s) for s in scores],
            }
            for size, scores in sizes.items()
        }
        for model, sizes in curves.items()
    }
    # Marginal value of the last doubling, the number that decides whether more
    # seasons are worth collecting.
    for model, sizes in curves.items():
        last = float(np.mean(sizes[4500]))
        half = float(np.mean(sizes[2000]))
        results[model]["brier_gain_from_doubling_2000_to_4500"] = round(half - last, 6)

    Path("artifacts/live_learning_curve_5season.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
