"""Rolling-fold comparison of the production XGBoost stack vs the logistic basis model.

Both candidates train on the *same* 85% of each fold's games. XGBoost gets the
remaining 15% for its isotonic calibrator and early stopping; the logistic model
never sees those games, so the split is if anything harsh on the candidate. Its
regularization strength is chosen by walk-forward folds nested inside the
training block, so no hyperparameter is picked using validation data.

The logistic model is the exact pipeline that ships (``build_logistic_model``),
so the numbers describe the served artifact rather than a research stand-in.
The deployed artifact is refit on every available game — these folds estimate
the quality of the *procedure*, which is the number to decide on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_winprob.training.advanced import _metrics, walk_forward_game_folds
from nba_winprob.training.logistic import build_logistic_model
from nba_winprob.training.train import FEATURE_COLS, TARGET_COL

PRODUCTION_XGB = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 4,
}
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)
CALIBRATION_FRACTION = 0.15
BOOTSTRAP_RESAMPLES = 400


def _select_c(fit_frame: pd.DataFrame) -> float:
    """Choose regularization strength using folds nested inside the train block."""
    scores = []
    for c_value in C_GRID:
        inner_losses = []
        for inner_train, inner_validation, _, _ in walk_forward_game_folds(
            fit_frame, min_train_games=500, validation_games=500, step_games=500
        ):
            train = fit_frame.iloc[inner_train]
            validation = fit_frame.iloc[inner_validation]
            model = build_logistic_model(c_value)
            model.fit(train, train[TARGET_COL].astype(int))
            inner_losses.append(
                _metrics(validation[TARGET_COL], model.predict_proba(validation)[:, 1])[
                    "log_loss"
                ]
            )
        scores.append((float(np.mean(inner_losses)), c_value))
    return min(scores)[1]


def main() -> None:
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression

    df = pd.read_parquet("artifacts/benchmark_live_25.parquet")
    predictions: dict[str, list[np.ndarray]] = {
        "production_xgb_isotonic": [],
        "xgb_early_stopped": [],
        "logistic_basis_nested_c": [],
    }
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    selected_c: list[float] = []

    for train_idx, validation_idx, train_end, validation_end in walk_forward_game_folds(df):
        train, validation = df.iloc[train_idx], df.iloc[validation_idx]
        # The most recent games in the training block become the calibration
        # and early-stopping set, so nothing after the fold boundary is used.
        train_games = sorted(train["game_id"].astype(str).unique())
        split = int(len(train_games) * (1 - CALIBRATION_FRACTION))
        fit = train[train["game_id"].astype(str).isin(set(train_games[:split]))]
        calibration = train[train["game_id"].astype(str).isin(set(train_games[split:]))]
        y_fit = fit[TARGET_COL].astype(int)
        y_calibration = calibration[TARGET_COL].astype(int).to_numpy()

        production = xgb.XGBClassifier(**PRODUCTION_XGB)
        production.fit(fit[FEATURE_COLS].astype(float), y_fit)
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            production.predict_proba(calibration[FEATURE_COLS].astype(float))[:, 1],
            y_calibration,
        )
        raw = production.predict_proba(validation[FEATURE_COLS].astype(float))[:, 1]
        predictions["production_xgb_isotonic"].append(calibrator.predict(raw))

        early_stopped = xgb.XGBClassifier(**PRODUCTION_XGB, early_stopping_rounds=30)
        early_stopped.fit(
            fit[FEATURE_COLS].astype(float),
            y_fit,
            eval_set=[(calibration[FEATURE_COLS].astype(float), y_calibration)],
            verbose=False,
        )
        predictions["xgb_early_stopped"].append(
            early_stopped.predict_proba(validation[FEATURE_COLS].astype(float))[:, 1]
        )

        c_value = _select_c(fit)
        selected_c.append(c_value)
        logistic = build_logistic_model(c_value)
        logistic.fit(fit, y_fit)
        predictions["logistic_basis_nested_c"].append(logistic.predict_proba(validation)[:, 1])

        labels.append(validation[TARGET_COL].astype(int).to_numpy())
        groups.append(validation["game_id"].astype(str).to_numpy())
        print(f"fold {train_end} → {validation_end}  C={c_value}", flush=True)

    y = np.concatenate(labels)
    game_ids = np.concatenate(groups)
    results = {name: _metrics(y, np.concatenate(p)) for name, p in predictions.items()}

    # Paired game-level bootstrap keeps both models on identical resamples.
    baseline = np.concatenate(predictions["production_xgb_isotonic"])
    rng = np.random.default_rng(0)
    games = np.unique(game_ids)
    positions = {game: np.flatnonzero(game_ids == game) for game in games}
    for name in ("xgb_early_stopped", "logistic_basis_nested_c"):
        candidate = np.concatenate(predictions[name])
        deltas = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            sample = np.concatenate(
                [positions[game] for game in rng.choice(games, len(games), replace=True)]
            )
            deltas.append(
                _metrics(y[sample], candidate[sample])["brier"]
                - _metrics(y[sample], baseline[sample])["brier"]
            )
        results[name]["paired_brier_delta_vs_production_95"] = [
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)),
        ]

    results["selected_c_per_fold"] = selected_c
    results["oof_rows"] = int(len(y))
    Path("artifacts/live_logistic_basis_comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
