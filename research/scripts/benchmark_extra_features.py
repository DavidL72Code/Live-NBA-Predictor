"""Do the box-score features help either model class? Side-by-side factorial test.

Every cell trains on the same 85% of each fold's games and is scored on the same
validation rows, so the only thing varying within a model class is the feature
set. Three model classes (production XGBoost + isotonic, early-stopped XGBoost,
logistic + diffusion basis) crossed with three feature sets (base, base +
shooting, base + all box-score columns).

The paired bootstrap resamples whole games and compares each augmented cell
against its own base, so "did the features help?" is answered separately for
each model class rather than confounded with the choice of model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_winprob.features.basis import expand_live_basis
from nba_winprob.training.advanced import _metrics, walk_forward_game_folds
from nba_winprob.training.train import FEATURE_COLS, TARGET_COL

SHOOTING_COLS = [
    "home_fga", "away_fga", "home_fgm", "away_fgm",
    "home_3pa", "away_3pa", "home_3pm", "away_3pm",
    "home_fta", "away_fta", "home_ftm", "away_ftm",
]
OTHER_BOX_COLS = [
    "home_fouls", "away_fouls", "home_turnovers", "away_turnovers",
    "home_rebounds", "away_rebounds", "home_off_rebounds", "away_off_rebounds",
    "home_assists", "away_assists", "home_steals", "away_steals",
    "home_blocks", "away_blocks",
]
FEATURE_SETS = {
    "base": [],
    "plus_shooting": SHOOTING_COLS,
    "plus_all_boxscore": [*SHOOTING_COLS, *OTHER_BOX_COLS],
}
PRODUCTION_XGB = {
    "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "objective": "binary:logistic",
    "eval_metric": "logloss", "tree_method": "hist", "random_state": 42, "n_jobs": 4,
}
CALIBRATION_FRACTION = 0.15
BOOTSTRAP_RESAMPLES = 2000
# Large enough to survive the float32 round-trip out of XGBoost, where
# 1 - 1e-9 rounds to exactly 1.0 and the upper clip becomes a no-op.
EPSILON = 1e-6


def _paired_delta(game_ids, y, base, candidate, rng):
    """Bootstrap whole games for the candidate-minus-base metric deltas.

    Both metrics are row-wise means, so a per-row difference can be aggregated
    by game once and resampled cheaply, which is exact rather than approximate.
    """
    base = np.clip(np.asarray(base, dtype=np.float64), EPSILON, 1 - EPSILON)
    candidate = np.clip(np.asarray(candidate, dtype=np.float64), EPSILON, 1 - EPSILON)
    brier_row = (candidate - y) ** 2 - (base - y) ** 2
    logloss_row = -(y * np.log(candidate) + (1 - y) * np.log(1 - candidate)) + (
        y * np.log(base) + (1 - y) * np.log(1 - base)
    )

    frame = pd.DataFrame({"game": game_ids, "brier": brier_row, "logloss": logloss_row})
    grouped = frame.groupby("game").agg(["sum", "count"])
    brier_sum = grouped[("brier", "sum")].to_numpy()
    logloss_sum = grouped[("logloss", "sum")].to_numpy()
    counts = grouped[("brier", "count")].to_numpy()

    n_games = len(counts)
    picks = rng.integers(0, n_games, size=(BOOTSTRAP_RESAMPLES, n_games))
    totals = counts[picks].sum(axis=1)
    out = {}
    for name, sums in (("brier", brier_sum), ("log_loss", logloss_sum)):
        draws = sums[picks].sum(axis=1) / totals
        out[name] = {
            "mean": float(sums.sum() / counts.sum()),
            "lower": float(np.percentile(draws, 2.5)),
            "upper": float(np.percentile(draws, 97.5)),
        }
    return out


def main() -> None:
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = pd.read_parquet("artifacts/benchmark_live_25_play_features.parquet")
    cells = [
        f"{model}__{features}"
        for model in ("xgb_production_isotonic", "xgb_early_stopped", "logistic_basis")
        for features in FEATURE_SETS
    ]
    predictions: dict[str, list[np.ndarray]] = {cell: [] for cell in cells}
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []

    for train_idx, validation_idx, train_end, validation_end in walk_forward_game_folds(df):
        train, validation = df.iloc[train_idx], df.iloc[validation_idx]
        train_games = sorted(train["game_id"].astype(str).unique())
        split = int(len(train_games) * (1 - CALIBRATION_FRACTION))
        fit = train[train["game_id"].astype(str).isin(set(train_games[:split]))]
        calibration = train[train["game_id"].astype(str).isin(set(train_games[split:]))]
        y_fit = fit[TARGET_COL].astype(int)
        y_calibration = calibration[TARGET_COL].astype(int).to_numpy()

        for features, extra in FEATURE_SETS.items():
            columns = [*FEATURE_COLS, *extra]
            X_fit = fit[columns].astype(float)
            X_calibration = calibration[columns].astype(float)
            X_validation = validation[columns].astype(float)

            production = xgb.XGBClassifier(**PRODUCTION_XGB).fit(X_fit, y_fit)
            calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                production.predict_proba(X_calibration)[:, 1], y_calibration
            )
            predictions[f"xgb_production_isotonic__{features}"].append(
                calibrator.predict(production.predict_proba(X_validation)[:, 1])
            )

            early_stopped = xgb.XGBClassifier(**PRODUCTION_XGB, early_stopping_rounds=30)
            early_stopped.fit(
                X_fit, y_fit, eval_set=[(X_calibration, y_calibration)], verbose=False
            )
            predictions[f"xgb_early_stopped__{features}"].append(
                early_stopped.predict_proba(X_validation)[:, 1]
            )

            # The basis already encodes the live state; the extra columns are
            # appended to it so both model classes see the same information.
            def build(frame, extra=extra):
                expanded = expand_live_basis(frame)
                if not extra:
                    return expanded
                return pd.concat(
                    [expanded, frame[extra].astype(float).add_prefix("box_")], axis=1
                )

            logistic = make_pipeline(
                StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)
            )
            logistic.fit(build(fit), y_fit)
            predictions[f"logistic_basis__{features}"].append(
                logistic.predict_proba(build(validation))[:, 1]
            )

        labels.append(validation[TARGET_COL].astype(int).to_numpy())
        groups.append(validation["game_id"].astype(str).to_numpy())
        print(f"fold {train_end} → {validation_end}", flush=True)

    y = np.concatenate(labels)
    game_ids = np.concatenate(groups)
    results = {
        "cells": {cell: _metrics(y, np.concatenate(p)) for cell, p in predictions.items()},
        "feature_effect_within_model": {},
        "oof_rows": int(len(y)),
    }

    rng = np.random.default_rng(0)
    for model in ("xgb_production_isotonic", "xgb_early_stopped", "logistic_basis"):
        base = np.concatenate(predictions[f"{model}__base"])
        for features in FEATURE_SETS:
            if features == "base":
                continue
            candidate = np.concatenate(predictions[f"{model}__{features}"])
            results["feature_effect_within_model"][f"{model}__{features}_minus_base"] = (
                _paired_delta(game_ids, y, base, candidate, rng)
            )

    Path("artifacts/live_extra_feature_side_by_side.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
