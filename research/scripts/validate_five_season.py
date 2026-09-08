"""Re-baseline the model comparison on five seasons of corrected feature data.

The previous feature table left pre-game context at its placeholder values for
26% of games, which is why dropping that block appeared to help. This rebuilds
the comparison on data where the context join is intact, and asks two questions
at once: does the shipped model still win, and do the pre-game features earn
their place now that they are actually populated?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nba_winprob.features.basis import expand_live_basis
from nba_winprob.training.advanced import _metrics, walk_forward_game_folds
from nba_winprob.training.logistic import build_logistic_model
from nba_winprob.training.train import FEATURE_COLS, TARGET_COL

PREGAME_COLS = [
    "home_win_pct", "home_avg_margin", "home_streak",
    "away_win_pct", "away_avg_margin", "away_streak",
]
PRODUCTION_XGB = {
    "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "objective": "binary:logistic",
    "eval_metric": "logloss", "tree_method": "hist", "random_state": 42, "n_jobs": 4,
}
CALIBRATION_FRACTION = 0.15
BOOTSTRAP_RESAMPLES = 2000
EPSILON = 1e-6


def _with_pregame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [expand_live_basis(frame), frame[PREGAME_COLS].astype(float)], axis=1
    )


def _paired_delta(game_ids, y, base, candidate, rng):
    base = np.clip(np.asarray(base, dtype=np.float64), EPSILON, 1 - EPSILON)
    candidate = np.clip(np.asarray(candidate, dtype=np.float64), EPSILON, 1 - EPSILON)
    rows = pd.DataFrame({
        "game": game_ids,
        "brier": (candidate - y) ** 2 - (base - y) ** 2,
        "logloss": (
            -(y * np.log(candidate) + (1 - y) * np.log(1 - candidate))
            + (y * np.log(base) + (1 - y) * np.log(1 - base))
        ),
    })
    grouped = rows.groupby("game").agg(["sum", "count"])
    counts = grouped[("brier", "count")].to_numpy()
    picks = rng.integers(0, len(counts), size=(BOOTSTRAP_RESAMPLES, len(counts)))
    totals = counts[picks].sum(axis=1)
    out = {}
    for name, column in (("brier", "brier"), ("log_loss", "logloss")):
        sums = grouped[(column, "sum")].to_numpy()
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

    df = pd.read_parquet("artifacts/benchmark_live_25_5season.parquet")
    cells = [
        "logistic_basis", "logistic_basis_plus_pregame",
        "xgb_production_isotonic", "xgb_early_stopped",
    ]
    predictions: dict[str, list[np.ndarray]] = {cell: [] for cell in cells}
    labels, groups, fold_log = [], [], []

    for train_idx, validation_idx, train_end, validation_end in walk_forward_game_folds(df):
        train, validation = df.iloc[train_idx], df.iloc[validation_idx]
        train_games = sorted(train["game_id"].astype(str).unique())
        split = int(len(train_games) * (1 - CALIBRATION_FRACTION))
        fit = train[train["game_id"].astype(str).isin(set(train_games[:split]))]
        calibration = train[train["game_id"].astype(str).isin(set(train_games[split:]))]
        y_fit = fit[TARGET_COL].astype(int)
        y_calibration = calibration[TARGET_COL].astype(int).to_numpy()

        model = build_logistic_model(1.0)
        model.fit(fit, y_fit)
        predictions["logistic_basis"].append(model.predict_proba(validation)[:, 1])

        augmented = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        augmented.fit(_with_pregame(fit), y_fit)
        predictions["logistic_basis_plus_pregame"].append(
            augmented.predict_proba(_with_pregame(validation))[:, 1]
        )

        production = xgb.XGBClassifier(**PRODUCTION_XGB)
        production.fit(fit[FEATURE_COLS].astype(float), y_fit)
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            production.predict_proba(calibration[FEATURE_COLS].astype(float))[:, 1],
            y_calibration,
        )
        predictions["xgb_production_isotonic"].append(
            calibrator.predict(
                production.predict_proba(validation[FEATURE_COLS].astype(float))[:, 1]
            )
        )

        early_stopped = xgb.XGBClassifier(**PRODUCTION_XGB, early_stopping_rounds=30)
        early_stopped.fit(
            fit[FEATURE_COLS].astype(float), y_fit,
            eval_set=[(calibration[FEATURE_COLS].astype(float), y_calibration)],
            verbose=False,
        )
        predictions["xgb_early_stopped"].append(
            early_stopped.predict_proba(validation[FEATURE_COLS].astype(float))[:, 1]
        )

        labels.append(validation[TARGET_COL].astype(int).to_numpy())
        groups.append(validation["game_id"].astype(str).to_numpy())
        fold_log.append({
            "train_end": train_end,
            "validation_end": validation_end,
            "train_games": len(train_games),
            "xgb_best_iteration": int(early_stopped.best_iteration),
        })
        print(f"fold {train_end} → {validation_end} ({len(train_games)} train games)", flush=True)

    y = np.concatenate(labels)
    game_ids = np.concatenate(groups)
    results = {
        "cells": {c: _metrics(y, np.concatenate(p)) for c, p in predictions.items()},
        "vs_shipped_logistic_basis": {},
        "folds": fold_log,
        "oof_rows": int(len(y)),
        "total_games": int(df["game_id"].nunique()),
    }
    rng = np.random.default_rng(0)
    base = np.concatenate(predictions["logistic_basis"])
    for cell in cells:
        if cell == "logistic_basis":
            continue
        results["vs_shipped_logistic_basis"][cell] = _paired_delta(
            game_ids, y, base, np.concatenate(predictions[cell]), rng
        )

    Path("artifacts/live_five_season_comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results["cells"], indent=2))
    print(json.dumps(results["vs_shipped_logistic_basis"], indent=2))


if __name__ == "__main__":
    main()
