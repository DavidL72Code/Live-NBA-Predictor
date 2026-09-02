"""Random forest and gradient boosting on the corrected five-season data.

Earlier benchmarks rejected both, but on a different fold scheme and on the
feature table whose pre-game context was 26% placeholder values. This re-tests
them on the same protocol as the shipped model, and additionally hands each one
the diffusion basis — the parameterization that makes the linear model work —
so they are not disadvantaged by feature encoding.

The boosting models pick their own iteration count from an internal validation
split, which slices rows rather than games and so leaks slightly within a game.
That favours the trees; if they still lose, the conclusion is safe.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_winprob.features.basis import expand_live_basis
from nba_winprob.training.advanced import _metrics, walk_forward_game_folds
from nba_winprob.training.logistic import build_logistic_model
from nba_winprob.training.train import FEATURE_COLS, TARGET_COL

CALIBRATION_FRACTION = 0.15
BOOTSTRAP_RESAMPLES = 2000
EPSILON = 1e-6


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
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )

    df = pd.read_parquet("artifacts/benchmark_live_25_5season.parquet")

    def raw(frame):
        return frame[FEATURE_COLS].astype(float)

    # Leaf size is the regularizer that matters here: unrestricted leaves let a
    # forest memorize individual games the way the 400-tree booster did.
    builders = {
        "random_forest": (lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=50, max_features="sqrt",
            n_jobs=4, random_state=42), raw),
        "random_forest_on_basis": (lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=50, max_features="sqrt",
            n_jobs=4, random_state=42), expand_live_basis),
        "hist_gradient_boosting": (lambda: HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20, random_state=42), raw),
        "hist_gradient_boosting_on_basis": (lambda: HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20, random_state=42),
            expand_live_basis),
    }

    cells = ["logistic_basis", *builders]
    predictions: dict[str, list[np.ndarray]] = {c: [] for c in cells}
    labels, groups, iterations = [], [], {k: [] for k in builders}

    for train_idx, validation_idx, train_end, validation_end in walk_forward_game_folds(df):
        train, validation = df.iloc[train_idx], df.iloc[validation_idx]
        train_games = sorted(train["game_id"].astype(str).unique())
        split = int(len(train_games) * (1 - CALIBRATION_FRACTION))
        fit = train[train["game_id"].astype(str).isin(set(train_games[:split]))]
        y_fit = fit[TARGET_COL].astype(int)

        baseline = build_logistic_model(1.0)
        baseline.fit(fit, y_fit)
        predictions["logistic_basis"].append(baseline.predict_proba(validation)[:, 1])

        for name, (make, transform) in builders.items():
            model = make()
            model.fit(transform(fit), y_fit)
            predictions[name].append(model.predict_proba(transform(validation))[:, 1])
            grown = getattr(model, "n_iter_", None)
            if grown is None:
                grown = len(getattr(model, "estimators_", []))
            iterations[name].append(int(grown))

        labels.append(validation[TARGET_COL].astype(int).to_numpy())
        groups.append(validation["game_id"].astype(str).to_numpy())
        print(f"fold {train_end} → {validation_end}", flush=True)

    y = np.concatenate(labels)
    game_ids = np.concatenate(groups)
    results = {
        "cells": {c: _metrics(y, np.concatenate(p)) for c, p in predictions.items()},
        "vs_logistic_basis": {},
        "iterations_or_trees_per_fold": iterations,
        "oof_rows": int(len(y)),
        "total_games": int(df["game_id"].nunique()),
    }
    rng = np.random.default_rng(0)
    base = np.concatenate(predictions["logistic_basis"])
    for name in builders:
        results["vs_logistic_basis"][name] = _paired_delta(
            game_ids, y, base, np.concatenate(predictions[name]), rng
        )

    Path("artifacts/live_forest_gbm_comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results["cells"], indent=2))


if __name__ == "__main__":
    main()
