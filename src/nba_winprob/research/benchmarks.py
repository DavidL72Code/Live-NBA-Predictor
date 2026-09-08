"""Model comparisons that informed the shipped choice but nothing calls now.

Each of these produced a committed artifact under ``artifacts/``. They are kept
so those numbers can be regenerated rather than taken on trust — which matters,
since a feature-table defect found later invalidated some earlier results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_winprob.training.advanced import (
    FEATURE_GROUPS,
    BetaCalibrator,
    _blend_weights,
    _game_bootstrap_intervals,
    _metrics,
    _paired_game_bootstrap_delta,
    _xgb_params,
    fit_early_stopped_xgb,
    oof_ensemble_benchmark,
    walk_forward_folds,
    walk_forward_game_folds,
)
from nba_winprob.training.train import FEATURE_COLS, TARGET_COL, _validate

__all__ = [
    "beta_calibration_benchmark",
    "feature_group_ablation_benchmark",
    "nested_game_dynamic_distribution_benchmark",
    "nested_game_horizon_logistic_benchmark",
    "nested_game_logistic_benchmark",
    "nested_game_margin_benchmark",
    "nested_game_xgb_vs_logistic",
    "nested_logistic_benchmark",
    "nested_spline_logistic_benchmark",
]


def nested_game_logistic_benchmark(
    df: pd.DataFrame,
    feature_cols: list[str],
    c_grid: tuple[float, ...] = (0.5, 1.0, 3.0, 10.0, 30.0, 100.0),
) -> dict:
    """Evaluate regularized logistic regression on rolling game-time folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = []
    predictions = []
    game_ids = []
    seasons = []
    for train_idx, validation_idx, train_end, validation_end in walk_forward_game_folds(df):
        outer_train = df.iloc[train_idx]
        outer_validation = df.iloc[validation_idx]
        scores = []
        for c_value in c_grid:
            inner_scores = []
            for inner_train_idx, inner_validation_idx, _, _ in walk_forward_game_folds(
                outer_train,
                min_train_games=500,
                validation_games=500,
                step_games=500,
            ):
                inner_train = outer_train.iloc[inner_train_idx]
                inner_validation = outer_train.iloc[inner_validation_idx]
                model = make_pipeline(
                    StandardScaler(), LogisticRegression(max_iter=1000, C=c_value)
                )
                model.fit(
                    inner_train[feature_cols].astype(float),
                    inner_train[TARGET_COL].astype(int),
                )
                inner_scores.append(_metrics(
                    inner_validation[TARGET_COL],
                    model.predict_proba(inner_validation[feature_cols].astype(float))[:, 1],
                ))
            scores.append({
                "C": c_value,
                "log_loss": float(np.mean([score["log_loss"] for score in inner_scores])),
                "brier": float(np.mean([score["brier"] for score in inner_scores])),
            })
        selected = min(scores, key=lambda score: (score["log_loss"], score["brier"]))
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, C=selected["C"])
        )
        model.fit(
            outer_train[feature_cols].astype(float),
            outer_train[TARGET_COL].astype(int),
        )
        y = outer_validation[TARGET_COL].astype(int).to_numpy()
        p = model.predict_proba(outer_validation[feature_cols].astype(float))[:, 1]
        labels.append(y)
        predictions.append(p)
        game_ids.append(outer_validation["game_id"].to_numpy())
        seasons.append({
            "train_end": train_end,
            "validation_end": validation_end,
            "selected": selected,
            "metrics": _metrics(y, p),
        })
    y = np.concatenate(labels)
    p = np.concatenate(predictions)
    return {
        "metrics": _metrics(y, p),
        "outer_folds": seasons,
        "game_bootstrap_95": _game_bootstrap_intervals(
            np.concatenate(game_ids), y, {"logistic": p}
        ),
        "oof_rows": int(len(y)),
    }


def nested_game_xgb_vs_logistic(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    feature_sets: dict[str, list[str]] | None = None,
    n_estimators: int = 150,
    n_jobs: int = 2,
    c_grid: tuple[float, ...] = (0.5, 1.0, 3.0, 10.0, 30.0, 100.0),
) -> dict:
    """Paired rolling-fold comparison with inner selection for both models."""
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = []
    xgb_predictions = []
    logistic_predictions = []
    game_ids = []
    folds = []
    if feature_sets is None:
        feature_sets = {"selected": feature_cols or FEATURE_COLS}
    for train_idx, validation_idx, train_end, validation_end in walk_forward_game_folds(df):
        outer_train = df.iloc[train_idx]
        outer_validation = df.iloc[validation_idx]
        inner_folds = walk_forward_game_folds(
            outer_train,
            min_train_games=500,
            validation_games=500,
            step_games=500,
        )
        logistic_scores = []
        for set_name, selected_features in feature_sets.items():
            for c_value in c_grid:
                scores = []
                for inner_train_idx, inner_validation_idx, _, _ in inner_folds:
                    inner_train = outer_train.iloc[inner_train_idx]
                    inner_validation = outer_train.iloc[inner_validation_idx]
                    logistic = make_pipeline(
                        StandardScaler(), LogisticRegression(max_iter=1000, C=c_value)
                    )
                    logistic.fit(
                        inner_train[selected_features].astype(float),
                        inner_train[TARGET_COL].astype(int),
                    )
                    scores.append(_metrics(
                        inner_validation[TARGET_COL],
                        logistic.predict_proba(
                            inner_validation[selected_features].astype(float)
                        )[:, 1],
                    ))
                logistic_scores.append({
                    "feature_set": set_name,
                    "C": c_value,
                    "log_loss": float(np.mean([score["log_loss"] for score in scores])),
                    "brier": float(np.mean([score["brier"] for score in scores])),
                })
        selected = min(logistic_scores, key=lambda score: (score["log_loss"], score["brier"]))
        selected_features = feature_sets[selected["feature_set"]]
        xgb_iterations = []
        for inner_train_idx, inner_validation_idx, _, _ in inner_folds:
            xgb_model = fit_early_stopped_xgb(
                outer_train.iloc[inner_train_idx],
                outer_train.iloc[inner_validation_idx],
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                feature_cols=selected_features,
            )
            xgb_iterations.append(
                int(getattr(xgb_model, "best_iteration", n_estimators - 1))
            )
        xgb_params = _xgb_params(
            n_estimators=max(1, int(np.mean(xgb_iterations)) + 1),
            n_jobs=n_jobs,
        )
        xgb_params.pop("early_stopping_rounds", None)
        final_xgb = xgb.XGBClassifier(**xgb_params)
        final_xgb.fit(
            outer_train[selected_features].astype(float),
            outer_train[TARGET_COL].astype(int),
            verbose=False,
        )
        final_logistic = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, C=selected["C"])
        )
        final_logistic.fit(
            outer_train[selected_features].astype(float),
            outer_train[TARGET_COL].astype(int),
        )
        labels.append(outer_validation[TARGET_COL].astype(int).to_numpy())
        xgb_predictions.append(
            final_xgb.predict_proba(outer_validation[selected_features].astype(float))[:, 1]
        )
        logistic_predictions.append(
            final_logistic.predict_proba(
                outer_validation[selected_features].astype(float)
            )[:, 1]
        )
        game_ids.append(outer_validation["game_id"].to_numpy())
        folds.append({
            "train_end": train_end,
            "validation_end": validation_end,
            "xgb_best_iterations": xgb_iterations,
            "selected": selected,
            "xgb_metrics": _metrics(labels[-1], xgb_predictions[-1]),
            "logistic_metrics": _metrics(labels[-1], logistic_predictions[-1]),
        })
    y = np.concatenate(labels)
    xgb_p = np.concatenate(xgb_predictions)
    logistic_p = np.concatenate(logistic_predictions)
    return {
        "xgb_metrics": _metrics(y, xgb_p),
        "logistic_metrics": _metrics(y, logistic_p),
        "paired_logistic_minus_xgb": _paired_game_bootstrap_delta(
            np.concatenate(game_ids), y, xgb_p, logistic_p
        ),
        "outer_folds": folds,
        "oof_rows": int(len(y)),
    }


def nested_game_margin_benchmark(
    df: pd.DataFrame,
    feature_cols: list[str],
    alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
) -> dict:
    """Evaluate a regularized remaining-margin Gaussian model without leakage."""
    from scipy.stats import norm
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if "final_margin" not in df.columns:
        raise ValueError("margin benchmark requires a final_margin evaluation target")
    labels = []
    predictions = []
    game_ids = []
    folds = []
    for train_idx, validation_idx, train_end, validation_end in walk_forward_game_folds(df):
        outer_train = df.iloc[train_idx]
        outer_validation = df.iloc[validation_idx]
        inner_folds = walk_forward_game_folds(
            outer_train,
            min_train_games=500,
            validation_games=500,
            step_games=500,
        )
        candidates = []
        for alpha in alpha_grid:
            residuals = []
            losses = []
            for inner_train_idx, inner_validation_idx, _, _ in inner_folds:
                inner_train = outer_train.iloc[inner_train_idx]
                inner_validation = outer_train.iloc[inner_validation_idx]
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(
                    inner_train[feature_cols].astype(float),
                    inner_train["final_margin"].astype(float),
                )
                margin = model.predict(inner_validation[feature_cols].astype(float))
                residual = inner_validation["final_margin"].to_numpy() - margin
                residuals.extend(residual.tolist())
            sigma = max(float(np.std(residuals, ddof=1)), 1.0)
            # Inner selection uses a probability score derived from held-out residuals.
            for inner_train_idx, inner_validation_idx, _, _ in inner_folds:
                inner_train = outer_train.iloc[inner_train_idx]
                inner_validation = outer_train.iloc[inner_validation_idx]
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(
                    inner_train[feature_cols].astype(float),
                    inner_train["final_margin"].astype(float),
                )
                probability = norm.cdf(
                    model.predict(inner_validation[feature_cols].astype(float)) / sigma
                )
                losses.append(_metrics(inner_validation[TARGET_COL], probability))
            candidates.append({
                "alpha": alpha,
                "sigma": sigma,
                "log_loss": float(np.mean([loss["log_loss"] for loss in losses])),
                "brier": float(np.mean([loss["brier"] for loss in losses])),
            })
        selected = min(candidates, key=lambda item: (item["log_loss"], item["brier"]))
        final_model = make_pipeline(StandardScaler(), Ridge(alpha=selected["alpha"]))
        final_model.fit(
            outer_train[feature_cols].astype(float),
            outer_train["final_margin"].astype(float),
        )
        probability = norm.cdf(
            final_model.predict(outer_validation[feature_cols].astype(float))
            / selected["sigma"]
        )
        y = outer_validation[TARGET_COL].astype(int).to_numpy()
        labels.append(y)
        predictions.append(probability)
        game_ids.append(outer_validation["game_id"].to_numpy())
        folds.append({
            "train_end": train_end,
            "validation_end": validation_end,
            "selected": selected,
            "metrics": _metrics(y, probability),
        })
    y = np.concatenate(labels)
    p = np.concatenate(predictions)
    return {
        "metrics": _metrics(y, p),
        "outer_folds": folds,
        "game_bootstrap_95": _game_bootstrap_intervals(
            np.concatenate(game_ids), y, {"margin": p}
        ),
        "oof_rows": int(len(y)),
    }


def nested_game_dynamic_distribution_benchmark(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    alpha: float = 10.0,
) -> dict:
    """Model the remaining score margin with time-bucketed uncertainty.

    The final margin is an evaluation target only. Inputs are current-event
    state and pregame features, and all fitting is repeated inside rolling
    game-time folds.  The time-bucketed residual scale gives the Gaussian
    conversion a simple heteroscedastic score-state model.
    """
    from scipy.stats import norm
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if {"final_margin", "score_diff"} - set(df.columns):
        raise ValueError("dynamic distribution benchmark requires final_margin and score_diff")
    feature_cols = feature_cols or FEATURE_COLS
    work = df.copy()
    work["remaining_margin"] = work["final_margin"] - work["score_diff"]
    labels = []
    predictions = []
    game_ids = []
    folds = []
    for train_idx, validation_idx, train_end, validation_end in walk_forward_game_folds(work):
        train_df = work.iloc[train_idx]
        validation_df = work.iloc[validation_idx]
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(
            train_df[feature_cols].astype(float),
            train_df["remaining_margin"].astype(float),
        )
        train_residual = train_df["remaining_margin"].to_numpy() - model.predict(
            train_df[feature_cols].astype(float)
        )
        scales = []
        train_time = train_df["seconds_remaining"].to_numpy()
        for lower, upper in ((60.0, 300.0), (300.0, 720.0), (720.0, np.inf)):
            residual = train_residual[(train_time > lower) & (train_time <= upper)]
            scales.append(max(float(np.std(residual, ddof=1)), 1.0))
        validation_time = validation_df["seconds_remaining"].to_numpy()
        sigma = np.where(
            validation_time <= 300.0,
            scales[0],
            np.where(validation_time <= 720.0, scales[1], scales[2]),
        )
        predicted_margin = validation_df["score_diff"].to_numpy() + model.predict(
            validation_df[feature_cols].astype(float)
        )
        probability = norm.cdf(predicted_margin / sigma)
        y = validation_df[TARGET_COL].astype(int).to_numpy()
        labels.append(y)
        predictions.append(probability)
        game_ids.append(validation_df["game_id"].to_numpy())
        folds.append({
            "train_end": train_end,
            "validation_end": validation_end,
            "sigma": scales,
            "metrics": _metrics(y, probability),
        })
    y = np.concatenate(labels)
    p = np.concatenate(predictions)
    return {
        "metrics": _metrics(y, p),
        "outer_folds": folds,
        "game_bootstrap_95": _game_bootstrap_intervals(
            np.concatenate(game_ids), y, {"dynamic_distribution": p}
        ),
        "oof_rows": int(len(y)),
    }


def nested_game_horizon_logistic_benchmark(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    horizon_edges: tuple[float, ...] = (60.0, 300.0, 720.0, 2881.0),
    c_value: float = 1.0,
) -> dict:
    """Fit separate historical logistic models for distinct clock horizons."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feature_cols = feature_cols or FEATURE_COLS
    labels = []
    predictions = []
    game_ids = []
    folds = []
    for train_idx, validation_idx, train_end, validation_end in walk_forward_game_folds(df):
        train_df = df.iloc[train_idx]
        validation_df = df.iloc[validation_idx]
        probability = np.full(len(validation_df), 0.5, dtype=float)
        for lower, upper in zip(horizon_edges[:-1], horizon_edges[1:], strict=True):
            train_mask = (
                (train_df["seconds_remaining"] > lower)
                & (train_df["seconds_remaining"] <= upper)
            )
            validation_mask = (
                (validation_df["seconds_remaining"] > lower)
                & (validation_df["seconds_remaining"] <= upper)
            )
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c_value, max_iter=1000),
            )
            model.fit(
                train_df.loc[train_mask, feature_cols].astype(float),
                train_df.loc[train_mask, TARGET_COL].astype(int),
            )
            probability[validation_mask.to_numpy()] = model.predict_proba(
                validation_df.loc[validation_mask, feature_cols].astype(float)
            )[:, 1]
        y = validation_df[TARGET_COL].astype(int).to_numpy()
        labels.append(y)
        predictions.append(probability)
        game_ids.append(validation_df["game_id"].to_numpy())
        folds.append({
            "train_end": train_end,
            "validation_end": validation_end,
            "metrics": _metrics(y, probability),
        })
    y = np.concatenate(labels)
    p = np.concatenate(predictions)
    return {
        "metrics": _metrics(y, p),
        "outer_folds": folds,
        "game_bootstrap_95": _game_bootstrap_intervals(
            np.concatenate(game_ids), y, {"horizon_logistic": p}
        ),
        "oof_rows": int(len(y)),
    }


def beta_calibration_benchmark(
    labels,
    raw_probabilities,
    regularization: float = 10.0,
) -> tuple[BetaCalibrator, dict[str, float]]:
    calibrator = BetaCalibrator(regularization=regularization).fit(
        raw_probabilities, labels
    )
    return calibrator, _metrics(labels, calibrator.predict(raw_probabilities))


def _pooled_oof_ensemble_benchmark(
    df: pd.DataFrame,
    min_train_seasons: int = 2,
    n_estimators: int = 200,
    n_jobs: int = 2,
    include_hist: bool = False,
) -> dict:
    """Build out-of-time OOF predictions and a nonnegative model blend."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _validate(df)
    labels = []
    predictions = []
    best_iterations = []
    for train_idx, validation_idx, _, _ in walk_forward_folds(
        df, min_train_seasons=min_train_seasons
    ):
        train_df = df.iloc[train_idx]
        validation_df = df.iloc[validation_idx]
        x_train = train_df[FEATURE_COLS].astype(float)
        x_validation = validation_df[FEATURE_COLS].astype(float)
        y_train = train_df[TARGET_COL].astype(int)
        xgb_model = fit_early_stopped_xgb(
            train_df,
            validation_df,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
        )
        best_iterations.append(int(getattr(xgb_model, "best_iteration", n_estimators - 1)))
        logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=0.5),
        )
        logistic.fit(x_train, y_train)
        fold_predictions = [
            xgb_model.predict_proba(x_validation)[:, 1],
            logistic.predict_proba(x_validation)[:, 1],
        ]
        if include_hist:
            hist = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                random_state=42,
            )
            hist.fit(x_train, y_train)
            fold_predictions.append(hist.predict_proba(x_validation)[:, 1])
        predictions.append(np.column_stack(fold_predictions))
        labels.append(validation_df[TARGET_COL].astype(int).to_numpy())

    matrix = np.vstack(predictions)
    y = np.concatenate(labels)
    weights = _blend_weights(matrix, y)
    blended = np.clip(matrix @ weights, 1e-6, 1 - 1e-6)
    beta = BetaCalibrator().fit(blended, y)
    model_names = ["xgboost", "logistic"]
    if include_hist:
        model_names.append("hist_gradient_boosting")
    return {
        "weights": weights,
        "base_metrics": {
            name: _metrics(y, matrix[:, index])
            for index, name in enumerate(model_names)
        },
        "blend_metrics": _metrics(y, blended),
        "blend_beta_metrics": _metrics(y, beta.predict(blended)),
        "oof_rows": int(len(y)),
        "xgb_best_iterations": best_iterations,
    }


def feature_group_ablation_benchmark(
    df: pd.DataFrame,
    min_train_seasons: int = 2,
    n_estimators: int = 200,
    n_jobs: int = 2,
) -> dict[str, dict]:
    """Measure nested outer-season impact of removing each feature group."""
    results = {}
    for group_name, removed in FEATURE_GROUPS.items():
        selected = [column for column in FEATURE_COLS if column not in removed]
        results[group_name] = oof_ensemble_benchmark(
            df,
            min_train_seasons=min_train_seasons,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            include_hist=False,
            feature_cols=selected,
        )
    return results


def nested_logistic_benchmark(
    df: pd.DataFrame,
    min_train_seasons: int = 2,
    c_grid: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0, 10.0, 100.0),
    feature_cols: list[str] | None = None,
) -> dict:
    """Select logistic regularization only inside historical inner folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _validate(df)
    feature_cols = feature_cols or FEATURE_COLS
    outer_labels = []
    outer_predictions = []
    season_rows = []
    for train_idx, validation_idx, _, validation_season in walk_forward_folds(
        df, min_train_seasons=min_train_seasons
    ):
        outer_train = df.iloc[train_idx]
        outer_validation = df.iloc[validation_idx]
        scores = []
        for c_value in c_grid:
            inner_scores = []
            for inner_train_idx, inner_validation_idx, _, _ in walk_forward_folds(
                outer_train, min_train_seasons=1
            ):
                inner_train = outer_train.iloc[inner_train_idx]
                inner_validation = outer_train.iloc[inner_validation_idx]
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, C=c_value),
                )
                model.fit(
                    inner_train[feature_cols].astype(float),
                    inner_train[TARGET_COL].astype(int),
                )
                probabilities = model.predict_proba(
                    inner_validation[feature_cols].astype(float)
                )[:, 1]
                inner_scores.append(_metrics(inner_validation[TARGET_COL], probabilities))
            scores.append({
                "C": c_value,
                "log_loss": float(np.mean([score["log_loss"] for score in inner_scores])),
                "brier": float(np.mean([score["brier"] for score in inner_scores])),
            })
        selected = min(scores, key=lambda score: (score["log_loss"], score["brier"]))
        final_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=selected["C"]),
        )
        final_model.fit(
            outer_train[feature_cols].astype(float),
            outer_train[TARGET_COL].astype(int),
        )
        labels = outer_validation[TARGET_COL].astype(int).to_numpy()
        probabilities = final_model.predict_proba(
            outer_validation[feature_cols].astype(float)
        )[:, 1]
        outer_labels.append(labels)
        outer_predictions.append(probabilities)
        season_rows.append({
            "validation_season": validation_season,
            "selected_C": selected["C"],
            "inner_scores": scores,
            "outer_metrics": _metrics(labels, probabilities),
        })
    labels = np.concatenate(outer_labels)
    probabilities = np.concatenate(outer_predictions)
    return {
        "metrics": _metrics(labels, probabilities),
        "outer_seasons": season_rows,
        "oof_rows": int(len(labels)),
        "feature_cols": feature_cols,
    }


def nested_spline_logistic_benchmark(
    df: pd.DataFrame,
    min_train_seasons: int = 2,
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
    knot_grid: tuple[int, ...] = (3, 4, 5),
    feature_cols: list[str] | None = None,
) -> dict:
    """Evaluate a low-complexity spline logistic model with nested tuning."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    _validate(df)
    feature_cols = feature_cols or FEATURE_COLS
    spline_cols = [
        column for column in (
            "seconds_remaining", "score_diff", "score_diff_norm", "run_diff"
        ) if column in feature_cols
    ]
    linear_cols = [column for column in feature_cols if column not in spline_cols]

    def build_model(c_value: float, knots: int):
        transformer = ColumnTransformer([
            (
                "spline",
                make_pipeline(
                    SplineTransformer(
                        n_knots=knots,
                        degree=2,
                        include_bias=False,
                    ),
                    StandardScaler(),
                ),
                spline_cols,
            ),
            ("linear", StandardScaler(), linear_cols),
        ])
        return make_pipeline(
            transformer,
            LogisticRegression(max_iter=1000, C=c_value),
        )

    _validate(df)
    outer_labels = []
    outer_predictions = []
    season_rows = []
    for train_idx, validation_idx, _, validation_season in walk_forward_folds(
        df, min_train_seasons=min_train_seasons
    ):
        outer_train = df.iloc[train_idx]
        outer_validation = df.iloc[validation_idx]
        scores = []
        for knots in knot_grid:
            for c_value in c_grid:
                inner_scores = []
                for inner_train_idx, inner_validation_idx, _, _ in walk_forward_folds(
                    outer_train, min_train_seasons=1
                ):
                    inner_train = outer_train.iloc[inner_train_idx]
                    inner_validation = outer_train.iloc[inner_validation_idx]
                    model = build_model(c_value, knots)
                    model.fit(
                        inner_train[feature_cols].astype(float),
                        inner_train[TARGET_COL].astype(int),
                    )
                    probabilities = model.predict_proba(
                        inner_validation[feature_cols].astype(float)
                    )[:, 1]
                    inner_scores.append(
                        _metrics(inner_validation[TARGET_COL], probabilities)
                    )
                scores.append({
                    "C": c_value,
                    "knots": knots,
                    "log_loss": float(
                        np.mean([score["log_loss"] for score in inner_scores])
                    ),
                    "brier": float(
                        np.mean([score["brier"] for score in inner_scores])
                    ),
                })
        selected = min(scores, key=lambda score: (score["log_loss"], score["brier"]))
        final_model = build_model(selected["C"], selected["knots"])
        final_model.fit(
            outer_train[feature_cols].astype(float),
            outer_train[TARGET_COL].astype(int),
        )
        labels = outer_validation[TARGET_COL].astype(int).to_numpy()
        probabilities = final_model.predict_proba(
            outer_validation[feature_cols].astype(float)
        )[:, 1]
        outer_labels.append(labels)
        outer_predictions.append(probabilities)
        season_rows.append({
            "validation_season": validation_season,
            "selected": selected,
            "outer_metrics": _metrics(labels, probabilities),
        })
    labels = np.concatenate(outer_labels)
    probabilities = np.concatenate(outer_predictions)
    return {
        "metrics": _metrics(labels, probabilities),
        "outer_seasons": season_rows,
        "oof_rows": int(len(labels)),
        "feature_cols": feature_cols,
        "spline_cols": spline_cols,
    }
