"""Out-of-time model and calibration experiments.

This module deliberately keeps experimental candidates separate from the
production ``train`` path.  Every comparison is made on future seasons, with
games kept together wherever cross-validation is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_winprob.training.train import (
    FEATURE_COLS,
    TARGET_COL,
    _validate,
    apply_temperature,
    fit_temperature,
)

DEFAULT_XGB_GRID = (
    {"max_depth": 3, "min_child_weight": 5, "learning_rate": 0.03},
    {"max_depth": 4, "min_child_weight": 5, "learning_rate": 0.03},
    {"max_depth": 5, "min_child_weight": 10, "learning_rate": 0.03},
    {"max_depth": 4, "min_child_weight": 10, "learning_rate": 0.05},
)


def _sample_events_per_game(
    df: pd.DataFrame,
    max_events_per_game: int | None,
    include_terminal: bool = False,
) -> pd.DataFrame:
    """Keep evenly spaced event states while retaining every game."""
    if max_events_per_game is None:
        return df
    if max_events_per_game < 1:
        raise ValueError("max_events_per_game must be positive or None")
    ordered = df.sort_values(["game_id", "event_num"])

    def select_events(game: pd.DataFrame) -> pd.DataFrame:
        eligible = game
        count = min(max_events_per_game, len(eligible))
        return eligible.iloc[np.linspace(0, len(eligible) - 1, count, dtype=int)]

    if not include_terminal:
        terminal_event = ordered.groupby("game_id")["event_num"].transform("max")
        ordered = ordered[ordered["event_num"] < terminal_event]

    return (
        ordered.groupby("game_id", group_keys=False, sort=False)
        .apply(select_events, include_groups=True)
        .reset_index(drop=True)
    )


def _metrics(labels, probabilities) -> dict[str, float]:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    y = np.asarray(labels, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
    }


def _game_bootstrap_intervals(
    game_ids,
    labels,
    predictions: dict[str, np.ndarray],
    samples: int = 300,
    seed: int = 42,
) -> dict[str, dict[str, dict[str, float]]]:
    """Bootstrap complete games, preserving within-game event dependence."""
    ids = np.asarray(game_ids)
    y = np.asarray(labels, dtype=int)
    unique_games = np.unique(ids)
    rows = {name: {metric: [] for metric in ("brier", "log_loss", "roc_auc")}
            for name in predictions}
    rng = np.random.default_rng(seed)
    game_rows = {game: np.flatnonzero(ids == game) for game in unique_games}
    for _ in range(samples):
        sampled_games = rng.choice(unique_games, size=len(unique_games), replace=True)
        indices = np.concatenate([game_rows[game] for game in sampled_games])
        for name, probabilities in predictions.items():
            metrics = _metrics(y[indices], probabilities[indices])
            for metric, value in metrics.items():
                if np.isfinite(value):
                    rows[name][metric].append(value)
    return {
        name: {
            metric: {
                "lower": float(np.percentile(values, 2.5)),
                "upper": float(np.percentile(values, 97.5)),
            }
            for metric, values in metric_values.items()
        }
        for name, metric_values in rows.items()
    }


def _season_start(df: pd.DataFrame) -> pd.Series:
    if "season" in df.columns:
        values = df["season"].astype(str).str.extract(r"^(\d{4})", expand=False)
        return values.astype(int)
    values = df["game_id"].astype(str).str.extract(r"^\d{3}(\d{2})", expand=False)
    if values.isna().any():
        raise ValueError("walk-forward evaluation needs a season column or NBA game IDs")
    return (2000 + values.astype(int)).rename("season_start")


def walk_forward_folds(
    df: pd.DataFrame,
    min_train_seasons: int = 2,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    """Return expanding train/future-validation folds by NBA season."""
    starts = _season_start(df)
    seasons = sorted(starts.unique())
    if len(seasons) <= min_train_seasons:
        raise ValueError("walk-forward evaluation needs more than min_train_seasons")
    folds = []
    for validation_season in seasons[min_train_seasons:]:
        train_mask = starts < validation_season
        validation_mask = starts == validation_season
        folds.append((
            np.flatnonzero(train_mask.to_numpy()),
            np.flatnonzero(validation_mask.to_numpy()),
            int(seasons[0]),
            int(validation_season),
        ))
    return folds


def walk_forward_game_folds(
    df: pd.DataFrame,
    min_train_games: int = 1500,
    validation_games: int = 500,
    step_games: int = 500,
) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    """Create rolling chronological folds using complete games as units."""
    games = (
        df[["game_id"]]
        .drop_duplicates()
        .sort_values("game_id")["game_id"]
        .astype(str)
        .to_numpy()
    )
    if len(games) <= min_train_games + validation_games:
        raise ValueError("not enough games for rolling chronological folds")
    folds = []
    for start in range(
        min_train_games,
        len(games) - validation_games + 1,
        step_games,
    ):
        train_games = set(games[:start])
        validation_games_set = set(games[start:start + validation_games])
        train_mask = df["game_id"].astype(str).isin(train_games).to_numpy()
        validation_mask = df["game_id"].astype(str).isin(validation_games_set).to_numpy()
        folds.append((
            np.flatnonzero(train_mask),
            np.flatnonzero(validation_mask),
            str(games[start - 1]),
            str(games[start + validation_games - 1]),
        ))
    return folds


def nested_game_xgb_benchmark(
    df: pd.DataFrame,
    n_estimators: int = 250,
    n_jobs: int = 2,
    feature_cols: list[str] | None = None,
) -> dict:
    """Evaluate XGBoost on rolling game-time outer folds without outer leakage."""
    import xgboost as xgb

    feature_cols = feature_cols or FEATURE_COLS
    labels = []
    predictions = []
    game_ids = []
    seasons = []
    for train_idx, validation_idx, train_end, validation_end in walk_forward_game_folds(df):
        outer_train = df.iloc[train_idx]
        outer_validation = df.iloc[validation_idx]
        inner_iterations = []
        for inner_train_idx, inner_validation_idx, _, _ in walk_forward_game_folds(
            outer_train,
            min_train_games=500,
            validation_games=500,
            step_games=500,
        ):
            model = fit_early_stopped_xgb(
                outer_train.iloc[inner_train_idx],
                outer_train.iloc[inner_validation_idx],
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                feature_cols=feature_cols,
            )
            inner_iterations.append(
                int(getattr(model, "best_iteration", n_estimators - 1))
            )
        final_params = _xgb_params(
            n_estimators=max(1, int(np.mean(inner_iterations)) + 1),
            n_jobs=n_jobs,
        )
        final_params.pop("early_stopping_rounds", None)
        final_model = xgb.XGBClassifier(**final_params)
        final_model.fit(
            outer_train[feature_cols].astype(float),
            outer_train[TARGET_COL].astype(int),
            verbose=False,
        )
        probabilities = final_model.predict_proba(
            outer_validation[feature_cols].astype(float)
        )[:, 1]
        labels.append(outer_validation[TARGET_COL].astype(int).to_numpy())
        predictions.append(probabilities)
        game_ids.append(outer_validation["game_id"].to_numpy())
        seasons.append({
            "train_end": train_end,
            "validation_end": validation_end,
            "inner_best_iterations": inner_iterations,
            "metrics": _metrics(labels[-1], probabilities),
        })
    y = np.concatenate(labels)
    p = np.concatenate(predictions)
    return {
        "metrics": _metrics(y, p),
        "outer_folds": seasons,
        "game_bootstrap_95": _game_bootstrap_intervals(
            np.concatenate(game_ids), y, {"xgboost": p}
        ),
        "oof_rows": int(len(y)),
    }


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


def _xgb_params(overrides: dict | None = None, n_estimators: int = 200, n_jobs: int = 2) -> dict:
    params = {
        "n_estimators": n_estimators,
        "max_depth": 4,
        "min_child_weight": 5,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": n_jobs,
        "early_stopping_rounds": 50,
    }
    params.update(overrides or {})
    return params


def fit_early_stopped_xgb(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    params: dict | None = None,
    n_estimators: int = 200,
    n_jobs: int = 2,
    feature_cols: list[str] | None = None,
):
    """Fit XGBoost with a future validation set and best-iteration stopping."""
    import xgboost as xgb

    feature_cols = feature_cols or FEATURE_COLS
    model = xgb.XGBClassifier(
        **_xgb_params(params, n_estimators=n_estimators, n_jobs=n_jobs)
    )
    model.fit(
        train_df[feature_cols].astype(float),
        train_df[TARGET_COL].astype(int),
        eval_set=[(
            validation_df[feature_cols].astype(float),
            validation_df[TARGET_COL].astype(int),
        )],
        verbose=False,
    )
    return model


def walk_forward_xgb_benchmark(
    df: pd.DataFrame,
    param_grid: tuple[dict, ...] | list[dict] = DEFAULT_XGB_GRID,
    min_train_seasons: int = 2,
    n_estimators: int = 200,
    n_jobs: int = 2,
) -> pd.DataFrame:
    """Compare early-stopped XGBoost candidates on future seasons."""
    _validate(df)
    rows = []
    for candidate_id, candidate in enumerate(param_grid):
        for train_idx, validation_idx, _first_season, validation_season in walk_forward_folds(
            df, min_train_seasons=min_train_seasons
        ):
            train_df = df.iloc[train_idx]
            validation_df = df.iloc[validation_idx]
            model = fit_early_stopped_xgb(
                train_df,
                validation_df,
                candidate,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
            )
            probabilities = model.predict_proba(
                validation_df[FEATURE_COLS].astype(float)
            )[:, 1]
            metrics = _metrics(validation_df[TARGET_COL], probabilities)
            rows.append({
                "candidate": candidate_id,
                "params": repr(candidate),
                "train_through": validation_season - 1,
                "validation_season": validation_season,
                "best_iteration": int(getattr(model, "best_iteration", n_estimators - 1)),
                **metrics,
            })
    return pd.DataFrame(rows)


class BetaCalibrator:
    """Smooth beta calibration for binary probabilities.

    The fitted map is logistic regression over ``log(p)`` and
    ``-log(1-p)``.  Unlike isotonic regression it is smooth and does not form
    step-function plateaus at the probability tails.
    """

    def __init__(self, regularization: float = 10.0) -> None:
        self.regularization = regularization
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    @staticmethod
    def _features(probabilities) -> np.ndarray:
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        return np.column_stack((np.log(p), -np.log1p(-p)))

    def fit(self, probabilities, labels) -> BetaCalibrator:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=self.regularization, max_iter=1000)
        model.fit(self._features(probabilities), np.asarray(labels, dtype=int))
        self.coef_ = model.coef_[0].astype(float)
        self.intercept_ = float(model.intercept_[0])
        return self

    def predict(self, probabilities) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("BetaCalibrator must be fitted before predict")
        logits = self._features(probabilities) @ self.coef_ + self.intercept_
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))


def beta_calibration_benchmark(
    labels,
    raw_probabilities,
    regularization: float = 10.0,
) -> tuple[BetaCalibrator, dict[str, float]]:
    calibrator = BetaCalibrator(regularization=regularization).fit(
        raw_probabilities, labels
    )
    return calibrator, _metrics(labels, calibrator.predict(raw_probabilities))


def _blend_weights(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize

    def objective(weights):
        return float(np.mean((predictions @ weights - labels) ** 2))

    result = minimize(
        objective,
        np.full(predictions.shape[1], 1.0 / predictions.shape[1]),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * predictions.shape[1],
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
    )
    if not result.success:
        return np.full(predictions.shape[1], 1.0 / predictions.shape[1])
    return np.asarray(result.x, dtype=float)


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


def oof_ensemble_benchmark(
    df: pd.DataFrame,
    min_train_seasons: int = 2,
    n_estimators: int = 200,
    n_jobs: int = 2,
    include_hist: bool = False,
    feature_cols: list[str] | None = None,
    logistic_C: float = 0.5,
) -> dict:
    """Evaluate blend and beta calibration with nested walk-forward folds.

    The outer future season is never used to learn blend weights or the beta
    calibrator. Inner historical folds learn those choices, preventing OOF
    model-selection leakage.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _validate(df)
    feature_cols = feature_cols or FEATURE_COLS
    outer_labels = []
    outer_blends = []
    outer_game_ids = []
    season_rows = []
    model_names = ["xgboost", "logistic"]
    if include_hist:
        model_names.append("hist_gradient_boosting")

    for train_idx, validation_idx, _, validation_season in walk_forward_folds(
        df, min_train_seasons=min_train_seasons
    ):
        outer_train = df.iloc[train_idx]
        outer_validation = df.iloc[validation_idx]
        inner_predictions = []
        inner_labels = []
        inner_iterations = []

        for inner_train_idx, inner_validation_idx, _, _ in walk_forward_folds(
            outer_train, min_train_seasons=1
        ):
            inner_train = outer_train.iloc[inner_train_idx]
            inner_validation = outer_train.iloc[inner_validation_idx]
            xgb_model = fit_early_stopped_xgb(
                inner_train,
                inner_validation,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                feature_cols=feature_cols,
            )
            inner_iterations.append(
                int(getattr(xgb_model, "best_iteration", n_estimators - 1))
            )
            x_train = inner_train[feature_cols].astype(float)
            x_validation = inner_validation[feature_cols].astype(float)
            y_train = inner_train[TARGET_COL].astype(int)
            logistic = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=logistic_C),
            )
            logistic.fit(x_train, y_train)
            fold_predictions = [
                xgb_model.predict_proba(x_validation)[:, 1],
                logistic.predict_proba(x_validation)[:, 1],
            ]
            if include_hist:
                from sklearn.ensemble import HistGradientBoostingClassifier

                hist = HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    random_state=42,
                )
                hist.fit(x_train, y_train)
                fold_predictions.append(hist.predict_proba(x_validation)[:, 1])
            inner_predictions.append(np.column_stack(fold_predictions))
            inner_labels.append(inner_validation[TARGET_COL].astype(int).to_numpy())

        inner_matrix = np.vstack(inner_predictions)
        inner_y = np.concatenate(inner_labels)
        weights = _blend_weights(inner_matrix, inner_y)
        inner_blended = np.clip(inner_matrix @ weights, 1e-6, 1 - 1e-6)
        beta = BetaCalibrator().fit(inner_blended, inner_y)
        temperature, _ = fit_temperature(inner_blended, inner_y)
        logistic_beta = BetaCalibrator().fit(inner_matrix[:, 1], inner_y)
        logistic_temperature, _ = fit_temperature(inner_matrix[:, 1], inner_y)

        final_estimators = max(1, int(np.mean(inner_iterations)) + 1)
        final_params = _xgb_params(n_estimators=final_estimators, n_jobs=n_jobs)
        final_params.pop("early_stopping_rounds", None)
        import xgboost as xgb

        final_xgb = xgb.XGBClassifier(**final_params)
        final_xgb.fit(
            outer_train[feature_cols].astype(float),
            outer_train[TARGET_COL].astype(int),
            verbose=False,
        )
        final_logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=logistic_C),
        )
        final_logistic.fit(
            outer_train[feature_cols].astype(float),
            outer_train[TARGET_COL].astype(int),
        )
        outer_predictions = [
            final_xgb.predict_proba(outer_validation[feature_cols].astype(float))[:, 1],
            final_logistic.predict_proba(outer_validation[feature_cols].astype(float))[:, 1],
        ]
        if include_hist:
            from sklearn.ensemble import HistGradientBoostingClassifier

            final_hist = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                random_state=42,
            )
            final_hist.fit(
                outer_train[feature_cols].astype(float),
                outer_train[TARGET_COL].astype(int),
            )
            outer_predictions.append(
                final_hist.predict_proba(outer_validation[feature_cols].astype(float))[:, 1]
            )
        outer_matrix = np.column_stack(outer_predictions)
        outer_blend = np.clip(outer_matrix @ weights, 1e-6, 1 - 1e-6)
        outer_beta = beta.predict(outer_blend)
        outer_temperature = apply_temperature(outer_blend, temperature)
        outer_logistic_beta = logistic_beta.predict(outer_matrix[:, 1])
        outer_logistic_temperature = apply_temperature(
            outer_matrix[:, 1], logistic_temperature
        )
        labels = outer_validation[TARGET_COL].astype(int).to_numpy()
        outer_labels.append(labels)
        outer_game_ids.append(outer_validation["game_id"].to_numpy())
        outer_blends.append(
            np.column_stack((
                outer_matrix,
                outer_blend,
                outer_beta,
                outer_temperature,
                outer_logistic_beta,
                outer_logistic_temperature,
            ))
        )
        season_rows.append({
            "validation_season": validation_season,
            "weights": weights.tolist(),
            "inner_best_iterations": inner_iterations,
            "final_n_estimators": final_estimators,
            "temperature": temperature,
            "xgboost_metrics": _metrics(labels, outer_matrix[:, 0]),
            "logistic_metrics": _metrics(labels, outer_matrix[:, 1]),
            "blend_metrics": _metrics(labels, outer_blend),
            "blend_beta_metrics": _metrics(labels, outer_beta),
            "blend_temperature_metrics": _metrics(labels, outer_temperature),
            "logistic_beta_metrics": _metrics(labels, outer_logistic_beta),
            "logistic_temperature_metrics": _metrics(
                labels, outer_logistic_temperature
            ),
        })

    y = np.concatenate(outer_labels)
    predictions = np.vstack(outer_blends)
    game_ids = np.concatenate(outer_game_ids)
    bootstrap = _game_bootstrap_intervals(
        game_ids,
        y,
        {
            "xgboost": predictions[:, 0],
            "logistic": predictions[:, 1],
            "blend": predictions[:, 2],
            "blend_beta": predictions[:, 3],
            "blend_temperature": predictions[:, 4],
            "logistic_beta": predictions[:, 5],
            "logistic_temperature": predictions[:, 6],
        },
    )
    return {
        "base_metrics": {
            "xgboost": _metrics(y, predictions[:, 0]),
            "logistic": _metrics(y, predictions[:, 1]),
        },
        "blend_metrics": _metrics(y, predictions[:, 2]),
        "blend_beta_metrics": _metrics(y, predictions[:, 3]),
        "blend_temperature_metrics": _metrics(y, predictions[:, 4]),
        "logistic_beta_metrics": _metrics(y, predictions[:, 5]),
        "logistic_temperature_metrics": _metrics(y, predictions[:, 6]),
        "oof_rows": int(len(y)),
        "game_bootstrap_95": bootstrap,
        "outer_seasons": season_rows,
        "model_names": model_names,
    }


FEATURE_GROUPS = {
    "live_state": [
        "seconds_remaining", "seconds_elapsed", "score_diff", "score_diff_norm",
        "run_home", "run_away", "run_diff", "is_overtime",
    ],
    "pregame_context": [
        "home_win_pct", "home_avg_margin", "home_streak",
        "away_win_pct", "away_avg_margin", "away_streak",
    ],
    "venue_form": [
        "home_venue_win_pct", "home_venue_avg_margin",
        "away_venue_win_pct", "away_venue_avg_margin",
    ],
    "elo": ["home_elo_rating", "away_elo_rating"],
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


def nested_logistic_feature_selection_benchmark(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    min_train_seasons: int = 2,
    c_grid: tuple[float, ...] = (0.5, 1.0, 3.0, 10.0, 30.0, 100.0),
) -> dict:
    """Select feature set and regularization only in inner historical folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _validate(df)
    outer_labels = []
    outer_predictions = []
    outer_game_ids = []
    season_rows = []
    for train_idx, validation_idx, _, validation_season in walk_forward_folds(
        df, min_train_seasons=min_train_seasons
    ):
        outer_train = df.iloc[train_idx]
        outer_validation = df.iloc[validation_idx]
        candidates = []
        for set_name, feature_cols in feature_sets.items():
            for c_value in c_grid:
                scores = []
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
                    scores.append(_metrics(inner_validation[TARGET_COL], probabilities))
                candidates.append({
                    "feature_set": set_name,
                    "C": c_value,
                    "log_loss": float(np.mean([score["log_loss"] for score in scores])),
                    "brier": float(np.mean([score["brier"] for score in scores])),
                })
        selected = min(candidates, key=lambda score: (score["log_loss"], score["brier"]))
        selected_features = feature_sets[selected["feature_set"]]
        final_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=selected["C"]),
        )
        final_model.fit(
            outer_train[selected_features].astype(float),
            outer_train[TARGET_COL].astype(int),
        )
        labels = outer_validation[TARGET_COL].astype(int).to_numpy()
        probabilities = final_model.predict_proba(
            outer_validation[selected_features].astype(float)
        )[:, 1]
        outer_labels.append(labels)
        outer_predictions.append(probabilities)
        outer_game_ids.append(outer_validation["game_id"].to_numpy())
        season_rows.append({
            "validation_season": validation_season,
            "selected": selected,
            "outer_metrics": _metrics(labels, probabilities),
        })
    labels = np.concatenate(outer_labels)
    probabilities = np.concatenate(outer_predictions)
    bootstrap = _game_bootstrap_intervals(
        np.concatenate(outer_game_ids),
        labels,
        {"logistic": probabilities},
    )
    return {
        "metrics": _metrics(labels, probabilities),
        "outer_seasons": season_rows,
        "oof_rows": int(len(labels)),
        "game_bootstrap_95": bootstrap,
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


@dataclass
class MarginDistributionModel:
    """Predict final margin and convert it to a win probability."""

    model: object
    sigma: float

    def predict_margin(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(frame[FEATURE_COLS].astype(float)))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        from scipy.stats import norm

        return norm.cdf(self.predict_margin(frame) / max(self.sigma, 1e-6))


def _final_margin(df: pd.DataFrame) -> pd.Series:
    if {"home_score", "away_score"} - set(df.columns):
        raise ValueError("margin model requires home_score and away_score columns")
    ordered = df.sort_values(["game_id", "event_num"])
    margins = (ordered["home_score"] - ordered["away_score"]).groupby(ordered["game_id"]).last()
    return df["game_id"].map(margins).astype(float)


def margin_distribution_benchmark(
    df: pd.DataFrame,
    min_train_seasons: int = 2,
    n_estimators: int = 200,
    n_jobs: int = 2,
) -> pd.DataFrame:
    """Evaluate a final-margin Gaussian model on future seasons."""
    import xgboost as xgb

    _validate(df)
    work = df.copy()
    work["final_margin"] = _final_margin(work)
    rows = []
    for train_idx, validation_idx, _, validation_season in walk_forward_folds(
        work, min_train_seasons=min_train_seasons
    ):
        train_df = work.iloc[train_idx]
        validation_df = work.iloc[validation_idx]
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=42,
            n_jobs=n_jobs,
            early_stopping_rounds=50,
        )
        # Each row is an event, but the target is one final margin per game.
        # Weighting by inverse event count prevents long games dominating the fit.
        sample_weights = (
            1.0 / train_df.groupby("game_id")["game_id"].transform("size")
        ).to_numpy()
        model.fit(
            train_df[FEATURE_COLS].astype(float),
            train_df["final_margin"],
            sample_weight=sample_weights,
            eval_set=[(validation_df[FEATURE_COLS].astype(float), validation_df["final_margin"])],
            verbose=False,
        )
        train_residual = train_df["final_margin"].to_numpy() - model.predict(
            train_df[FEATURE_COLS].astype(float)
        )
        sigma = float(np.std(train_residual, ddof=1))
        candidate = MarginDistributionModel(model, sigma)
        metrics = _metrics(validation_df[TARGET_COL], candidate.predict_proba(validation_df))
        rows.append({"validation_season": validation_season, "sigma": sigma, **metrics})
    return pd.DataFrame(rows)


def run_advanced_benchmark(
    df: pd.DataFrame,
    min_train_seasons: int = 2,
    n_estimators: int = 200,
    n_jobs: int = 2,
    candidates: int = 2,
    max_events_per_game: int | None = 25,
) -> dict:
    """Run all requested experiments and return serializable summaries."""
    if not 1 <= candidates <= len(DEFAULT_XGB_GRID):
        raise ValueError(f"candidates must be between 1 and {len(DEFAULT_XGB_GRID)}")
    work = _sample_events_per_game(df, max_events_per_game)
    xgb_results = walk_forward_xgb_benchmark(
        work,
        param_grid=DEFAULT_XGB_GRID[:candidates],
        min_train_seasons=min_train_seasons,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )
    ensemble = oof_ensemble_benchmark(
        work,
        min_train_seasons=min_train_seasons,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )
    margin = margin_distribution_benchmark(
        work,
        min_train_seasons=min_train_seasons,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )
    return {
        "config": {
            "n_estimators": n_estimators,
            "n_jobs": n_jobs,
            "candidates": candidates,
            "max_events_per_game": max_events_per_game,
            "rows": int(len(work)),
            "games": int(work["game_id"].nunique()),
        },
        "walk_forward_xgb": xgb_results.to_dict(orient="records"),
        "oof_ensemble": ensemble,
        "margin_distribution": margin.to_dict(orient="records"),
    }
