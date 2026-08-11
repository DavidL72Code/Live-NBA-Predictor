"""Out-of-time model and calibration experiments.

This module deliberately keeps experimental candidates separate from the
production ``train`` path.  Every comparison is made on future seasons, with
games kept together wherever cross-validation is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_winprob.training.train import FEATURE_COLS, TARGET_COL, _validate

DEFAULT_XGB_GRID = (
    {"max_depth": 3, "min_child_weight": 5, "learning_rate": 0.03},
    {"max_depth": 4, "min_child_weight": 5, "learning_rate": 0.03},
    {"max_depth": 5, "min_child_weight": 10, "learning_rate": 0.03},
    {"max_depth": 4, "min_child_weight": 10, "learning_rate": 0.05},
)


def _sample_events_per_game(df: pd.DataFrame, max_events_per_game: int | None) -> pd.DataFrame:
    """Keep evenly spaced event states while retaining every game."""
    if max_events_per_game is None:
        return df
    if max_events_per_game < 1:
        raise ValueError("max_events_per_game must be positive or None")
    ordered = df.sort_values(["game_id", "event_num"])
    return (
        ordered.groupby("game_id", group_keys=False, sort=False)
        .apply(
            lambda game: game.iloc[
                np.linspace(
                    0,
                    len(game) - 1,
                    min(max_events_per_game, len(game)),
                    dtype=int,
                )
            ],
            include_groups=True,
        )
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
):
    """Fit XGBoost with a future validation set and best-iteration stopping."""
    import xgboost as xgb

    model = xgb.XGBClassifier(
        **_xgb_params(params, n_estimators=n_estimators, n_jobs=n_jobs)
    )
    model.fit(
        train_df[FEATURE_COLS].astype(float),
        train_df[TARGET_COL].astype(int),
        eval_set=[(
            validation_df[FEATURE_COLS].astype(float),
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


def oof_ensemble_benchmark(
    df: pd.DataFrame,
    min_train_seasons: int = 2,
    n_estimators: int = 200,
    n_jobs: int = 2,
) -> dict:
    """Build out-of-time OOF predictions and a nonnegative model blend."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _validate(df)
    labels = []
    predictions = []
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
        logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=0.5),
        )
        logistic.fit(x_train, y_train)
        hist = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42,
        )
        hist.fit(x_train, y_train)
        predictions.append(np.column_stack((
            xgb_model.predict_proba(x_validation)[:, 1],
            logistic.predict_proba(x_validation)[:, 1],
            hist.predict_proba(x_validation)[:, 1],
        )))
        labels.append(validation_df[TARGET_COL].astype(int).to_numpy())

    matrix = np.vstack(predictions)
    y = np.concatenate(labels)
    weights = _blend_weights(matrix, y)
    blended = np.clip(matrix @ weights, 1e-6, 1 - 1e-6)
    beta = BetaCalibrator().fit(blended, y)
    return {
        "weights": weights,
        "base_metrics": {
            name: _metrics(y, matrix[:, index])
            for index, name in enumerate(("xgboost", "logistic", "hist_gradient_boosting"))
        },
        "blend_metrics": _metrics(y, blended),
        "blend_beta_metrics": _metrics(y, beta.predict(blended)),
        "oof_rows": int(len(y)),
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
