"""XGBoost training pipeline with MLflow tracking.

Reads the offline feature table (parquet or Postgres), trains an XGBoost
win-probability model, calibrates it with isotonic regression on a held-out
calibration fold, and logs everything to MLflow.

Split strategy:
    Train (80%) → fit XGBoost
    Calibration (10%) → fit isotonic calibrator on XGBoost raw outputs
    Test (10%) → evaluate the full stack, never touched until final eval

All splits are by game (not row) to prevent future-score leakage.

Feature columns are defined once here (``FEATURE_COLS``) and imported by the
serving layer — single source of truth for what the model expects at inference.
"""

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "seconds_remaining",
    "seconds_elapsed",
    "score_diff",
    "score_diff_norm",
    "run_home",
    "run_away",
    "run_diff",
    "is_overtime",
    # Pre-game team context (season stats entering this game)
    "home_win_pct",
    "home_avg_margin",
    "home_streak",
    "away_win_pct",
    "away_avg_margin",
    "away_streak",
    # Team-specific venue form and opponent-adjusted strength.
    "home_venue_win_pct",
    "home_venue_avg_margin",
    "away_venue_win_pct",
    "away_venue_avg_margin",
    "home_elo_rating",
    "away_elo_rating",
]

TARGET_COL = "home_win"


def apply_temperature(probabilities, temperature: float):
    """Apply temperature scaling to probabilities without changing rank order."""
    import numpy as np

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(probs / (1 - probs))
    return 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -40, 40)))


def fit_temperature(probabilities, labels) -> tuple[float, float]:
    """Fit one global temperature by minimizing calibration-set log loss.

    Unlike isotonic regression, this has one degree of freedom and operates in
    the same per-event probability space used by the live serving path.
    Returns ``(temperature, optimized_log_loss)``.
    """
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import log_loss

    probs = apply_temperature(probabilities, 1.0)
    labels = list(labels)

    def objective(log_temperature: float) -> float:
        temperature = float(math.exp(log_temperature))
        return log_loss(labels, apply_temperature(probs, temperature), labels=[0, 1])

    result = minimize_scalar(objective, bounds=(-2.0, 2.0), method="bounded")
    temperature = float(math.exp(result.x))
    return temperature, float(result.fun)


def load_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    _validate(df)
    return df


def load_postgres(dsn: str, seasons: list[str] | None = None) -> pd.DataFrame:
    from nba_winprob.store.offline import OfflineStore

    df = OfflineStore(dsn=dsn).read_training_data(seasons=seasons)
    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"training data missing columns: {missing}")
    if df[TARGET_COL].isna().any():
        raise ValueError("training data contains rows with null home_win labels")


def reliability_diagram(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Return a reliability diagram DataFrame for logging / inspection."""
    import numpy as np

    buckets = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(buckets[:-1], buckets[1:], strict=True):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bucket": f"{lo:.0%}–{hi:.0%}",
            "n": int(mask.sum()),
            "mean_pred": round(float(y_prob[mask].mean()), 4),
            "actual_rate": round(float(y_true[mask].mean()), 4),
            "gap": round(float(y_prob[mask].mean() - y_true[mask].mean()), 4),
        })
    return pd.DataFrame(rows)


def train(
    df: pd.DataFrame,
    experiment_name: str = "nba-winprob",
    run_name: str | None = None,
    test_size: float = 0.1,
    cal_size: float = 0.1,
    xgb_params: dict | None = None,
    mlflow_uri: str | None = None,
) -> object:
    """Train XGBoost, calibrate with isotonic regression, log everything to MLflow.

    Returns the MLflow run object.
    """
    import mlflow
    import mlflow.xgboost
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit

    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    else:
        from nba_winprob.config import get_settings

        uri = get_settings().mlflow_tracking_uri
        if uri:
            mlflow.set_tracking_uri(uri)

    params = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
    }
    if xgb_params:
        params.update(xgb_params)

    # ── 3-way split by game ────────────────────────────────────────────────
    # Step 1: carve out test set
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    dev_idx, test_idx = next(splitter.split(df, groups=df["game_id"]))
    dev_df, test_df = df.iloc[dev_idx], df.iloc[test_idx]

    # Step 2: split remaining dev into train / calibration
    cal_fraction = cal_size / (1 - test_size)
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=cal_fraction, random_state=42)
    train_idx, cal_idx = next(splitter2.split(dev_df, groups=dev_df["game_id"]))
    train_df, cal_df = dev_df.iloc[train_idx], dev_df.iloc[cal_idx]

    X_train = train_df[FEATURE_COLS].astype(float)
    y_train = train_df[TARGET_COL].astype(int)
    X_cal   = cal_df[FEATURE_COLS].astype(float)
    y_cal   = cal_df[TARGET_COL].astype(int).values
    X_test  = test_df[FEATURE_COLS].astype(float)
    y_test  = test_df[TARGET_COL].astype(int).values

    logger.info(
        "split — train: %d rows (%d games)  cal: %d rows (%d games)  test: %d rows (%d games)",
        len(train_df), train_df["game_id"].nunique(),
        len(cal_df),   cal_df["game_id"].nunique(),
        len(test_df),  test_df["game_id"].nunique(),
    )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:

        # ── 1. Train XGBoost ───────────────────────────────────────────────
        mlflow.log_params(params)
        mlflow.log_params({
            "train_games":    train_df["game_id"].nunique(),
            "cal_games":      cal_df["game_id"].nunique(),
            "test_games":     test_df["game_id"].nunique(),
            "calibration":    "isotonic",
            "feature_cols":   ",".join(FEATURE_COLS),
        })

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)

        # ── 2. Evaluate raw (uncalibrated) on test ─────────────────────────
        y_prob_raw = model.predict_proba(X_test)[:, 1]
        brier_raw  = brier_score_loss(y_test, y_prob_raw)
        auc_raw    = roc_auc_score(y_test, y_prob_raw)
        logger.info("uncalibrated  brier=%.4f  auc=%.4f", brier_raw, auc_raw)
        mlflow.log_metrics({
            "brier_score_raw": brier_raw,
            "roc_auc_raw":     auc_raw,
            "log_loss_raw":    log_loss(y_test, y_prob_raw),
        })

        # ── 3. Fit isotonic calibrator on calibration fold ─────────────────
        y_prob_cal_raw = model.predict_proba(X_cal)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(y_prob_cal_raw, y_cal)

        # ── 4. Evaluate calibrated on test ─────────────────────────────────
        y_prob_cal = calibrator.predict(y_prob_raw)
        brier_cal  = brier_score_loss(y_test, y_prob_cal)
        auc_cal    = roc_auc_score(y_test, y_prob_cal)
        logloss_cal = log_loss(y_test, y_prob_cal)
        brier_improvement = brier_raw - brier_cal

        logger.info(
            "calibrated    brier=%.4f  auc=%.4f  improvement=%.4f",
            brier_cal, auc_cal, brier_improvement,
        )
        mlflow.log_metrics({
            "brier_score":        brier_cal,
            "roc_auc":            auc_cal,
            "log_loss":           logloss_cal,
            "brier_improvement":  brier_improvement,
        })

        # ── 5. Reliability diagrams (before + after) ───────────────────────
        diag_raw = reliability_diagram(y_test, y_prob_raw)
        diag_cal = reliability_diagram(y_test, y_prob_cal)
        logger.info("reliability (raw):\n%s", diag_raw.to_string(index=False))
        logger.info("reliability (calibrated):\n%s", diag_cal.to_string(index=False))

        # Log as CSV artifacts so they're inspectable in the MLflow UI
        diag_raw.to_csv("/tmp/reliability_raw.csv", index=False)
        diag_cal.to_csv("/tmp/reliability_cal.csv", index=False)
        mlflow.log_artifact("/tmp/reliability_raw.csv", artifact_path="calibration")
        mlflow.log_artifact("/tmp/reliability_cal.csv", artifact_path="calibration")

        # ── 6. Save both artifacts ─────────────────────────────────────────
        mlflow.xgboost.log_model(model, name="xgb_model")

        calibrator_path = "/tmp/isotonic_calibrator.pkl"
        with open(calibrator_path, "wb") as f:
            pickle.dump(calibrator, f)
        mlflow.log_artifact(calibrator_path, artifact_path="calibration")

        logger.info("run %s logged", run.info.run_id)

    return run


