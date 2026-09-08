"""Out-of-fold XGBoost training for the retired production model.

Superseded by ``training/logistic.py``. Kept because the artifacts in MLflow and
under ``artifacts/`` were produced by this path, and reproducing them needs it.
"""

from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd

from nba_winprob.training.train import (
    FEATURE_COLS,
    TARGET_COL,
    reliability_diagram,
)

logger = logging.getLogger(__name__)

__all__ = ["train_oof"]


def _quarter(seconds_elapsed: np.ndarray, is_overtime: np.ndarray) -> np.ndarray:
    """Map seconds_elapsed → quarter (1–4) or 5 for overtime."""
    import numpy as np

    q = np.clip((seconds_elapsed / 720).astype(int) + 1, 1, 4)
    q[is_overtime.astype(bool)] = 5
    return q


def train_oof(
    df: pd.DataFrame,
    experiment_name: str = "nba-winprob",
    run_name: str | None = None,
    test_size: float = 0.1,
    n_splits: int = 5,
    cal_seconds_remaining: float | None = None,
    game_level_cal: bool = False,
    quarter_cal: bool = False,
    xgb_params: dict | None = None,
    mlflow_uri: str | None = None,
) -> object:
    """Train with out-of-fold isotonic calibration.

    Instead of a held-out calibration fold (which wastes training data and
    gives isotonic only ~10% of the games), this method uses k-fold CV to
    generate out-of-fold predictions for every training game:

        1. Hold out test_size games as the final eval set (never seen during
           either training or calibration).
        2. On the remaining games, run n_splits-fold GroupKFold:
              - each fold: fit XGBoost on (k-1) folds → predict on the kth
              - collect all OOF predictions (≈ 100% of training games)
        3. Fit IsotonicRegression on the OOF predictions.
             - game_level_cal=True: aggregate rows to one mean prediction per
               game before fitting, so isotonic sees 1,107 independent
               (mean_pred, home_win) points rather than 538k rows that all
               share the same label within each game.
             - cal_seconds_remaining: row-level filter applied before
               aggregation when game_level_cal is also True.
        4. Retrain the final XGBoost on ALL training games (max signal).
        5. Evaluate final_model + calibrator on the held-out test set.

    With 1,230 games and test_size=0.10 this yields ~1,107 games of
    calibration data vs ~123 with the single-fold approach.
    """
    import mlflow
    import mlflow.xgboost
    import numpy as np
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

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

    # ── Step 1: hold out test set ──────────────────────────────────────────
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    dev_idx, test_idx = next(splitter.split(df, groups=df["game_id"]))
    dev_df  = df.iloc[dev_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx]

    X_test = test_df[FEATURE_COLS].astype(float)
    y_test = test_df[TARGET_COL].astype(int).values

    logger.info(
        "OOF split — dev: %d rows (%d games)  test: %d rows (%d games)",
        len(dev_df), dev_df["game_id"].nunique(),
        len(test_df), test_df["game_id"].nunique(),
    )

    # ── Step 2: k-fold OOF predictions on dev set ─────────────────────────
    oof_preds = np.zeros(len(dev_df))

    kfold = GroupKFold(n_splits=n_splits)
    fold_params = dict(params)  # fold models have no early-stopping eval set
    fold_params.pop("eval_metric", None)

    for fold_i, (tr_idx, val_idx) in enumerate(
        kfold.split(dev_df, groups=dev_df["game_id"])
    ):
        X_f = dev_df.iloc[tr_idx][FEATURE_COLS].astype(float)
        y_f = dev_df.iloc[tr_idx][TARGET_COL].astype(int)
        X_v = dev_df.iloc[val_idx][FEATURE_COLS].astype(float)

        fold_model = xgb.XGBClassifier(**fold_params)
        fold_model.fit(X_f, y_f, verbose=False)
        oof_preds[val_idx] = fold_model.predict_proba(X_v)[:, 1]
        logger.info(
            "fold %d/%d — val games: %d",
            fold_i + 1, n_splits, dev_df.iloc[val_idx]["game_id"].nunique(),
        )

    # ── Step 3: fit isotonic calibrator(s) on OOF predictions ────────────
    if quarter_cal:
        # One independent calibrator per quarter: each is trained and served on
        # the same prediction distribution, eliminating covariate shift.
        dev_quarters  = _quarter(dev_df["seconds_elapsed"].values, dev_df["is_overtime"].values)
        calibrators: dict[int, IsotonicRegression] = {}
        cal_games = 0
        for q in sorted(np.unique(dev_quarters)):
            mask      = dev_quarters == q
            q_preds   = oof_preds[mask]
            q_labels  = dev_df.loc[mask, TARGET_COL].astype(int).values
            q_games   = dev_df.loc[mask, "game_id"].nunique()
            cal       = IsotonicRegression(out_of_bounds="clip")
            cal.fit(q_preds, q_labels)
            calibrators[int(q)] = cal
            cal_games += q_games
            label = "OT" if q == 5 else f"Q{q}"
            logger.info("calibrator %s — %d rows from %d games", label, mask.sum(), q_games)
        calibrator = None  # not used in quarter_cal mode
    else:
        if cal_seconds_remaining is not None:
            cal_mask  = dev_df["seconds_remaining"].values <= cal_seconds_remaining
            fit_df    = dev_df[cal_mask].copy()
            fit_preds = oof_preds[cal_mask]
            logger.info(
                "segment filter seconds_remaining <= %g — %d rows from %d games",
                cal_seconds_remaining, cal_mask.sum(), fit_df["game_id"].nunique(),
            )
        else:
            fit_df    = dev_df.copy()
            fit_preds = oof_preds

        if game_level_cal:
            fit_df = fit_df.copy()
            fit_df["_pred"] = fit_preds
            game_agg = (
                fit_df.groupby("game_id")
                .agg(_mean_pred=("_pred", "mean"), _label=(TARGET_COL, "first"))
                .reset_index()
            )
            cal_preds  = game_agg["_mean_pred"].values
            cal_labels = game_agg["_label"].astype(int).values
            cal_games  = len(game_agg)
            logger.info(
                "game-level aggregation — %d independent (mean_pred, label) points",
                cal_games,
            )
        else:
            cal_preds  = fit_preds
            cal_labels = fit_df[TARGET_COL].astype(int).values
            cal_games  = fit_df["game_id"].nunique()

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(cal_preds, cal_labels)
        calibrators = {-1: calibrator}  # single-key dict for uniform logging
        logger.info("isotonic fitted on %d points from %d games", len(cal_preds), cal_games)

    # ── Step 4: final XGBoost trained on all dev data ─────────────────────
    X_dev = dev_df[FEATURE_COLS].astype(float)
    y_dev = dev_df[TARGET_COL].astype(int)
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X_dev, y_dev, verbose=False)

    # ── Step 5: evaluate on held-out test set ─────────────────────────────
    y_prob_raw = final_model.predict_proba(X_test)[:, 1]

    if quarter_cal:
        test_quarters = _quarter(test_df["seconds_elapsed"].values, test_df["is_overtime"].values)
        y_prob_cal = y_prob_raw.copy()
        for q, cal in calibrators.items():
            mask = test_quarters == q
            if mask.sum() > 0:
                y_prob_cal[mask] = cal.predict(y_prob_raw[mask])
    else:
        y_prob_cal = calibrator.predict(y_prob_raw)

    brier_raw = brier_score_loss(y_test, y_prob_raw)
    brier_cal = brier_score_loss(y_test, y_prob_cal)
    auc_raw   = roc_auc_score(y_test, y_prob_raw)
    auc_cal   = roc_auc_score(y_test, y_prob_cal)
    brier_improvement = brier_raw - brier_cal

    logger.info(
        "OOF result — raw brier=%.4f  cal brier=%.4f  improvement=%.4f",
        brier_raw, brier_cal, brier_improvement,
    )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        cal_mode = (
            "isotonic-oof-per-quarter" if quarter_cal else
            "isotonic-oof-game-level"  if game_level_cal else
            "isotonic-oof"
        )
        mlflow.log_params({
            "calibration":            cal_mode,
            "cv_folds":               n_splits,
            "cal_games":              cal_games,
            "cal_seconds_remaining": (
                cal_seconds_remaining if cal_seconds_remaining is not None else "all"
            ),
            "game_level_cal":         game_level_cal,
            "quarter_cal":            quarter_cal,
            "test_games":             test_df["game_id"].nunique(),
            "feature_cols":           ",".join(FEATURE_COLS),
        })
        mlflow.log_metrics({
            "brier_score_raw":    brier_raw,
            "roc_auc_raw":        auc_raw,
            "log_loss_raw":       log_loss(y_test, y_prob_raw),
            "brier_score":        brier_cal,
            "roc_auc":            auc_cal,
            "log_loss":           log_loss(y_test, y_prob_cal),
            "brier_improvement":  brier_improvement,
        })

        diag_raw = reliability_diagram(y_test, y_prob_raw)
        diag_cal = reliability_diagram(y_test, y_prob_cal)
        logger.info("reliability (raw):\n%s",  diag_raw.to_string(index=False))
        logger.info("reliability (oof cal):\n%s", diag_cal.to_string(index=False))

        diag_raw.to_csv("/tmp/reliability_raw_oof.csv", index=False)
        diag_cal.to_csv("/tmp/reliability_cal_oof.csv", index=False)
        mlflow.log_artifact("/tmp/reliability_raw_oof.csv", artifact_path="calibration")
        mlflow.log_artifact("/tmp/reliability_cal_oof.csv", artifact_path="calibration")

        mlflow.xgboost.log_model(final_model, name="xgb_model")
        calibrator_path = "/tmp/isotonic_calibrator_oof.pkl"
        save_obj = calibrators if quarter_cal else calibrator
        with open(calibrator_path, "wb") as f:
            pickle.dump(save_obj, f)
        mlflow.log_artifact(calibrator_path, artifact_path="calibration")

        logger.info("run %s logged", run.info.run_id)

    return run
