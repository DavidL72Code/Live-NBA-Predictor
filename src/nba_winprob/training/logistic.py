"""Train the live logistic win-probability model.

The production XGBoost stack (``training.train``) fits 400 trees with no early
stopping, which overfits the live event table badly: on rolling chronological
game folds it scores Brier 0.1709 after isotonic calibration, versus 0.1533 for
the model built here, trained on the same games. See
``artifacts/live_logistic_basis_comparison.json``.

The model is a single scikit-learn pipeline —
``expand_live_basis -> StandardScaler -> LogisticRegression`` — so the served
artifact carries its own feature engineering and ``LogisticWinProbServer`` can
keep passing a plain ``FEATURE_COLS`` frame.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd

from nba_winprob.features.basis import expand_live_basis
from nba_winprob.training.train import TARGET_COL

logger = logging.getLogger(__name__)

DEFAULT_C = 1.0


def build_logistic_model(c_value: float = DEFAULT_C):
    """Return the unfitted serving pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    return make_pipeline(
        FunctionTransformer(expand_live_basis, validate=False),
        StandardScaler(),
        LogisticRegression(C=c_value, max_iter=3000),
    )


def train_logistic(
    df: pd.DataFrame,
    model_path: str | Path,
    c_value: float = DEFAULT_C,
):
    """Fit the pipeline on every row of ``df`` and pickle it to ``model_path``."""
    model = build_logistic_model(c_value)
    model.fit(df, df[TARGET_COL].astype(int))

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as artifact:
        pickle.dump(model, artifact)
    logger.info("logistic model trained on %d rows → %s", len(df), path)
    return model
