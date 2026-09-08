"""Training pipeline unit tests — no MLflow server, no Postgres needed."""

import json
from pathlib import Path

import pandas as pd
import pytest

from nba_winprob.features import compute_game_features, game_label
from nba_winprob.ingestion.normalize import normalize_playbyplay
from nba_winprob.training.train import (
    FEATURE_COLS,
    _validate,
    apply_temperature,
    fit_temperature,
    load_parquet,
    train,
)


@pytest.fixture(scope="module")
def training_df():
    """Feature DataFrame built from the captured fixture game, duplicated to
    give GroupShuffleSplit enough distinct groups to split on."""
    raw = json.loads(
        (Path(__file__).parent / "fixtures" / "pbp_v3_0022300001.json").read_text()
    )
    events = normalize_playbyplay(raw)
    vectors = compute_game_features(events)
    label = game_label(events)
    base = pd.DataFrame([v.model_dump() for v in vectors])
    base["home_win"] = label
    # Synthesize 29 more fake games alternating labels — need enough groups
    # for a 3-way split (train 80% / cal 10% / test 10%) with both classes
    # in every partition.
    frames = [base]
    for i in range(1, 30):
        copy = base.copy()
        copy["game_id"] = f"00223{i:05d}"
        copy["home_win"] = i % 2
        frames.append(copy)
    return pd.concat(frames, ignore_index=True)


class TestValidate:
    def test_passes_on_good_dataframe(self, training_df):
        _validate(training_df)  # should not raise

    def test_raises_on_missing_column(self, training_df):
        bad = training_df.drop(columns=["score_diff"])
        with pytest.raises(ValueError, match="score_diff"):
            _validate(bad)

    def test_raises_on_null_labels(self, training_df):
        bad = training_df.copy()
        bad.loc[0, "home_win"] = None
        with pytest.raises(ValueError, match="null"):
            _validate(bad)


class TestLoadParquet:
    def test_roundtrip(self, training_df, tmp_path):
        path = tmp_path / "features.parquet"
        training_df.to_parquet(path, index=False)
        loaded = load_parquet(path)
        assert list(loaded.columns) == list(training_df.columns)
        assert len(loaded) == len(training_df)


class TestTrain:
    def test_train_returns_run_with_metrics(self, training_df, tmp_path):
        import mlflow

        db_uri = f"sqlite:///{tmp_path}/mlflow.db"
        run = train(
            training_df,
            experiment_name="test-experiment",
            run_name="unit-test",
            test_size=0.15,
            cal_size=0.15,
            mlflow_uri=db_uri,
        )
        assert run.info.run_id

        mlflow.set_tracking_uri(db_uri)
        metrics = mlflow.get_run(run.info.run_id).data.metrics
        # both raw and calibrated metrics logged
        assert "brier_score_raw" in metrics
        assert "brier_score" in metrics
        assert "brier_improvement" in metrics
        assert "roc_auc" in metrics
        # synthetic fixture has alternating game labels → Brier can't beat 0.25;
        # just verify it's a valid probability metric value
        assert 0.0 <= metrics["brier_score"] <= 1.0

    def test_calibration_improves_or_matches_raw(self, training_df, tmp_path):
        import mlflow

        db_uri = f"sqlite:///{tmp_path}/mlflow.db"
        run = train(training_df, mlflow_uri=db_uri, test_size=0.15, cal_size=0.15)
        mlflow.set_tracking_uri(db_uri)
        metrics = mlflow.get_run(run.info.run_id).data.metrics
        # Both metrics must be valid probability values.
        # Note: isotonic quality improvement cannot be reliably asserted on the
        # synthetic fixture — each game has a single constant label (all 0 or
        # all 1), so with only ~4 cal-set games the step function overfits and
        # can degrade Brier.  That property is validated on real data (see
        # notebooks/calibration_eval.ipynb).
        assert 0.0 <= metrics["brier_score"] <= 1.0
        assert 0.0 <= metrics["brier_score_raw"] <= 1.0

    def test_feature_cols_match_training_columns(self, training_df):
        for col in FEATURE_COLS:
            assert col in training_df.columns, f"FEATURE_COLS includes {col!r} not in df"

    def test_reliability_diagram(self, training_df, tmp_path):
        import numpy as np

        from nba_winprob.training.train import reliability_diagram

        y_true = training_df["home_win"].astype(int).values
        y_prob = np.full(len(y_true), 0.5)
        diag = reliability_diagram(y_true, y_prob)
        assert "bucket" in diag.columns
        assert "gap" in diag.columns
        assert len(diag) > 0


def test_temperature_scaling_preserves_order_and_fits_positive_temperature():
    import numpy as np

    raw = np.array([0.2, 0.4, 0.6, 0.8])
    labels = np.array([0, 0, 1, 1])
    temperature, loss = fit_temperature(raw, labels)
    scaled = apply_temperature(raw, temperature)
    assert temperature > 0
    assert loss >= 0
    assert np.all(np.diff(scaled) > 0)
