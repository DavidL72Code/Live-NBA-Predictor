"""Load and serve the trained XGBoost + isotonic calibrator.

WinProbServer wraps the two-stage pipeline (raw XGBoost score → isotonic
calibration) and exposes a single ``predict(feature) -> float`` method.
Load from MLflow with ``from_mlflow(run_id)`` or directly from files with
``from_paths(model_path, calibrator_path)``.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from nba_winprob.schemas import FeatureVector
from nba_winprob.training.train import FEATURE_COLS


class _TrustedArtifactUnpickler(pickle.Unpickler):
    """Reject executable or unexpected globals in legacy calibrator artifacts."""

    _ALLOWED_GLOBALS = {
        ("sklearn.isotonic", "IsotonicRegression"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
    }

    def find_class(self, module: str, name: str):  # noqa: ANN001
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise ValueError(f"blocked untrusted pickle global: {module}.{name}")
        return super().find_class(module, name)


def _load_trusted_calibrator(path: str | Path):
    """Load only the known sklearn/numpy calibrator object shape."""
    with open(path, "rb") as artifact:
        return _TrustedArtifactUnpickler(artifact).load()


class WinProbServer:
    def __init__(self, model, calibrator) -> None:
        self._model = model
        self._calibrator = calibrator

    @classmethod
    def from_mlflow(cls, run_id: str, tracking_uri: str | None = None) -> WinProbServer:
        """Load model and calibrator from an MLflow run."""
        import mlflow
        import mlflow.xgboost

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            from nba_winprob.config import get_settings
            uri = get_settings().mlflow_tracking_uri
            if uri:
                mlflow.set_tracking_uri(uri)

        try:
            model = mlflow.xgboost.load_model(f"runs:/{run_id}/xgb_model")
        except Exception as mlflow_error:
            # MLflow 3 stores the run-to-logged-model mapping in its tracking
            # database. Deployments that ship the immutable ``mlruns/``
            # artifacts but not that database can still load the model by
            # matching the logged model's embedded run_id.
            local_paths = _find_local_artifacts(run_id)
            if local_paths is None:
                raise mlflow_error
            model_path, calibrator_path = local_paths
            return cls.from_paths(model_path, calibrator_path)

        client = mlflow.tracking.MlflowClient()
        # Support both basic-train and OOF-train calibrator filenames.
        for cal_name in ("isotonic_calibrator.pkl", "isotonic_calibrator_oof.pkl"):
            try:
                cal_local = client.download_artifacts(run_id, f"calibration/{cal_name}")
                break
            except Exception:
                continue
        else:
            raise FileNotFoundError(
                f"No calibrator artifact found in run {run_id} "
                "(expected calibration/isotonic_calibrator.pkl or _oof.pkl)"
            )
        calibrator = _load_trusted_calibrator(cal_local)

        return cls(model, calibrator)

    @classmethod
    def from_paths(cls, model_path: str | Path, calibrator_path: str | Path) -> WinProbServer:
        """Load model and calibrator from local file paths."""
        import xgboost as xgb

        model = xgb.XGBClassifier()
        model.load_model(str(model_path))

        calibrator = _load_trusted_calibrator(calibrator_path)

        return cls(model, calibrator)

    def predict(self, feature: FeatureVector) -> float:
        """Return calibrated home-team win probability in [0, 1]."""
        import pandas as pd

        row = {col: getattr(feature, col) for col in FEATURE_COLS}
        X = pd.DataFrame([row]).astype(float)
        raw_prob = self._model.predict_proba(X)[:, 1]
        return float(self._calibrator.predict(raw_prob)[0])

    def predict_batch(self, features: list[FeatureVector]) -> list[float]:
        """Batch-predict calibrated probabilities for a list of feature vectors."""
        import pandas as pd

        rows = [{col: getattr(f, col) for col in FEATURE_COLS} for f in features]
        X = pd.DataFrame(rows).astype(float)
        raw_probs = self._model.predict_proba(X)[:, 1]
        return list(map(float, self._calibrator.predict(raw_probs)))


def _find_local_artifacts(run_id: str) -> tuple[Path, Path] | None:
    """Find committed MLflow artifacts without requiring the MLflow DB."""
    roots = [Path.cwd()]
    source_root = Path(__file__).resolve().parents[3]
    if source_root not in roots:
        roots.append(source_root)

    for root in roots:
        mlruns = root / "mlruns"
        if not mlruns.is_dir():
            continue
        model_path: Path | None = None
        for descriptor in mlruns.rglob("MLmodel"):
            try:
                descriptor_text = descriptor.read_text(encoding="utf-8")
            except OSError:
                continue
            run_id_pattern = rf"^run_id:\s*['\"]?{re.escape(run_id)}['\"]?\s*$"
            if not re.search(run_id_pattern, descriptor_text, re.MULTILINE):
                continue
            candidate = descriptor.parent / "model.ubj"
            if candidate.is_file():
                model_path = candidate
                break
        if model_path is None:
            continue
        for name in ("isotonic_calibrator.pkl", "isotonic_calibrator_oof.pkl"):
            calibrator = mlruns / "1" / run_id / "artifacts" / "calibration" / name
            if calibrator.is_file():
                return model_path, calibrator
    return None
