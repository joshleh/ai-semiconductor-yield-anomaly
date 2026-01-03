from __future__ import annotations

from pathlib import Path

DATA_PATH = Path("data/raw/uci-secom.csv")
MODELS_DIR = Path("models")


def ensure_models_exist() -> None:
    """
    Ensures required model files exist. If missing, trains them.
    Intended for demo deployments where models aren't shipped.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    baseline = MODELS_DIR / "baseline_logreg.joblib"
    iforest = MODELS_DIR / "isolation_forest.joblib"

    if baseline.exists() and iforest.exists():
        return

    # Import here to avoid import cost at module import time
    from src.models.train_baseline import main as train_baseline
    from src.models.train_anomaly_iforest import main as train_iforest

    # Train missing pieces
    if not baseline.exists():
        train_baseline()
    if not iforest.exists():
        train_iforest()