from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.ingestion.load_data import load_raw_data, basic_sanity_checks
from src.features.preprocess import PreprocessConfig, build_preprocessor, split_X_y


ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = Path("models")


def infer_numeric_features(X: pd.DataFrame) -> list[str]:
    # Use numeric columns only for the baseline.
    # (Manufacturing sensor datasets often are all numeric.)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric feature columns found for baseline model")
    return numeric_cols


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    Train a baseline yield classifier and save metrics.
    Expects a CSV in data/raw/ with a target column named 'yield'.
    """
    ensure_dirs()

    df = load_raw_data("uci-secom.csv")
    basic_sanity_checks(df)

    # SECOM label: 1 = pass, -1 = fail  -> convert to 1/0
    if "Pass/Fail" in df.columns:
        df["yield"] = (df["Pass/Fail"] == 1).astype(int)

    cfg = PreprocessConfig(target_col="yield", drop_cols=["Time", "Pass/Fail"])
    X, y = split_X_y(df, cfg)

    # Basic train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() == 2 else None
    )

    numeric_features = infer_numeric_features(X_train)
    preprocessor = build_preprocessor(numeric_features)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    model = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_val)

    metrics = {"accuracy": float(accuracy_score(y_val, y_pred))}

    # AUC (only if binary + probabilities available)
    if y.nunique() == 2 and hasattr(model.named_steps["clf"], "predict_proba"):
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_val, y_proba))

    # Save metrics + report
    (ARTIFACTS_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACTS_DIR / "baseline_classification_report.txt").write_text(
        classification_report(y_val, y_pred)
    )

    # Save model (joblib)
    import joblib  # local import so repo can still install minimal deps later

    joblib.dump(model, MODELS_DIR / "baseline_logreg.joblib")

    print("Saved:")
    print(f"- {ARTIFACTS_DIR / 'baseline_metrics.json'}")
    print(f"- {ARTIFACTS_DIR / 'baseline_classification_report.txt'}")
    print(f"- {MODELS_DIR / 'baseline_logreg.joblib'}")
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()