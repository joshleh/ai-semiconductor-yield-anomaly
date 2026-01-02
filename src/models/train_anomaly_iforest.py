from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from src.ingestion.load_data import load_raw_data, basic_sanity_checks
from src.features.preprocess import PreprocessConfig, build_preprocessor, split_X_y


ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = Path("models")


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def infer_numeric_features(X: pd.DataFrame) -> list[str]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric feature columns found for anomaly detection")
    return numeric_cols


def main() -> None:
    """
    Train an Isolation Forest model for process anomaly detection
    and generate a ranked anomaly score report.
    """
    ensure_dirs()

    df = load_raw_data("uci-secom.csv")
    basic_sanity_checks(df)

    # Map SECOM label (if present) but DO NOT use it for training
    if "Pass/Fail" in df.columns:
        df["yield"] = (df["Pass/Fail"] == 1).astype(int)

    cfg = PreprocessConfig(target_col="yield", drop_cols=["Pass/Fail", "Time"])
    X, _ = split_X_y(df, cfg)

    numeric_features = infer_numeric_features(X)
    preprocessor = build_preprocessor(numeric_features)

    iforest = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("iforest", iforest),
        ]
    )

    model.fit(X)

    # IsolationForest: lower score = more anomalous
    anomaly_scores = -model.named_steps["iforest"].score_samples(
        model.named_steps["preprocess"].transform(X)
    )

    report = df.copy()
    report["anomaly_score"] = anomaly_scores
    report = report.sort_values("anomaly_score", ascending=False)

    report_path = ARTIFACTS_DIR / "anomaly_scores.csv"
    report.to_csv(report_path, index=False)

    import joblib
    model_path = MODELS_DIR / "isolation_forest.joblib"
    joblib.dump(model, model_path)

    print("Saved:")
    print(f"- {report_path}")
    print(f"- {model_path}")
    print("\nTop 5 anomalous samples:")
    print(report[["anomaly_score"]].head())


if __name__ == "__main__":
    main()