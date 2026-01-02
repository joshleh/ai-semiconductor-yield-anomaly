from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import joblib

from src.ingestion.load_data import load_raw_data, basic_sanity_checks
from src.features.preprocess import PreprocessConfig, split_X_y


ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = Path("models")


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    Loads the trained baseline yield model and computes permutation importance
    on a validation split to surface global drivers of yield.
    """
    ensure_dirs()

    df = load_raw_data("uci-secom.csv")
    basic_sanity_checks(df)

    if "Pass/Fail" in df.columns:
        df["yield"] = (df["Pass/Fail"] == 1).astype(int)

    cfg = PreprocessConfig(target_col="yield", drop_cols=["Pass/Fail", "Time"])
    X, y = split_X_y(df, cfg)

    # Use a split so permutation importance is computed out-of-sample
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_path = MODELS_DIR / "baseline_logreg.joblib"
    model = joblib.load(model_path)

    # Permutation importance expects a fitted model with predict method
    result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
        scoring="roc_auc",
    )

    # Feature names: use the original numeric columns (pipeline drops non-numeric anyway)
    feature_names = X_val.columns.tolist()

    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    out_csv = ARTIFACTS_DIR / "yield_permutation_importance.csv"
    importances.to_csv(out_csv, index=False)

    summary = {
        "scoring": "roc_auc",
        "n_repeats": 5,
        "top_features": importances.head(10)["feature"].tolist(),
    }
    (ARTIFACTS_DIR / "yield_importance_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("Saved:")
    print(f"- {out_csv}")
    print(f"- {ARTIFACTS_DIR / 'yield_importance_summary.json'}")
    print("\nTop 10 features:")
    print(importances.head(10))


if __name__ == "__main__":
    main()