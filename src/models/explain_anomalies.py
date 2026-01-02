from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.ingestion.load_data import load_raw_data, basic_sanity_checks
from src.features.preprocess import PreprocessConfig, split_X_y
from src.models.explain import top_abs_contributors


ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = Path("models")


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    Loads anomaly score report, then for top anomalous samples,
    reports which features deviate most from the population baseline
    using robust z-scores (median/MAD).
    """
    ensure_dirs()

    df = load_raw_data("uci-secom.csv")
    basic_sanity_checks(df)

    if "Pass/Fail" in df.columns:
        df["yield"] = (df["Pass/Fail"] == 1).astype(int)

    cfg = PreprocessConfig(target_col="yield", drop_cols=["Pass/Fail", "Time"])
    X, _ = split_X_y(df, cfg)

    report_path = ARTIFACTS_DIR / "anomaly_scores.csv"
    report = pd.read_csv(report_path)

    # choose top N anomalous indices
    top_n = 5
    top_idx = report.head(top_n).index.tolist()

    # robust baseline stats
    X_num = X.select_dtypes(include=[np.number])
    median = X_num.median(axis=0)
    mad = (X_num.sub(median).abs()).median(axis=0).replace(0, 1e-9)

    feature_names = X_num.columns.tolist()

    rows = []
    for i in top_idx:
        x = X_num.iloc[i]
        robust_z = (x - median) / mad  # robust deviation

        contrib = top_abs_contributors(feature_names, robust_z.values, top_k=10)
        contrib["row_index"] = i
        rows.append(contrib)

    out = pd.concat(rows, ignore_index=True)
    out_path = ARTIFACTS_DIR / "top_anomaly_contributors.csv"
    out.to_csv(out_path, index=False)

    print("Saved:")
    print(f"- {out_path}")
    print("\nTop contributors for top anomalous samples saved to CSV.")


if __name__ == "__main__":
    main()