from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from src.features.preprocess import PreprocessConfig, split_X_y
from src.models.explain import top_abs_contributors

DATA_DEFAULT = Path("data/raw/uci-secom.csv")
MODELS_DIR = Path("models")


def load_secom_df(df: pd.DataFrame) -> pd.DataFrame:
    # Map SECOM label if present (1=pass, -1=fail) -> yield 1/0
    if "Pass/Fail" in df.columns and "yield" not in df.columns:
        df = df.copy()
        df["yield"] = (df["Pass/Fail"] == 1).astype(int)
    return df


def get_X(df: pd.DataFrame) -> pd.DataFrame:
    cfg = PreprocessConfig(target_col="yield", drop_cols=["Pass/Fail", "Time"])
    X, _ = split_X_y(df, cfg)
    # keep numeric only for these models
    X = X.select_dtypes(include=[np.number])
    return X


@st.cache_data
def read_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def compute_anomaly_scores(iforest_pipeline, X: pd.DataFrame) -> np.ndarray:
    # Our saved model is a Pipeline: preprocess -> iforest
    X_t = iforest_pipeline.named_steps["preprocess"].transform(X)
    raw_scores = iforest_pipeline.named_steps["iforest"].score_samples(X_t)
    # Convert to "higher = more anomalous"
    return -raw_scores


def robust_deviation_explain(X_num: pd.DataFrame, row_idx: int, top_k: int = 10) -> pd.DataFrame:
    # Robust baseline stats (ignore NaNs)
    median = X_num.median(axis=0, skipna=True)
    mad = (X_num.sub(median).abs()).median(axis=0, skipna=True)

    # Avoid divide-by-zero and NaNs
    mad = mad.replace(0, 1e-9).fillna(1e-9)

    # For the selected row, fill missing sensor readings with baseline median
    x = X_num.iloc[row_idx].fillna(median)

    robust_z = (x - median) / mad
    robust_z = robust_z.replace([np.inf, -np.inf], 0).fillna(0)

    out = top_abs_contributors(X_num.columns.tolist(), robust_z.values, top_k=top_k)

    # Make the table nicer for Streamlit
    out["delta"] = out["delta"].astype(float).round(4)
    out["abs_delta"] = out["abs_delta"].astype(float).round(4)
    return out

def main():
    st.set_page_config(page_title="Semiconductor Yield & Anomaly Dashboard", layout="wide")
    st.title("AI-Driven Semiconductor Yield & Anomaly Detection")
    st.caption("SECOM-style manufacturing sensor data → yield prediction + anomaly detection + root-cause signals")

    # Sidebar: data source
    st.sidebar.header("Data Source")
    uploaded = st.sidebar.file_uploader("Upload a CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        source_name = "Uploaded CSV"
    else:
        if not DATA_DEFAULT.exists():
            st.error(
                "No CSV uploaded and default file not found.\n\n"
                "Place your dataset at `data/raw/uci-secom.csv` or upload a CSV in the sidebar."
            )
            st.stop()
        df = read_csv_cached(str(DATA_DEFAULT))
        source_name = str(DATA_DEFAULT)

    df = load_secom_df(df)

    st.sidebar.header("Models")
    yield_model_path = MODELS_DIR / "baseline_logreg.joblib"
    anomaly_model_path = MODELS_DIR / "isolation_forest.joblib"

    have_yield_model = yield_model_path.exists()
    have_anomaly_model = anomaly_model_path.exists()

    st.sidebar.write("Yield model:", "✅" if have_yield_model else "❌")
    st.sidebar.write("Anomaly model:", "✅" if have_anomaly_model else "❌")

    if not have_yield_model:
        st.warning("Yield model not found. Run: `python -m src.models.train_baseline`")
    if not have_anomaly_model:
        st.warning("Anomaly model not found. Run: `python -m src.models.train_anomaly_iforest`")

    # Prepare features
    try:
        X = get_X(df)
    except Exception as e:
        st.error(f"Failed to build feature matrix X. Error: {e}")
        st.stop()

    # Main layout
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.subheader("Dataset Overview")
        st.write(f"Source: `{source_name}`")
        st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
        if "yield" in df.columns:
            dist = df["yield"].value_counts(dropna=False).to_dict()
            st.write("Yield distribution (1=pass, 0=fail):", dist)

        st.dataframe(df.head(20), use_container_width=True, height=320)

    # Compute predictions / scores (if models exist)
    preds_df = pd.DataFrame(index=df.index)

    if have_yield_model:
        yield_model = load_model(str(yield_model_path))
        try:
            if hasattr(yield_model.named_steps["clf"], "predict_proba"):
                proba = yield_model.predict_proba(X)[:, 1]
                preds_df["yield_proba_pass"] = proba
            preds_df["yield_pred"] = yield_model.predict(X)
        except Exception as e:
            st.error(f"Yield prediction failed: {e}")

    if have_anomaly_model:
        anomaly_model = load_model(str(anomaly_model_path))
        try:
            scores = compute_anomaly_scores(anomaly_model, X)
            preds_df["anomaly_score"] = scores
        except Exception as e:
            st.error(f"Anomaly scoring failed: {e}")

    with col_b:
        st.subheader("Model Outputs")
        if preds_df.empty:
            st.info("Train models first to see predictions and anomaly scores.")
        else:
            # Show top anomalies
            if "anomaly_score" in preds_df.columns:
                st.markdown("**Top Anomalies** (higher score = more anomalous)")
                top_n = st.slider("Show top N", 5, 50, 10)
                top = preds_df.sort_values("anomaly_score", ascending=False).head(top_n).copy()
                top["row_index"] = top.index
                cols = [c for c in ["anomaly_score", "yield_proba_pass", "yield_pred"] if c in top.columns]
                st.dataframe(top[cols], use_container_width=True, height=320)
            else:
                st.info("Anomaly model not available yet.")

    # Drilldown section
    st.divider()
    st.subheader("Row Drilldown: Root-Cause Signals")

    default_idx = int(preds_df["anomaly_score"].idxmax()) if "anomaly_score" in preds_df.columns else 0
    row_idx = st.number_input("Row index", min_value=0, max_value=int(len(df) - 1), value=int(default_idx), step=1)

    drill_cols = st.columns([1, 1])

    with drill_cols[0]:
        st.markdown("**Selected Row (raw)**")
        st.dataframe(df.iloc[[row_idx]], use_container_width=True)

    with drill_cols[1]:
        st.markdown("**Selected Row (scores)**")
        row_out = preds_df.iloc[[row_idx]] if not preds_df.empty else pd.DataFrame()
        st.dataframe(row_out, use_container_width=True)

    # Root-cause: robust deviation from baseline across sensors
    st.markdown("**Top contributing sensors (robust deviation from baseline)**")
    try:
        contrib = robust_deviation_explain(X, row_idx=row_idx, top_k=10)

        contrib = contrib.rename(columns={"feature": "sensor_id"})

        st.dataframe(contrib, use_container_width=True)

        st.caption(
            "Feature IDs correspond to sensor channels / process parameters in the dataset."
        )
        
    except Exception as e:
        st.error(f"Root-cause computation failed: {e}")

    st.caption(
        "Interpretation: features with the largest absolute robust deviation are candidates for process shift or measurement anomaly."
    )

if __name__ == "__main__":
    main()
