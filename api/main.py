from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# --- Fix PYTHONPATH for src/ layout ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features.preprocess import PreprocessConfig, split_X_y
from src.models.explain import top_abs_contributors
from src.ops.bootstrap import ensure_models_exist
ensure_models_exist()


MODELS_DIR = Path("models")

app = FastAPI(
    title="Semiconductor Yield & Anomaly API",
    description="Inference API for yield prediction, anomaly detection, and root-cause signals",
    version="0.1.0",
)


# ---------- Request / Response Schemas ----------

class RowsRequest(BaseModel):
    rows: List[dict]


class YieldResponse(BaseModel):
    yield_pred: List[int]
    yield_proba_pass: List[float] | None = None


class AnomalyResponse(BaseModel):
    anomaly_score: List[float]


class ExplainResponse(BaseModel):
    sensor_id: List[str]
    delta: List[float]
    abs_delta: List[float]


# ---------- Utilities ----------

def load_models():
    try:
        yield_model = joblib.load(MODELS_DIR / "baseline_logreg.joblib")
    except Exception:
        yield_model = None

    try:
        anomaly_model = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    except Exception:
        anomaly_model = None

    return yield_model, anomaly_model


def build_X(df: pd.DataFrame) -> pd.DataFrame:
    if "Pass/Fail" in df.columns:
        df = df.copy()
        df["yield"] = (df["Pass/Fail"] == 1).astype(int)

    cfg = PreprocessConfig(target_col="yield", drop_cols=["Pass/Fail", "Time"])
    X, _ = split_X_y(df, cfg)
    return X.select_dtypes(include=[np.number])


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/yield", response_model=YieldResponse)
def predict_yield(req: RowsRequest):
    yield_model, _ = load_models()
    if yield_model is None:
        raise HTTPException(status_code=500, detail="Yield model not found")

    df = pd.DataFrame(req.rows)
    X = build_X(df)

    preds = yield_model.predict(X).tolist()
    out = {"yield_pred": preds}

    clf = yield_model.named_steps.get("clf")
    if clf is not None and hasattr(clf, "predict_proba"):
        out["yield_proba_pass"] = clf.predict_proba(X)[:, 1].tolist()

    return out


@app.post("/score/anomaly", response_model=AnomalyResponse)
def score_anomaly(req: RowsRequest):
    _, anomaly_model = load_models()
    if anomaly_model is None:
        raise HTTPException(status_code=500, detail="Anomaly model not found")

    df = pd.DataFrame(req.rows)
    X = build_X(df)

    X_t = anomaly_model.named_steps["preprocess"].transform(X)
    scores = -anomaly_model.named_steps["iforest"].score_samples(X_t)

    return {"anomaly_score": scores.tolist()}


@app.post("/explain/row", response_model=ExplainResponse)
def explain_row(req: RowsRequest):
    if len(req.rows) != 1:
        raise HTTPException(status_code=400, detail="Provide exactly one row")

    df = pd.DataFrame(req.rows)
    X = build_X(df)

    median = X.median(axis=0, skipna=True)
    mad = (X.sub(median).abs()).median(axis=0, skipna=True)
    mad = mad.replace(0, 1e-9).fillna(1e-9)

    x = X.iloc[0].fillna(median)
    robust_z = ((x - median) / mad).replace([np.inf, -np.inf], 0).fillna(0)

    contrib = top_abs_contributors(
        feature_names=X.columns.tolist(),
        deltas=robust_z.values,
        top_k=10,
    )

    contrib = contrib.rename(columns={"feature": "sensor_id"})
    return {
        "sensor_id": contrib["sensor_id"].astype(str).tolist(),
        "delta": contrib["delta"].astype(float).tolist(),
        "abs_delta": contrib["abs_delta"].astype(float).tolist(),
    }