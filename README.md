# AI-Driven Semiconductor Yield & Anomaly Detection

End-to-end machine learning system for predicting semiconductor manufacturing yield and detecting process anomalies using high-dimensional sensor data.

## Project Overview
This project builds a production-style ML pipeline to:
- Predict manufacturing yield
- Detect anomalous process behavior
- Surface root-cause signals from sensor data

## Planned Components
- Data ingestion & validation
- Feature engineering for manufacturing data
- Yield prediction models
- Unsupervised anomaly detection
- Model explainability
- API and dashboard for inference

## Tech Stack
Python, pandas, scikit-learn, FastAPI, Streamlit

## Status
Scaffolded — implementation in progress.

## Run Baseline Yield Model

Place your dataset CSV in `data/raw/manufacturing_data.csv` with a target column named `yield`, then run:

```bash
python -m src.models.train_baseline
```

## Run Anomaly Detection

Train an unsupervised Isolation Forest model to detect abnormal
manufacturing process behavior:

```bash
python -m src.models.train_anomaly_iforest
```

## Explainability / Root Cause Signals

Global yield drivers (permutation importance):
```bash
python -m src.models.explain_yield
```

## Dashboard (Streamlit)

Run the interactive dashboard:

```bash
streamlit run dashboards/app.py
```

## API (FastAPI)

Run the inference API:

```bash
uvicorn api.main:app --reload
```

## Run with Docker (API + Dashboard)

Build and start both services:

```bash
docker compose up --build
```
