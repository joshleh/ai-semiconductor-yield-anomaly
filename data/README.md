# Data Directory

This directory contains manufacturing process and sensor datasets used for yield prediction and anomaly detection.

## Structure
- raw/        : Original, immutable source data (not committed)
- interim/    : Cleaned intermediate datasets
- processed/  : Feature-engineered datasets for modeling

## Notes
Raw data is excluded from version control. Small synthetic or sample datasets may be added for demonstration purposes.

## Dataset Source

This project uses a publicly available semiconductor manufacturing dataset
containing high-dimensional sensor measurements and a binary yield label
(pass/fail).

The dataset simulates real-world fabrication and test environments where:
- Sensor readings may be noisy or missing
- Yield loss events are rare
- Root-cause analysis is non-trivial

Raw data is stored locally and excluded from version control.

