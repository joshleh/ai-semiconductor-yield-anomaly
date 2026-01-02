from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessConfig:
    target_col: str = "yield"
    drop_cols: Optional[List[str]] = None


def build_preprocessor(
    numeric_features: List[str],
) -> ColumnTransformer:
    """
    Build a preprocessing transformer for numeric sensor features:
    - median imputation
    - standard scaling
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipe, numeric_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def split_X_y(
    df: pd.DataFrame,
    cfg: PreprocessConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    drop_cols = cfg.drop_cols or []
    if cfg.target_col not in df.columns:
        raise KeyError(f"Target column '{cfg.target_col}' not found in dataframe")

    X = df.drop(columns=[cfg.target_col] + drop_cols, errors="ignore")
    y = df[cfg.target_col]
    return X, y