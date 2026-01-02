from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd


def top_abs_contributors(
    feature_names: List[str],
    deltas: np.ndarray,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Given per-feature deltas (e.g., z-scores or reconstruction-like errors),
    return the top-k absolute contributors.
    """
    if deltas.ndim != 1:
        raise ValueError("deltas must be a 1D array for a single sample")

    abs_vals = np.abs(deltas)
    idx = np.argsort(abs_vals)[::-1][:top_k]

    return pd.DataFrame(
        {
            "feature": [feature_names[i] for i in idx],
            "delta": deltas[idx],
            "abs_delta": abs_vals[idx],
        }
    )