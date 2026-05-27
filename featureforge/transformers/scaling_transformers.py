"""Numeric scaling transformers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def apply_scaling(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
    params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scale numeric columns using sklearn scalers."""
    params = params or {}
    result = df.copy()
    valid_columns = [col for col in columns if col in result.columns]
    numeric_columns = [col for col in valid_columns if pd.api.types.is_numeric_dtype(result[col])]
    skipped = sorted(set(valid_columns) - set(numeric_columns))

    if method not in SCALERS:
        raise ValueError(f"Unsupported scaling method: {method}")

    metadata: dict[str, Any] = {
        "method": method,
        "columns": numeric_columns,
        "skipped_columns": skipped,
        "warnings": [],
    }
    if skipped:
        metadata["warnings"].append(f"Skipped non-numeric columns: {', '.join(skipped)}")
    if not numeric_columns:
        metadata["warnings"].append("No numeric columns selected for scaling.")
        return result, metadata

    scaler = SCALERS[method](**params)
    result[numeric_columns] = scaler.fit_transform(result[numeric_columns])
    metadata["scaler"] = scaler
    metadata["feature_names"] = numeric_columns
    return result, metadata
