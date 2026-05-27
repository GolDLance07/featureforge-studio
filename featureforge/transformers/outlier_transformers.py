"""Outlier handling transformers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def apply_outlier_strategy(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
    params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter rows containing outliers in selected numeric columns."""
    params = params or {}
    result = df.copy()
    valid_columns = [col for col in columns if col in result.columns]
    numeric_columns = [col for col in valid_columns if pd.api.types.is_numeric_dtype(result[col])]
    skipped = sorted(set(valid_columns) - set(numeric_columns))
    metadata: dict[str, Any] = {
        "method": method,
        "columns": numeric_columns,
        "skipped_columns": skipped,
        "rows_before": len(result),
        "thresholds": {},
        "warnings": [],
    }

    if skipped:
        metadata["warnings"].append(f"Skipped non-numeric columns: {', '.join(skipped)}")
    if not numeric_columns:
        metadata["rows_after"] = len(result)
        metadata["removed_count"] = 0
        metadata["warnings"].append("No numeric columns selected for outlier handling.")
        return result, metadata

    mask = pd.Series(True, index=result.index)
    if method == "iqr_filter":
        multiplier = float(params.get("threshold", 1.5))
        for col in numeric_columns:
            q1, q3 = result[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            metadata["thresholds"][col] = {"lower": float(lower), "upper": float(upper)}
            mask &= result[col].between(lower, upper) | result[col].isna()
    elif method == "zscore_filter":
        threshold = float(params.get("threshold", 3.0))
        for col in numeric_columns:
            std = result[col].std()
            if pd.isna(std) or std == 0:
                metadata["warnings"].append(f"{col} skipped: standard deviation is zero or missing.")
                continue
            z_score = (result[col] - result[col].mean()) / std
            metadata["thresholds"][col] = {"z_abs_max": threshold}
            mask &= (z_score.abs() <= threshold) | result[col].isna()
    else:
        raise ValueError(f"Unsupported outlier method: {method}")

    result = result[mask].copy()
    metadata["rows_after"] = len(result)
    metadata["removed_count"] = metadata["rows_before"] - metadata["rows_after"]
    return result, metadata
