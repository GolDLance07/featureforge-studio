"""Missing-value transformers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def apply_missing_value_strategy(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
    params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply imputation or row removal to selected columns."""
    params = params or {}
    result = df.copy()
    valid_columns = [col for col in columns if col in result.columns]
    metadata: dict[str, Any] = {
        "method": method,
        "columns": valid_columns,
        "filled_values": {},
        "rows_before": len(result),
        "rows_after": len(result),
        "warnings": [],
    }

    if not valid_columns:
        metadata["warnings"].append("No selected columns exist in the dataframe.")
        return result, metadata

    if method == "drop_rows":
        result = result.dropna(subset=valid_columns)
        metadata["rows_after"] = len(result)
        metadata["removed_rows"] = metadata["rows_before"] - metadata["rows_after"]
        return result, metadata

    for col in valid_columns:
        missing_before = int(result[col].isna().sum())
        if missing_before == 0:
            continue

        fill_value: Any
        if method == "mean":
            if not pd.api.types.is_numeric_dtype(result[col]):
                metadata["warnings"].append(f"{col} skipped: mean requires a numeric column.")
                continue
            fill_value = result[col].mean()
        elif method == "median":
            if not pd.api.types.is_numeric_dtype(result[col]):
                metadata["warnings"].append(f"{col} skipped: median requires a numeric column.")
                continue
            fill_value = result[col].median()
        elif method == "mode":
            modes = result[col].mode(dropna=True)
            if modes.empty:
                metadata["warnings"].append(f"{col} skipped: no non-missing mode value.")
                continue
            fill_value = modes.iloc[0]
        elif method == "constant":
            fill_value = params.get("constant_value", "Unknown")
        else:
            raise ValueError(f"Unsupported missing-value method: {method}")

        result[col] = result[col].fillna(fill_value)
        metadata["filled_values"][col] = {
            "value": fill_value.item() if hasattr(fill_value, "item") else fill_value,
            "count": missing_before,
        }

    return result, metadata
