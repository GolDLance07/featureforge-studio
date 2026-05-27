"""Categorical encoding transformers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def apply_encoding(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
    params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Encode categorical columns and preserve generated feature names."""
    params = params or {}
    result = df.copy()
    valid_columns = [col for col in columns if col in result.columns]
    metadata: dict[str, Any] = {
        "method": method,
        "columns": valid_columns,
        "generated_columns": [],
        "warnings": [],
    }
    if not valid_columns:
        metadata["warnings"].append("No selected columns exist in the dataframe.")
        return result, metadata

    values = result[valid_columns].astype("string").fillna("__missing__")

    if method == "onehot":
        encoder = OneHotEncoder(
            handle_unknown=params.get("handle_unknown", "ignore"),
            sparse_output=False,
        )
        encoded = encoder.fit_transform(values)
        names = encoder.get_feature_names_out(valid_columns).tolist()
        encoded_df = pd.DataFrame(encoded, columns=names, index=result.index)
        result = pd.concat([result.drop(columns=valid_columns), encoded_df], axis=1)
        metadata["encoder"] = encoder
        metadata["generated_columns"] = names
        return result, metadata

    if method == "ordinal":
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        result[valid_columns] = encoder.fit_transform(values)
        metadata["encoder"] = encoder
        metadata["generated_columns"] = valid_columns
        return result, metadata

    if method == "label":
        mappings: dict[str, dict[str, int]] = {}
        for col in valid_columns:
            categorical = pd.Categorical(values[col])
            result[col] = categorical.codes
            mappings[col] = {str(category): int(code) for code, category in enumerate(categorical.categories)}
        metadata["mappings"] = mappings
        metadata["generated_columns"] = valid_columns
        return result, metadata

    raise ValueError(f"Unsupported encoding method: {method}")
