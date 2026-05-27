"""Transformation queue and application orchestration."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

try:
    from featureforge.transformers.encoding_transformers import apply_encoding
    from featureforge.transformers.missing_value_transformers import apply_missing_value_strategy
    from featureforge.transformers.outlier_transformers import apply_outlier_strategy
    from featureforge.transformers.scaling_transformers import apply_scaling
except ModuleNotFoundError:
    from transformers.encoding_transformers import apply_encoding
    from transformers.missing_value_transformers import apply_missing_value_strategy
    from transformers.outlier_transformers import apply_outlier_strategy
    from transformers.scaling_transformers import apply_scaling


class TransformationManager:
    """Validate, store, preview, and apply user-selected preprocessing steps."""

    valid_methods = {
        "missing_values": {"mean", "median", "mode", "constant", "drop_rows"},
        "scaling": {"standard", "minmax", "robust"},
        "encoding": {"onehot", "ordinal", "label"},
        "outliers": {"iqr_filter", "zscore_filter"},
    }

    def add_step(
        self,
        config: dict[str, Any],
        step: dict[str, Any],
        existing_steps: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Validate a UI step and append it to the config bucket."""
        existing_steps = existing_steps or []
        step_type = step["type"]
        method = step["method"]
        columns = list(dict.fromkeys(step.get("columns", [])))
        params = step.get("params", {})

        if step_type not in self.valid_methods:
            raise ValueError(f"Unsupported transformation type: {step_type}")
        if method not in self.valid_methods[step_type]:
            raise ValueError(f"Unsupported {step_type} method: {method}")
        if not columns:
            raise ValueError("Select at least one column before adding a transformation.")

        normalized = {
            "id": f"{step_type}_{len(existing_steps) + 1}",
            "type": step_type,
            "method": method,
            "columns": columns,
            "params": params,
            "label": f"{step_type.replace('_', ' ').title()} - {method}",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        updated_config = self.empty_config() | {key: list(value) for key, value in config.items()}
        updated_config[step_type].append({"method": method, "columns": columns, "params": params})
        return updated_config, normalized

    def apply_steps(self, df: pd.DataFrame, steps: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply queued transformations in order."""
        result = df.copy()
        step_metadata: list[dict[str, Any]] = []

        for step in steps:
            step_type = step["type"]
            if step_type == "missing_values":
                result, metadata = apply_missing_value_strategy(result, step["columns"], step["method"], step.get("params", {}))
            elif step_type == "scaling":
                result, metadata = apply_scaling(result, step["columns"], step["method"], step.get("params", {}))
            elif step_type == "encoding":
                result, metadata = apply_encoding(result, step["columns"], step["method"], step.get("params", {}))
            elif step_type == "outliers":
                result, metadata = apply_outlier_strategy(result, step["columns"], step["method"], step.get("params", {}))
            else:
                raise ValueError(f"Unsupported transformation type: {step_type}")

            step_metadata.append({"step_id": step.get("id"), "label": step.get("label"), **metadata})

        return result, {
            "rows_before": len(df),
            "rows_after": len(result),
            "columns_before": df.columns.tolist(),
            "columns_after": result.columns.tolist(),
            "steps": step_metadata,
        }

    @staticmethod
    def empty_config() -> dict[str, list[dict[str, Any]]]:
        return {"missing_values": [], "scaling": [], "encoding": [], "outliers": []}
