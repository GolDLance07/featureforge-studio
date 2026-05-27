"""ColumnTransformer assembly helpers."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler


class ColumnTransformerBuilder:
    """Group compatible column operations into sklearn ColumnTransformer blocks."""

    def build(self, transformation_config: dict[str, Any], column_groups: dict[str, list[str]]):
        """Create a best-effort ColumnTransformer from the JSON config."""
        transformers = []
        numeric_steps = []
        categorical_steps = []

        missing = transformation_config.get("missing_values", [])
        for step in missing:
            strategy = self._imputer_strategy(step["method"])
            if strategy is None:
                continue
            cols = step["columns"]
            is_numeric = all(col in column_groups.get("numeric", []) for col in cols)
            fill_value = step.get("params", {}).get("constant_value")
            imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
            if is_numeric:
                numeric_steps.append(("imputer", imputer))
            else:
                categorical_steps.append(("imputer", imputer))

        for step in transformation_config.get("scaling", []):
            scaler = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}[step["method"]]()
            numeric_steps.append((f"{step['method']}_scaler", scaler))

        for step in transformation_config.get("encoding", []):
            if step["method"] == "onehot":
                categorical_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
            elif step["method"] == "ordinal":
                categorical_steps.append(("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))

        numeric_cols = self._columns_for_types(transformation_config, "scaling", column_groups.get("numeric", []))
        categorical_cols = self._columns_for_types(transformation_config, "encoding", column_groups.get("categorical", []))

        if numeric_steps and numeric_cols:
            transformers.append(("numeric", Pipeline(numeric_steps), numeric_cols))
        if categorical_steps and categorical_cols:
            transformers.append(("categorical", Pipeline(categorical_steps), categorical_cols))

        return ColumnTransformer(transformers, remainder="passthrough", verbose_feature_names_out=False)

    @staticmethod
    def _imputer_strategy(method: str) -> str | None:
        return {
            "mean": "mean",
            "median": "median",
            "mode": "most_frequent",
            "constant": "constant",
        }.get(method)

    @staticmethod
    def _columns_for_types(config: dict[str, Any], bucket: str, fallback: list[str]) -> list[str]:
        columns: list[str] = []
        for step in config.get(bucket, []):
            columns.extend(step.get("columns", []))
        return list(dict.fromkeys(columns)) or fallback
