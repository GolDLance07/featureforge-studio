"""Generate reproducible preprocessing code from transformation config."""

from __future__ import annotations

from io import StringIO
from typing import Any


class CodeGenerator:
    """Create standalone Python code users can export from FeatureForge."""

    def generate_pipeline_code(
        self,
        transformation_config: dict[str, Any],
        column_groups: dict[str, list[str]],
        dataset_path: str = "your_dataset.csv",
    ) -> str:
        """Generate a standalone sklearn preprocessing script."""
        numeric_cols = self._selected_columns(transformation_config, "scaling", column_groups.get("numeric", []))
        categorical_cols = self._selected_columns(transformation_config, "encoding", column_groups.get("categorical", []))
        buffer = StringIO()
        buffer.write("import pandas as pd\n")
        buffer.write("from sklearn.compose import ColumnTransformer\n")
        buffer.write("from sklearn.impute import SimpleImputer\n")
        buffer.write("from sklearn.pipeline import Pipeline\n")
        buffer.write("from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler\n\n")
        buffer.write(f"DATASET_PATH = {dataset_path!r}\n")
        buffer.write(f"NUMERIC_COLUMNS = {numeric_cols!r}\n")
        buffer.write(f"CATEGORICAL_COLUMNS = {categorical_cols!r}\n")
        buffer.write(f"TRANSFORMATION_CONFIG = {transformation_config!r}\n\n")
        buffer.write(self._builder_function(transformation_config))
        buffer.write("\n\nif __name__ == '__main__':\n")
        buffer.write("    df = pd.read_csv(DATASET_PATH)\n")
        buffer.write("    preprocessor = build_preprocessor()\n")
        buffer.write("    transformed = preprocessor.fit_transform(df)\n")
        buffer.write("    print(transformed.shape)\n")
        return buffer.getvalue()

    @staticmethod
    def _selected_columns(config: dict[str, Any], bucket: str, fallback: list[str]) -> list[str]:
        columns: list[str] = []
        for step in config.get(bucket, []):
            columns.extend(step.get("columns", []))
        return list(dict.fromkeys(columns)) or fallback

    @staticmethod
    def _builder_function(config: dict[str, Any]) -> str:
        scaler_map = {"standard": "StandardScaler()", "minmax": "MinMaxScaler()", "robust": "RobustScaler()"}
        encoder_map = {
            "onehot": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
            "ordinal": "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)",
            "label": "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)",
        }
        numeric_steps = []
        categorical_steps = []

        for step in config.get("missing_values", []):
            method = step["method"]
            if method == "drop_rows":
                continue
            strategy = {"mean": "mean", "median": "median", "mode": "most_frequent", "constant": "constant"}[method]
            if method == "constant":
                value = step.get("params", {}).get("constant_value", "Unknown")
                categorical_steps.append(f"('imputer', SimpleImputer(strategy='constant', fill_value={value!r}))")
            else:
                numeric_steps.append(f"('imputer', SimpleImputer(strategy='{strategy}'))")

        for step in config.get("scaling", []):
            numeric_steps.append(f"('{step['method']}_scaler', {scaler_map[step['method']]})")
        for step in config.get("encoding", []):
            categorical_steps.append(f"('{step['method']}_encoder', {encoder_map[step['method']]})")

        numeric_steps_text = ", ".join(numeric_steps) or "('identity', 'passthrough')"
        categorical_steps_text = ", ".join(categorical_steps) or "('identity', 'passthrough')"
        return (
            "def build_preprocessor():\n"
            f"    numeric_pipeline = Pipeline([{numeric_steps_text}])\n"
            f"    categorical_pipeline = Pipeline([{categorical_steps_text}])\n"
            "    return ColumnTransformer(\n"
            "        transformers=[\n"
            "            ('numeric', numeric_pipeline, NUMERIC_COLUMNS),\n"
            "            ('categorical', categorical_pipeline, CATEGORICAL_COLUMNS),\n"
            "        ],\n"
            "        remainder='passthrough',\n"
            "        verbose_feature_names_out=False,\n"
            "    )"
        )
