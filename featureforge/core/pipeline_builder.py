"""Build sklearn preprocessing pipelines from FeatureForge config."""

from __future__ import annotations

from typing import Any

from sklearn.pipeline import Pipeline

try:
    from featureforge.core.column_transformer_builder import ColumnTransformerBuilder
except ModuleNotFoundError:
    from core.column_transformer_builder import ColumnTransformerBuilder


class PipelineBuilder:
    """Translate transformation_config JSON into sklearn Pipeline objects."""

    def from_config(self, transformation_config: dict[str, Any], column_groups: dict[str, list[str]]):
        """Build a sklearn Pipeline containing a ColumnTransformer preprocessor."""
        preprocessor = ColumnTransformerBuilder().build(transformation_config, column_groups)
        return Pipeline([("preprocessor", preprocessor)])
