"""Reusable EDA chart generation for FeatureForge."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px


class EDAEngine:
    """Build Plotly figures from dataframe plus chart configuration."""

    def build_figure(self, df: pd.DataFrame, chart_config: dict[str, Any]):
        """Create a Plotly figure from a declarative chart config."""
        chart_type = chart_config["chart_type"]
        sample_size = min(int(chart_config.get("sample_size", len(df))), len(df))
        plot_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df.copy()
        color = chart_config.get("color")
        color = None if color in (None, "None") else color

        if chart_type == "Histogram":
            return px.histogram(
                plot_df,
                x=chart_config["x"],
                color=color,
                nbins=int(chart_config.get("bins", 35)),
                marginal="box",
            )
        if chart_type == "Boxplot":
            x_col = chart_config.get("x")
            return px.box(plot_df, x=None if x_col == "None" else x_col, y=chart_config["y"], color=color)
        if chart_type == "Countplot":
            col = chart_config["x"]
            counts = (
                plot_df[col]
                .astype("string")
                .fillna("Missing")
                .value_counts()
                .head(int(chart_config.get("top_n", 15)))
                .reset_index()
            )
            counts.columns = [col, "count"]
            return px.bar(counts, x=col, y="count", color=col)
        if chart_type == "Scatter":
            return px.scatter(plot_df, x=chart_config["x"], y=chart_config["y"], color=color)
        if chart_type == "Correlation Heatmap":
            columns = chart_config["columns"]
            corr = plot_df[columns].corr(numeric_only=True)
            return px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", aspect="auto")
        if chart_type == "Missing Values":
            missing = df.isna().sum().reset_index()
            missing.columns = ["column", "missing_count"]
            missing = missing[missing["missing_count"] > 0].sort_values("missing_count", ascending=False)
            return px.bar(missing, x="column", y="missing_count", color="missing_count")

        raise ValueError(f"Unsupported chart type: {chart_type}")
