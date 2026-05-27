"""Dataset profiling and warning generation for FeatureForge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class ProfilingEngine:
    """Generate reusable dataset summaries for UI, exports, and recommendations."""

    def generate_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return the profile data used by UI, exports, and recommendations."""
        column_types = self.detect_column_types(df)
        missing_values = self.missing_value_table(df)
        numeric_summary = self.numeric_summary(df)
        categorical_summary = self.categorical_summary(df)
        return {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "column_types": column_types,
            "missing_values": missing_values,
            "duplicates": int(df.duplicated().sum()),
            "memory_usage": self.memory_usage(df),
            "numeric_summary": numeric_summary,
            "categorical_summary": categorical_summary,
            "warnings": self.smart_warnings(df, column_types, missing_values),
        }

    def generate_ydata_report(
        self,
        df: pd.DataFrame,
        title: str = "FeatureForge Report",
        output_path: str | Path | None = None,
    ) -> str:
        """Generate a ydata-profiling HTML report when the optional package exists."""
        try:
            from ydata_profiling import ProfileReport
        except ImportError as exc:
            raise RuntimeError("Install ydata-profiling to generate profiling reports.") from exc

        report = ProfileReport(df, title=title, minimal=True)
        if output_path is not None:
            report.to_file(output_path)
            return str(output_path)
        return report.to_html()

    @staticmethod
    def detect_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        boolean = df.select_dtypes(include=["bool"]).columns.tolist()
        categorical = [
            col
            for col in df.columns
            if col not in numeric and col not in datetime_cols and col not in boolean
        ]
        return {"numeric": numeric, "categorical": categorical, "datetime": datetime_cols, "boolean": boolean}

    @staticmethod
    def missing_value_table(df: pd.DataFrame) -> pd.DataFrame:
        missing = df.isna().sum()
        table = pd.DataFrame(
            {
                "column": missing.index,
                "missing_count": missing.values,
                "missing_percent": (missing.values / max(len(df), 1) * 100).round(2),
            }
        )
        return table[table["missing_count"] > 0].sort_values("missing_percent", ascending=False)

    @staticmethod
    def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.empty:
            return pd.DataFrame()
        summary = numeric_df.describe().T
        summary["median"] = numeric_df.median(numeric_only=True)
        summary["skew"] = numeric_df.skew(numeric_only=True)
        return summary.round(3)

    @staticmethod
    def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for col in df.select_dtypes(exclude=["number", "datetime", "datetimetz", "bool"]).columns:
            mode = df[col].mode(dropna=True)
            rows.append(
                {
                    "column": col,
                    "unique": int(df[col].nunique(dropna=True)),
                    "top": None if mode.empty else mode.iloc[0],
                    "top_frequency": 0 if mode.empty else int((df[col] == mode.iloc[0]).sum()),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def memory_usage(df: pd.DataFrame) -> str:
        bytes_used = int(df.memory_usage(deep=True).sum())
        if bytes_used < 1024:
            return f"{bytes_used} B"
        if bytes_used < 1024**2:
            return f"{bytes_used / 1024:.2f} KB"
        return f"{bytes_used / 1024**2:.2f} MB"

    def smart_warnings(
        self,
        df: pd.DataFrame,
        column_types: dict[str, list[str]],
        missing_table: pd.DataFrame,
    ) -> list[dict[str, str]]:
        warnings: list[dict[str, str]] = []
        duplicates = int(df.duplicated().sum())
        if duplicates:
            warnings.append({"severity": "medium", "title": "Duplicate rows", "detail": f"{duplicates:,} duplicate rows detected."})

        for _, row in missing_table[missing_table["missing_percent"] >= 30].iterrows():
            warnings.append({"severity": "high", "title": "High missingness", "detail": f"{row['column']} is {row['missing_percent']}% missing."})

        for col in column_types["numeric"]:
            skew = df[col].skew()
            if pd.notna(skew) and abs(skew) >= 1:
                warnings.append({"severity": "low", "title": "Skewed numeric column", "detail": f"{col} has skewness {skew:.2f}."})

        for col in column_types["categorical"]:
            cardinality = df[col].nunique(dropna=True)
            if cardinality >= max(50, len(df) * 0.2):
                warnings.append({"severity": "medium", "title": "High cardinality", "detail": f"{col} has {cardinality:,} unique values."})

        for col in df.columns:
            lower = col.lower()
            if lower == "id" or lower.endswith("_id") or df[col].nunique(dropna=True) == len(df):
                warnings.append({"severity": "low", "title": "Potential ID column", "detail": f"{col} may not be useful as a model feature."})

        return warnings[:10]
