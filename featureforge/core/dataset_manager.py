"""Dataset loading and validation boundary for FeatureForge."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import BinaryIO

import pandas as pd


@dataclass(frozen=True)
class DatasetLoadResult:
    dataframe: pd.DataFrame
    filename: str
    warnings: list[str]


class DatasetManager:
    """Own CSV parsing, validation, and cache-safe dataset metadata."""

    def load_csv(self, uploaded_file: BinaryIO) -> DatasetLoadResult:
        """Load a CSV-like uploaded file and normalize obvious schema issues."""
        filename = getattr(uploaded_file, "name", "uploaded_dataset.csv")
        warnings: list[str] = []

        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin-1")
            warnings.append("Loaded with latin-1 encoding after UTF-8 failed.")
        except pd.errors.EmptyDataError as exc:
            raise ValueError("The uploaded CSV is empty.") from exc
        except pd.errors.ParserError as exc:
            raise ValueError("The uploaded CSV could not be parsed.") from exc

        if df.empty:
            warnings.append("Dataset loaded, but it has no rows.")

        df.columns = self._deduplicate_columns([self._clean_column_name(col) for col in df.columns])
        for col in df.select_dtypes(include=["object"]).columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.9:
                df[col] = parsed

        return DatasetLoadResult(dataframe=df, filename=filename, warnings=warnings)

    @staticmethod
    def _deduplicate_columns(columns: list[str]) -> list[str]:
        seen: dict[str, int] = {}
        result: list[str] = []
        for col in columns:
            count = seen.get(col, 0)
            result.append(col if count == 0 else f"{col}_{count}")
            seen[col] = count + 1
        return result

    @staticmethod
    def _clean_column_name(column: object) -> str:
        name = str(column).strip() or "unnamed"
        return re.sub(r"\.(\d+)$", r"_\1", name)
