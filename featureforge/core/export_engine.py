"""Build and optionally persist export artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class ExportEngine:
    """Produce cleaned dataset, config, pipeline code, and report exports."""

    def build_exports(
        self,
        df: pd.DataFrame,
        transformation_config: dict[str, Any],
        history: list[dict[str, Any]],
        pipeline_code: str,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Build download payloads and optionally save them under exports/."""
        artifacts = {
            "cleaned_csv": df.to_csv(index=False).encode("utf-8"),
            "config_json": json.dumps(transformation_config, indent=2),
            "pipeline_code": pipeline_code,
            "report_markdown": self.generate_report(history, df),
        }
        if output_dir is not None:
            self.persist_exports(artifacts, Path(output_dir))
        return artifacts

    @staticmethod
    def generate_report(history: list[dict[str, Any]], df: pd.DataFrame) -> str:
        lines = [
            "# FeatureForge Transformation Report",
            "",
            f"- Rows: {df.shape[0]:,}",
            f"- Columns: {df.shape[1]:,}",
            f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "## Steps",
        ]
        if not history:
            lines.append("No transformations were applied.")
        else:
            for index, step in enumerate(history, start=1):
                columns = ", ".join(step.get("columns", []))
                lines.append(f"{index}. {step.get('label', step.get('type'))} on {columns}")
        return "\n".join(lines)

    @staticmethod
    def persist_exports(artifacts: dict[str, Any], output_dir: Path) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        targets = {
            "cleaned_csv": output_dir / "datasets" / f"cleaned_{timestamp}.csv",
            "config_json": output_dir / "configs" / f"config_{timestamp}.json",
            "pipeline_code": output_dir / "pipelines" / f"pipeline_{timestamp}.py",
            "report_markdown": output_dir / "reports" / f"report_{timestamp}.md",
        }
        for key, path in targets.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(artifacts[key], bytes):
                path.write_bytes(artifacts[key])
            else:
                path.write_text(artifacts[key], encoding="utf-8")
