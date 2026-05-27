import pandas as pd

from core.code_generator import CodeGenerator
from core.export_engine import ExportEngine
from core.profiling_engine import ProfilingEngine


def test_export_engine_builds_all_artifacts():
    df = pd.DataFrame({"age": [10, 20], "city": ["Pune", "Delhi"]})
    config = {"missing_values": [], "scaling": [], "encoding": [], "outliers": []}
    code = CodeGenerator().generate_pipeline_code(config, ProfilingEngine.detect_column_types(df))

    artifacts = ExportEngine().build_exports(df, config, [], code)

    assert set(artifacts) == {"cleaned_csv", "config_json", "pipeline_code", "report_markdown"}
    assert b"age,city" in artifacts["cleaned_csv"]
    assert "build_preprocessor" in artifacts["pipeline_code"]
