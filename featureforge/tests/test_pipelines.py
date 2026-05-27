import pandas as pd

from core.pipeline_builder import PipelineBuilder
from core.profiling_engine import ProfilingEngine


def test_pipeline_builder_returns_fit_transformable_pipeline():
    df = pd.DataFrame({"age": [10.0, 20.0], "city": ["Pune", "Delhi"]})
    config = {
        "missing_values": [],
        "scaling": [{"method": "standard", "columns": ["age"], "params": {}}],
        "encoding": [{"method": "onehot", "columns": ["city"], "params": {}}],
        "outliers": [],
    }
    column_groups = ProfilingEngine.detect_column_types(df)

    pipeline = PipelineBuilder().from_config(config, column_groups)
    transformed = pipeline.fit_transform(df)

    assert transformed.shape[0] == 2
