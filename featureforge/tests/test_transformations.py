import pandas as pd

from core.transformation_manager import TransformationManager


def test_apply_steps_imputes_scales_and_encodes():
    df = pd.DataFrame(
        {
            "age": [10.0, None, 30.0],
            "city": ["Pune", None, "Delhi"],
        }
    )
    manager = TransformationManager()
    steps = [
        {
            "id": "missing_values_1",
            "type": "missing_values",
            "method": "median",
            "columns": ["age"],
            "params": {},
            "label": "Missing Values - median",
        },
        {
            "id": "missing_values_2",
            "type": "missing_values",
            "method": "mode",
            "columns": ["city"],
            "params": {},
            "label": "Missing Values - mode",
        },
        {
            "id": "encoding_3",
            "type": "encoding",
            "method": "onehot",
            "columns": ["city"],
            "params": {"handle_unknown": "ignore"},
            "label": "Encoding - onehot",
        },
    ]

    transformed, metadata = manager.apply_steps(df, steps)

    assert transformed["age"].isna().sum() == 0
    assert "city" not in transformed.columns
    assert metadata["rows_after"] == 3
