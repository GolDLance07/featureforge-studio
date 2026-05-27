# FeatureForge Studio

FeatureForge Studio is an interactive EDA and feature-engineering workspace for machine learning preprocessing. It gives you a Streamlit app for visual workflows and an importable Python library for scripts, notebooks, and reusable preprocessing pipelines.

> Install name: `featureforge-studio`  
> Import name: `featureforge`

## Highlights

- Upload CSV datasets and inspect schema, missing values, duplicates, memory usage, and column types.
- Generate smart data-quality warnings for missingness, skew, high cardinality, duplicate rows, and likely ID columns.
- Explore data with interactive Plotly charts: histograms, boxplots, count plots, scatter plots, correlation heatmaps, and missing-value views.
- Queue preprocessing steps for imputation, scaling, encoding, and outlier handling.
- Export cleaned datasets, transformation config JSON, preprocessing code, and transformation reports.
- Use the same backend logic from Python via `featureforge`.

## Installation

Install from a local wheel:

```bash
pip install dist/featureforge_studio-0.1.0-py3-none-any.whl
```

For local development:

```bash
git clone https://github.com/aarush07/featureforge.git
cd featureforge/featureforge
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

After publishing to PyPI:

```bash
pip install featureforge-studio
```

## Run The App

From an installed package:

```bash
featureforge-app
```

From the repository:

```bash
cd featureforge
streamlit run app.py
```

If the default Streamlit port is busy:

```bash
streamlit run app.py --server.port 8502
```

## Library Usage

```python
import pandas as pd
from featureforge import ProfilingEngine, TransformationManager

df = pd.DataFrame(
    {
        "age": [21, None, 35],
        "city": ["Pune", "Delhi", None],
        "salary": [40000, 52000, 61000],
    }
)

profile = ProfilingEngine().generate_summary(df)

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
        "id": "encoding_2",
        "type": "encoding",
        "method": "onehot",
        "columns": ["city"],
        "params": {"handle_unknown": "ignore"},
        "label": "Encoding - onehot",
    },
]

processed_df, metadata = TransformationManager().apply_steps(df, steps)
```

## Core API

- `DatasetManager`: CSV loading, column cleanup, basic validation.
- `ProfilingEngine`: dataset summaries, missing-value tables, type detection, smart warnings, optional ydata-profiling reports.
- `EDAEngine`: Plotly figure generation from declarative chart configs.
- `TransformationManager`: validates and applies transformation queues.
- `PipelineBuilder`: builds sklearn preprocessing pipelines from transformation config.
- `CodeGenerator`: exports standalone sklearn preprocessing code.
- `ExportEngine`: creates CSV, JSON, Python, and Markdown export payloads.

## Supported Transformations

| Area | Methods |
| --- | --- |
| Missing values | mean, median, mode, constant, drop rows |
| Scaling | StandardScaler, MinMaxScaler, RobustScaler |
| Encoding | OneHotEncoder, OrdinalEncoder, label-style category codes |
| Outliers | IQR filtering, Z-score filtering |

## Build A Distributable Package

```bash
python -m pip install build twine
python -m build
```

This creates:

```text
dist/featureforge_studio-0.1.0-py3-none-any.whl
dist/featureforge_studio-0.1.0.tar.gz
```

## Publish To PyPI

First test the package on TestPyPI:

```bash
twine upload --repository testpypi dist/*
```

Then publish to PyPI:

```bash
twine upload dist/*
```

Note: the PyPI name `featureforge` is already taken by another project, so this package uses the distribution name `featureforge-studio` while keeping the clean import name `featureforge`.

## Development

Run tests:

```bash
pytest -q
```

Rebuild after metadata changes:

```bash
rm -rf dist build *.egg-info
python -m build
```

## License

MIT License. See [LICENSE](LICENSE).
