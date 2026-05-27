from __future__ import annotations

import streamlit as st

from core.code_generator import CodeGenerator
from core.export_engine import ExportEngine
from core.profiling_engine import ProfilingEngine
from utils.frontend import get_active_dataframe, inject_global_styles, initialize_session_state


st.set_page_config(page_title="FeatureForge | Exports", page_icon="FF", layout="wide")
initialize_session_state()
inject_global_styles()

st.title("Exports")
st.caption("Download reproducible artifacts from the current FeatureForge session.")

df = get_active_dataframe()
if df is None:
    st.info("Upload a dataset first from the Upload & Summary page.")
    st.stop()

config = st.session_state["transformation_config"]
history = st.session_state["transformation_history"]
column_groups = ProfilingEngine.detect_column_types(df)
pipeline_code = CodeGenerator().generate_pipeline_code(config, column_groups)
artifacts = ExportEngine().build_exports(df, config, history, pipeline_code)

st.markdown("### Export Center")
c1, c2, c3, c4 = st.columns(4)

c1.download_button(
    "Download cleaned CSV",
    data=artifacts["cleaned_csv"],
    file_name="featureforge_cleaned_dataset.csv",
    mime="text/csv",
    use_container_width=True,
)

c2.download_button(
    "Download config JSON",
    data=artifacts["config_json"],
    file_name="featureforge_transformation_config.json",
    mime="application/json",
    use_container_width=True,
)

c3.download_button(
    "Download pipeline code",
    data=artifacts["pipeline_code"],
    file_name="featureforge_preprocessing_pipeline.py",
    mime="text/x-python",
    use_container_width=True,
)

c4.download_button(
    "Download report",
    data=artifacts["report_markdown"],
    file_name="featureforge_transformation_report.md",
    mime="text/markdown",
    use_container_width=True,
)

left, right = st.columns([1, 1])
with left:
    st.markdown("### Pipeline Code Preview")
    st.code(artifacts["pipeline_code"], language="python")
with right:
    st.markdown("### Transformation Report")
    st.markdown(artifacts["report_markdown"])
