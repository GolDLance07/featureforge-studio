from __future__ import annotations

import pandas as pd
import streamlit as st

from core.dataset_manager import DatasetManager
from core.profiling_engine import ProfilingEngine
from utils.frontend import (
    inject_global_styles,
    initialize_session_state,
)


st.set_page_config(page_title="FeatureForge | Upload", page_icon="FF", layout="wide")
initialize_session_state()
inject_global_styles()

st.title("Upload & Dataset Summary")
st.caption("Load a CSV and inspect the dataset before feature engineering.")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file is not None:
    try:
        result = DatasetManager().load_csv(uploaded_file)
    except Exception as exc:  # noqa: BLE001 - UI should show friendly parse errors.
        st.error(f"Could not load this CSV: {exc}")
        st.stop()

    st.session_state["raw_df"] = result.dataframe
    st.session_state["processed_df"] = None
    st.session_state["active_filename"] = result.filename
    st.session_state["dataset_warnings"] = result.warnings

df = st.session_state.get("raw_df")

if df is None:
    st.info("Upload a CSV to generate the dataset profile.")
    st.stop()

profile = ProfilingEngine().generate_summary(df)
column_types = profile["column_types"]
missing_table = profile["missing_values"]
warnings = profile["warnings"]

st.markdown("### Dataset Health")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Duplicates", f"{profile['duplicates']:,}")
c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
c5.metric("Numeric", len(column_types["numeric"]))
c6.metric("Categorical", len(column_types["categorical"]))

st.markdown("### Preview")
preview_rows = st.slider("Preview rows", min_value=5, max_value=min(max(len(df), 5), 100), value=min(10, max(len(df), 5)))
st.dataframe(df.head(preview_rows), use_container_width=True, height=360)

st.markdown("### Smart Warnings")
if warnings:
    for item in warnings:
        st.warning(f"{item['title']}: {item['detail']}")
else:
    st.success("No major data quality warnings detected.")

for warning in st.session_state.get("dataset_warnings", []):
    st.info(warning)

left, right = st.columns([1.1, 0.9])
with left:
    st.markdown("### Column Types")
    type_rows = []
    for type_name, columns in column_types.items():
        type_rows.append({"type": type_name, "count": len(columns), "columns": ", ".join(columns[:12])})
    st.dataframe(pd.DataFrame(type_rows), use_container_width=True, hide_index=True)

    st.markdown("### Numeric Summary")
    numeric_summary = profile["numeric_summary"]
    if numeric_summary.empty:
        st.info("No numeric columns found.")
    else:
        st.dataframe(numeric_summary, use_container_width=True)

with right:
    st.markdown("### Missing Value Analysis")
    if missing_table.empty:
        st.success("No missing values detected.")
    else:
        st.dataframe(missing_table, use_container_width=True, hide_index=True)

    st.markdown("### Dataset Footprint")
    st.metric("Memory usage", profile["memory_usage"])
    st.metric("Active file", st.session_state.get("active_filename", "Uploaded CSV"))
