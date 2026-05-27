from __future__ import annotations

import streamlit as st

from core.profiling_engine import ProfilingEngine
from core.transformation_manager import TransformationManager
from utils.frontend import (
    get_active_dataframe,
    inject_global_styles,
    initialize_session_state,
)


st.set_page_config(page_title="FeatureForge | Transform", page_icon="FF", layout="wide")
initialize_session_state()
inject_global_styles()

st.title("Feature Transformations")
st.caption("Queue preprocessing choices and preview the transformed dataset.")

df = get_active_dataframe()
if df is None:
    st.info("Upload a dataset first from the Upload & Summary page.")
    st.stop()

column_types = ProfilingEngine.detect_column_types(df)
numeric_cols = column_types["numeric"]
categorical_cols = column_types["categorical"]
manager = TransformationManager()


def add_step(step_type: str, method: str, columns: list[str], params: dict) -> None:
    try:
        config, step = manager.add_step(
            st.session_state["transformation_config"],
            {"type": step_type, "method": method, "columns": columns, "params": params},
            st.session_state["transformation_history"],
        )
    except ValueError as exc:
        st.warning(str(exc))
        return
    st.session_state["transformation_config"] = config
    st.session_state["transformation_history"].append(step)
    st.success(f"{step['label']} added to the queue.")

tabs = st.tabs(["Missing Values", "Scaling", "Encoding", "Outliers"])

with tabs[0]:
    st.markdown("#### Missing Value Handling")
    mv_cols = st.multiselect("Columns", df.columns.tolist(), default=[col for col in df.columns if df[col].isna().any()][:3])
    mv_method = st.selectbox("Method", ["mean", "median", "mode", "constant", "drop_rows"])
    constant_value = st.text_input("Constant value", value="Unknown", disabled=mv_method != "constant")
    if st.button("Add missing-value step", type="primary"):
        params = {"constant_value": constant_value} if mv_method == "constant" else {}
        add_step("missing_values", mv_method, mv_cols, params)

with tabs[1]:
    st.markdown("#### Feature Scaling")
    scale_cols = st.multiselect("Numeric columns", numeric_cols, key="scale_cols")
    scale_method = st.selectbox("Scaler", ["standard", "minmax", "robust"])
    if st.button("Add scaling step", type="primary"):
        add_step("scaling", scale_method, scale_cols, {})

with tabs[2]:
    st.markdown("#### Encoding")
    enc_cols = st.multiselect("Categorical columns", categorical_cols, key="enc_cols")
    enc_method = st.selectbox("Encoder", ["onehot", "ordinal", "label"])
    handle_unknown = st.selectbox("Unknown categories", ["ignore", "error"], disabled=enc_method == "label")
    if st.button("Add encoding step", type="primary"):
        add_step("encoding", enc_method, enc_cols, {"handle_unknown": handle_unknown})

with tabs[3]:
    st.markdown("#### Outlier Handling")
    outlier_cols = st.multiselect("Numeric columns", numeric_cols, key="outlier_cols")
    outlier_method = st.selectbox("Method", ["iqr_filter", "zscore_filter"])
    threshold = st.slider("Threshold", 1.0, 5.0, 3.0, 0.1)
    if st.button("Add outlier step", type="primary"):
        add_step("outliers", outlier_method, outlier_cols, {"threshold": threshold})

left, right = st.columns([0.95, 1.05])

with left:
    st.markdown("### Transformation Queue")
    history = st.session_state["transformation_history"]
    if not history:
        st.info("No transformations queued yet.")
    else:
        for index, step in enumerate(history, start=1):
            with st.container(border=True):
                st.markdown(f"**{index}. {step['label']}**")
                st.caption(f"Columns: {', '.join(step['columns']) if step['columns'] else 'None selected'}")
                st.json(step, expanded=False)

    c1, c2 = st.columns(2)
    if c1.button("Preview queued steps"):
        preview_df, metadata = manager.apply_steps(df, history)
        st.session_state["preview_df"] = preview_df
        st.session_state["transformation_metadata"] = metadata
    if c2.button("Clear queue"):
        st.session_state["transformation_history"] = []
        st.session_state["transformation_config"] = {
            "missing_values": [],
            "scaling": [],
            "encoding": [],
            "outliers": [],
        }
        st.session_state["preview_df"] = None
        st.session_state["transformation_metadata"] = {}
        st.rerun()

with right:
    st.markdown("### Preview")
    preview_df = st.session_state.get("preview_df")
    if preview_df is None:
        st.dataframe(df.head(12), use_container_width=True, height=380)
    else:
        st.dataframe(preview_df.head(12), use_container_width=True, height=380)
        if st.button("Apply preview as processed dataset", type="primary"):
            st.session_state["processed_df"] = preview_df.copy()
            st.success("Preview stored as processed dataset for export.")

metadata = st.session_state.get("transformation_metadata")
if metadata:
    st.markdown("### Transformation Metadata")
    st.json(metadata, expanded=False)

st.markdown("### Current Transformation Config")
st.json(st.session_state["transformation_config"], expanded=True)
