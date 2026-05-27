from __future__ import annotations

import streamlit as st

from utils.frontend import (
    APP_NAME,
    TRANSFORMATION_BUCKETS,
    get_active_dataframe,
    inject_global_styles,
    initialize_session_state,
)


st.set_page_config(
    page_title="FeatureForge",
    page_icon="FF",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_session_state()
inject_global_styles()

raw_df = st.session_state.get("raw_df")
processed_df = st.session_state.get("processed_df")
active_df = get_active_dataframe()
history = st.session_state.get("transformation_history", [])

with st.sidebar:
    st.markdown('<div class="brand-block">', unsafe_allow_html=True)
    st.markdown(f"<h1>{APP_NAME}</h1>", unsafe_allow_html=True)
    st.caption("Interactive Feature Engineering & EDA Studio")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Workflow")
    st.page_link("app.py", label="Home")
    st.page_link("pages/1_upload_summary.py", label="Upload & Summary")
    st.page_link("pages/2_eda.py", label="Exploratory Analysis")
    st.page_link("pages/3_transformations.py", label="Transformations")
    st.page_link("pages/4_exports.py", label="Exports")
    st.page_link("pages/5_future_modules.py", label="Future Modules")

    st.markdown("### Dataset Status")
    if raw_df is None:
        st.info("No dataset loaded yet.")
    else:
        st.success("Dataset loaded")
        st.metric("Rows", f"{len(raw_df):,}")
        st.metric("Columns", f"{raw_df.shape[1]:,}")
        if processed_df is not None:
            st.metric("Processed rows", f"{len(processed_df):,}")
        st.metric("Queued steps", len(history))

    st.markdown("### Backend Progress")
    completed = sum(1 for bucket in TRANSFORMATION_BUCKETS if st.session_state["transformation_config"].get(bucket))
    st.progress(completed / len(TRANSFORMATION_BUCKETS), text="Transformation modules wired")


st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown(
    """
    <p class="eyebrow">Portfolio MVP</p>
    <h1>Forge clean ML-ready features from raw datasets.</h1>
    <p class="hero-copy">
    Upload a CSV, inspect data quality, explore relationships, queue preprocessing
    steps, and export reproducible artifacts from one guided workspace.
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

if active_df is None:
    st.markdown("### Start with a dataset")
    st.write("Open **Upload & Summary** from the sidebar and upload a CSV to unlock the workflow.")
else:
    st.markdown("### Current workspace")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{active_df.shape[0]:,}")
    c2.metric("Columns", f"{active_df.shape[1]:,}")
    c3.metric("Queued transforms", len(history))
    c4.metric("Active dataframe", "Processed" if processed_df is not None else "Raw")

    st.markdown("### Quick Preview")
    st.dataframe(active_df.head(8), use_container_width=True)

st.markdown("### Build Path")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <div class="info-panel">
        <h3>1. Understand</h3>
        <p>Upload a dataset, review schema health, and identify missingness,
        duplicates, skew, cardinality, and likely ID columns.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
        <div class="info-panel">
        <h3>2. Explore</h3>
        <p>Use interactive Plotly views for distributions, relationships,
        category counts, and numeric correlations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """
        <div class="info-panel">
        <h3>3. Transform</h3>
        <p>Queue preprocessing decisions and export a cleaned dataset,
        JSON config, pipeline code, and report.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Library Modules")
st.write(
    "The Streamlit app now sits on top of reusable Python modules: "
    "`core.DatasetManager`, `core.ProfilingEngine`, `core.TransformationManager`, "
    "`core.CodeGenerator`, `core.ExportEngine`, and the transformer functions."
)
