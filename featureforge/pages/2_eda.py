from __future__ import annotations

import streamlit as st

from core.eda_engine import EDAEngine
from core.profiling_engine import ProfilingEngine
from utils.frontend import get_active_dataframe, inject_global_styles, initialize_session_state


st.set_page_config(page_title="FeatureForge | EDA", page_icon="FF", layout="wide")
initialize_session_state()
inject_global_styles()

st.title("Exploratory Data Analysis")
st.caption("Generate interactive visual checks before deciding transformations.")

df = get_active_dataframe()
if df is None:
    st.info("Upload a dataset first from the Upload & Summary page.")
    st.stop()

column_types = ProfilingEngine.detect_column_types(df)
numeric_cols = column_types["numeric"]
categorical_cols = column_types["categorical"]

top_controls = st.columns([0.8, 1, 1, 1])
chart_type = top_controls[0].selectbox(
    "Chart",
    ["Histogram", "Boxplot", "Countplot", "Scatter", "Correlation Heatmap", "Missing Values"],
)
sample_max = max(1, min(len(df), 10000))
sample_default = min(sample_max, 2000)
sample_step = 100 if sample_max >= 100 else 1
sample_size = top_controls[1].slider(
    "Sample size",
    min_value=1,
    max_value=sample_max,
    value=sample_default,
    step=sample_step,
)
theme = top_controls[2].selectbox("Visual style", ["Clean", "Presentation", "Compact"])
color_col = top_controls[3].selectbox("Color/group", ["None"] + categorical_cols)

plot_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df.copy()
color = None if color_col == "None" else color_col
engine = EDAEngine()

st.markdown("### Visualization")

if chart_type == "Histogram":
    if not numeric_cols:
        st.warning("Histogram needs at least one numeric column.")
    else:
        col = st.selectbox("Numeric column", numeric_cols)
        bins = st.slider("Bins", 10, 100, 35)
        fig = engine.build_figure(df, {"chart_type": chart_type, "x": col, "color": color, "bins": bins, "sample_size": sample_size})
        st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Boxplot":
    if not numeric_cols:
        st.warning("Boxplot needs at least one numeric column.")
    else:
        y_col = st.selectbox("Value column", numeric_cols)
        x_options = ["None"] + categorical_cols
        x_col = st.selectbox("Group by", x_options)
        fig = engine.build_figure(df, {"chart_type": chart_type, "x": x_col, "y": y_col, "color": color, "sample_size": sample_size})
        st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Countplot":
    if not categorical_cols:
        st.warning("Countplot needs at least one categorical column.")
    else:
        col = st.selectbox("Categorical column", categorical_cols)
        top_n = st.slider("Top categories", 5, 30, 15)
        fig = engine.build_figure(df, {"chart_type": chart_type, "x": col, "top_n": top_n, "sample_size": sample_size})
        st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Scatter":
    if len(numeric_cols) < 2:
        st.warning("Scatter plot needs at least two numeric columns.")
    else:
        x_col = st.selectbox("X axis", numeric_cols)
        y_col = st.selectbox("Y axis", [col for col in numeric_cols if col != x_col] or numeric_cols)
        fig = engine.build_figure(df, {"chart_type": chart_type, "x": x_col, "y": y_col, "color": color, "sample_size": sample_size})
        st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Correlation Heatmap":
    if len(numeric_cols) < 2:
        st.warning("Correlation heatmap needs at least two numeric columns.")
    else:
        selected = st.multiselect("Numeric columns", numeric_cols, default=numeric_cols[: min(8, len(numeric_cols))])
        if selected:
            fig = engine.build_figure(df, {"chart_type": chart_type, "columns": selected, "sample_size": sample_size})
            st.plotly_chart(fig, use_container_width=True)

else:
    if df.isna().sum().sum() == 0:
        st.success("No missing values detected.")
    else:
        fig = engine.build_figure(df, {"chart_type": chart_type, "sample_size": sample_size})
        st.plotly_chart(fig, use_container_width=True)

st.markdown("### Profiling Report")
st.info(
    "ydata-profiling will plug in here as a generated dataset report. "
    "Use the button below when you want a deeper HTML profile."
)

if st.button("Generate profiling HTML"):
    with st.spinner("Generating profile..."):
        try:
            html = ProfilingEngine().generate_ydata_report(df)
            st.components.v1.html(html, height=700, scrolling=True)
        except RuntimeError as exc:
            st.error(str(exc))
