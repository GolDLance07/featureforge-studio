from __future__ import annotations

import streamlit as st

from utils.frontend import inject_global_styles, initialize_session_state


st.set_page_config(page_title="FeatureForge | Roadmap", page_icon="FF", layout="wide")
initialize_session_state()
inject_global_styles()

st.title("Future Modules")
st.caption("PRD-aligned modules for the post-MVP roadmap.")

modules = [
    ("Feature Construction", "Polynomial features, interaction terms, datetime decomposition, log transforms."),
    ("Feature Selection", "Correlation selection, variance thresholding, feature importance, RFE."),
    ("Feature Extraction", "PCA, dimensionality reduction, embedding-style features."),
    ("Smart Recommendations", "Actionable preprocessing suggestions based on profiling insights."),
]

for title, description in modules:
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.write(description)
        st.info("Planned for V2/V3 after the V1 preprocessing path is stable.")
