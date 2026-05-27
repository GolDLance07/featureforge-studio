from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


APP_NAME = "FeatureForge"
TRANSFORMATION_BUCKETS = ["missing_values", "scaling", "encoding", "outliers"]


def initialize_session_state() -> None:
    defaults: dict[str, Any] = {
        "raw_df": None,
        "processed_df": None,
        "preview_df": None,
        "active_filename": None,
        "transformation_config": {
            "missing_values": [],
            "scaling": [],
            "encoding": [],
            "outliers": [],
        },
        "transformation_history": [],
        "generated_pipeline_code": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ff-ink: #172026;
            --ff-muted: #5f6f73;
            --ff-line: #d7e0df;
            --ff-panel: #f7faf8;
            --ff-teal: #0e7c7b;
            --ff-coral: #c75f4b;
            --ff-gold: #b6872f;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }
        h1, h2, h3 {
            color: var(--ff-ink);
            letter-spacing: 0;
        }
        .brand-block {
            border-bottom: 1px solid var(--ff-line);
            padding-bottom: 1rem;
            margin-bottom: 1rem;
        }
        .brand-block h1 {
            font-size: 1.45rem;
            margin: 0;
        }
        .hero {
            border: 1px solid var(--ff-line);
            border-radius: 8px;
            background: linear-gradient(135deg, #f7faf8 0%, #edf5f2 48%, #fff7ed 100%);
            padding: 2rem;
            margin-bottom: 1.5rem;
        }
        .hero h1 {
            font-size: clamp(2rem, 3vw, 3.2rem);
            line-height: 1.05;
            max-width: 820px;
            margin: 0.25rem 0 0.75rem 0;
        }
        .hero-copy {
            color: var(--ff-muted);
            font-size: 1.08rem;
            max-width: 760px;
            margin: 0;
        }
        .eyebrow {
            color: var(--ff-teal);
            text-transform: uppercase;
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            margin: 0;
        }
        .info-panel {
            min-height: 172px;
            border: 1px solid var(--ff-line);
            border-radius: 8px;
            background: var(--ff-panel);
            padding: 1.1rem;
        }
        .info-panel p {
            color: var(--ff-muted);
            margin-bottom: 0;
        }
        div[data-testid="stMetric"] {
            border: 1px solid var(--ff-line);
            border-radius: 8px;
            padding: 0.85rem;
            background: #ffffff;
        }
        div[data-testid="stTabs"] button {
            font-weight: 650;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_active_dataframe() -> pd.DataFrame | None:
    processed_df = st.session_state.get("processed_df")
    if processed_df is not None:
        return processed_df
    return st.session_state.get("raw_df")
