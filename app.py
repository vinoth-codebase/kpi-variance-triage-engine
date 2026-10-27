"""
Automated KPI Variance & Root-Cause Triage Engine
Enterprise Streamlit Diagnostic Workbench (Zero-Emoji Consulting Aesthetic)
"""

from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    DB_PATH,
    EXPORTS_DIR,
    DIMENSIONS,
    DIMENSION_LABELS,
    SUPPORTED_KPIS,
    BASELINE_MODES,
    Z_SCORE_THRESHOLD,
    TOP_N_DRIVERS,
    COLOR_PALETTE,
)
from src.database import (
    fetch_daily_kpi_series,
    fetch_records_for_dates,
    fetch_filtered_fact_records,
    is_warehouse_populated,
)
from src.data_generator import generate_warehouse_data
from src.anomaly_detector import KPIAnomalyDetector
from src.attribution_engine import VarianceAttributionEngine
from src.exporter import generate_audit_pack_excel


st.set_page_config(
    page_title="Automated KPI Variance & Root-Cause Triage Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_institutional_css() -> None:
    """Apply minimalist consulting portal CSS styling and hide Streamlit chrome."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #0F172A;
            background-color: #F8FAFC;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}

        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2.5rem !important;
            padding-left: 2.2rem !important;
            padding-right: 2.2rem !important;
            max-width: 1600px;
        }

        /* Institutional Top Header */
        .portal-header {
            background-color: #0F172A;
            color: #FFFFFF;
            padding: 1.35rem 1.75rem;
            border: 1px solid #1E293B;
            border-radius: 4px;
            margin-bottom: 1.15rem;
        }
        .portal-eyebrow {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            color: #94A3B8;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .portal-title {
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #FFFFFF;
            margin: 0 0 0.3rem 0;
        }
        .portal-subtitle {
            font-size: 0.88rem;
            color: #CBD5E1;
            font-weight: 400;
            margin: 0;
        }

        /* Executive Diagnostic Narrative Banner */
        .exec-banner {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #0F172A;
            border-radius: 4px;
            padding: 1.15rem 1.4rem;
            margin-top: 0.5rem;
            margin-bottom: 1.15rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        }
        .exec-banner-critical {
            border-left: 4px solid #DC2626;
        }
        .exec-banner-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.45rem;
        }
        .exec-banner-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: #475569;
        }
        .exec-banner-body {
            font-size: 0.94rem;
            line-height: 1.55;
            color: #0F172A;
            font-weight: 400;
            margin-bottom: 0.65rem;
        }
        .exec-meta-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem;
            padding-top: 0.55rem;
            border-top: 1px solid #F1F5F9;
            font-size: 0.78rem;
            color: #475569;
        }
        .exec-meta-item strong {
            color: #0F172A;
            font-weight: 600;
        }

        /* 4 Core KPI Cards */
        .kpi-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 4px;
            padding: 1.05rem 1.2rem;
            height: 138px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
        }
        .kpi-card-label {
            font-size: 0.73rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #475569;
        }
        .kpi-card-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.75rem;
            font-weight: 700;
            color: #0F172A;
            letter-spacing: -0.03em;
            margin: 0.2rem 0;
        }
        .kpi-card-value-crimson {
            color: #DC2626;
        }
        .kpi-card-value-teal {
            color: #0D9488;
        }
        .kpi-card-sub {
            font-size: 0.76rem;
            color: #64748B;
            font-weight: 500;
        }

        /* Status Badges */
        .status-badge {
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            padding: 0.25rem 0.6rem;
            border-radius: 3px;
            text-transform: uppercase;
        }
        .badge-critical {
            background-color: #FEE2E2;
            color: #991B1B;
            border: 1px solid #FCA5A5;
        }
        .badge-moderate {
            background-color: #FEF3C7;
            color: #92400E;
            border: 1px solid #FCD34D;
        }
        .badge-normal {
            background-color: #CCFBF1;
            color: #115E59;
            border: 1px solid #5EEAD4;
        }

        /* Section Containers & Headers */
        .section-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 4px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
        }
        .section-title {
            font-size: 0.92rem;
            font-weight: 700;
            color: #0F172A;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.2rem;
        }
        .section-subtitle {
            font-size: 0.8rem;
            color: #64748B;
            margin-bottom: 0.75rem;
        }

        /* Streamlit Tabs Customization */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            border-bottom: 1px solid #E2E8F0;
            background-color: #FFFFFF;
            padding: 0.35rem 0.75rem 0 0.75rem;
            border: 1px solid #E2E8F0;
            border-radius: 4px 4px 0 0;
        }
        .stTabs [data-baseweb="tab"] {
            height: 2.65rem;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            color: #475569;
            padding: 0 1.1rem;
        }
        .stTabs [aria-selected="true"] {
            color: #0F172A !important;
            border-bottom-color: #0F172A !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_daily_warehouse_series() -> pd.DataFrame:
    """Ensure the SQLite warehouse is initialized and load the 60-day aggregated time series."""
    if not is_warehouse_populated(DB_PATH):
        generate_warehouse_data(DB_PATH, force_rebuild=True)
    return fetch_daily_kpi_series(DB_PATH)


@st.cache_data(show_spinner=False)
def load_window_fact_records(dates_tuple: tuple) -> pd.DataFrame:
    """Load granular execution records for a tuple of dates with caching."""
    return fetch_records_for_dates(list(dates_tuple), DB_PATH)


def build_surveillance_chart(
    surveillance_df: pd.DataFrame,
    kpi_key: str,
    selected_date: str,
    z_threshold: float,
) -> go.Figure:
    """Construct the Tab 1 60-day KPI Surveillance & Bollinger Control Band chart."""
    kpi_meta = SUPPORTED_KPIS[kpi_key]
    unit = str(kpi_meta["value_suffix"])
    label = str(kpi_meta["label"])

    fig = go.Figure()

    # Upper and lower Bollinger control band shading
    fig.add_trace(
        go.Scatter(
            x=surveillance_df["execution_date"],
            y=surveillance_df["upper_band"],
            mode="lines",
            line=dict(width=1, color="rgba(148, 163, 184, 0.45)", dash="dot"),
            name=f"Upper Control Band (+{z_threshold:.1f} SD)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=surveillance_df["execution_date"],
            y=surveillance_df["lower_band"],
            mode="lines",
            line=dict(width=1, color="rgba(148, 163, 184, 0.45)", dash="dot"),
