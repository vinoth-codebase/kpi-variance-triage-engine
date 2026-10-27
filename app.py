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
            fill="tonexty",
            fillcolor="rgba(226, 232, 240, 0.42)",
            name=f"Control Band (+/- {z_threshold:.1f} SD)",
        )
    )

    # Prior 14-Day Rolling Baseline
    fig.add_trace(
        go.Scatter(
            x=surveillance_df["execution_date"],
            y=surveillance_df["prior_14d_mean"],
            mode="lines",
            line=dict(color=COLOR_PALETTE["slate"], width=1.8, dash="dash"),
            name="14-Day Rolling Baseline",
            hovertemplate="Date: %{x}<br>14D Baseline: %{y:.2f}" + unit + "<extra></extra>",
        )
    )

    # Daily KPI Value Trend
    fig.add_trace(
        go.Scatter(
            x=surveillance_df["execution_date"],
            y=surveillance_df["kpi_value"],
            mode="lines+markers",
            line=dict(color=COLOR_PALETTE["navy"], width=2.4),
            marker=dict(size=5, color=COLOR_PALETTE["navy"]),
            name=f"Daily {label}",
            customdata=np.stack(
                [
                    surveillance_df["day_index"],
                    surveillance_df["z_score"],
                    surveillance_df["status_badge"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>Date: %{x} (Day %{customdata[0]})</b><br>"
                f"{label}: %{{y:.2f}}{unit}<br>"
                "Z-Score: %{customdata[1]:+.2f}<br>"
                "Status: %{customdata[2]}<extra></extra>"
            ),
        )
    )

    # Highlighted Statistical Anomalies
    anomalies = surveillance_df[surveillance_df["is_anomaly"]]
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["execution_date"],
                y=anomalies["kpi_value"],
                mode="markers",
                marker=dict(
                    size=12,
                    color=COLOR_PALETTE["crimson"],
                    symbol="diamond",
                    line=dict(width=2, color="#FFFFFF"),
                ),
                name="Detected Statistical Anomaly",
                customdata=np.stack(
                    [
                        anomalies["day_index"],
                        anomalies["z_score"],
                        anomalies["p_value"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>ANOMALY DETECTED: %{x} (Day %{customdata[0]})</b><br>"
                    f"Observed {label}: %{{y:.2f}}{unit}<br>"
                    "Z-Score: %{customdata[1]:+.2f} (p = %{customdata[2]:.5f})<extra></extra>"
                ),
            )
        )

    # Selected Target Date Marker
    selected_row = surveillance_df[surveillance_df["execution_date"] == selected_date]
    if not selected_row.empty:
        fig.add_vline(
            x=selected_date,
            line_width=1.6,
            line_dash="solid",
            line_color=COLOR_PALETTE["crimson"],
        )

    fig.update_layout(
        title=dict(
            text=f"60-Day Operational Surveillance Horizon — {label} ({unit})",
            font=dict(size=14, color="#0F172A", family="Inter, sans-serif"),
            x=0.0,
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=410,
        margin=dict(l=45, r=25, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#475569"),
        ),
        xaxis=dict(
            title="Execution Date",
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            tickfont=dict(size=11, color="#475569"),
        ),
        yaxis=dict(
            title=f"{label} ({unit})",
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            tickfont=dict(size=11, color="#475569"),
        ),
        hovermode="x unified",
    )
    return fig


def build_waterfall_chart(
    df_slices: pd.DataFrame,
    baseline_val: float,
    target_val: float,
    kpi_key: str,
    top_n: int = 6,
    scope_title: str = "Top Contributing Drivers",
) -> go.Figure:
    """Construct the Tab 2 Plotly Bridge / Waterfall Chart from Baseline KPI to Target Date KPI."""
    kpi_meta = SUPPORTED_KPIS[kpi_key]
    unit = str(kpi_meta["value_suffix"])
    decimals = int(kpi_meta["decimals"])
    higher_is_better = bool(kpi_meta["higher_is_better"])

    sorted_df = df_slices.sort_values("variance_explained_pct", ascending=False).reset_index(
        drop=True
    )
    top_df = sorted_df.head(top_n)
    residual_contrib = float(target_val - baseline_val - top_df["total_contribution"].sum())

    x_labels = ["Baseline KPI"]
    y_values = [baseline_val]
    measures = ["absolute"]
    text_labels = [f"{baseline_val:.{decimals}f}{unit}"]

    for _, row in top_df.iterrows():
        c_val = float(row["total_contribution"])
        x_labels.append(str(row["slice_short_label"]))
        y_values.append(c_val)
        measures.append("relative")
        text_labels.append(f"{c_val:+.{decimals + 1}f}{unit}")

    if abs(residual_contrib) > 1e-5 and len(sorted_df) > top_n:
        x_labels.append("All Other Segments")
        y_values.append(residual_contrib)
        measures.append("relative")
        text_labels.append(f"{residual_contrib:+.{decimals + 1}f}{unit}")

    x_labels.append("Target Date KPI")
    y_values.append(target_val)
    measures.append("total")
    text_labels.append(f"{target_val:.{decimals}f}{unit}")

    inc_color = COLOR_PALETTE["teal"] if higher_is_better else COLOR_PALETTE["crimson"]
    dec_color = COLOR_PALETTE["crimson"] if higher_is_better else COLOR_PALETTE["teal"]

    fig = go.Figure(
        go.Waterfall(
            name="KPI Bridge",
            orientation="v",
            measure=measures,
            x=x_labels,
            y=y_values,
            text=text_labels,
            textposition="outside",
            connector=dict(line=dict(color="#94A3B8", width=1.2, dash="dot")),
            increasing=dict(marker=dict(color=inc_color)),
            decreasing=dict(marker=dict(color=dec_color)),
            totals=dict(marker=dict(color=COLOR_PALETTE["navy"])),
        )
    )

    min_bound = min(baseline_val, target_val)
    max_bound = max(baseline_val, target_val)
    span = max(abs(max_bound - min_bound), 1.0)

    fig.update_layout(
        title=dict(
            text=f"KPI Variance Bridge Waterfall — {scope_title}",
            font=dict(size=13.5, color="#0F172A", family="Inter, sans-serif"),
            x=0.0,
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=430,
        margin=dict(l=40, r=20, t=50, b=85),
        xaxis=dict(
            tickangle=-22,
            tickfont=dict(size=10.5, color="#0F172A"),
            showgrid=False,
            linecolor="#CBD5E1",
        ),
        yaxis=dict(
            title=f"{kpi_meta['label']} ({unit})",
            range=[min_bound - span * 0.45, max_bound + span * 0.35],
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
        ),
        showlegend=False,
    )
    return fig


def build_mix_rate_bar_chart(
    df_slices: pd.DataFrame,
    kpi_key: str,
    top_n: int = 8,
    scope_title: str = "Top Contributing Drivers",
) -> go.Figure:
    """Construct the Tab 2 Mix vs. Rate Stacked Horizontal Bar Chart."""
    kpi_meta = SUPPORTED_KPIS[kpi_key]
    delta_unit = str(kpi_meta["delta_unit"])

    top_df = (
        df_slices.sort_values("variance_explained_pct", ascending=False)
        .head(top_n)
        .iloc[::-1]
        .reset_index(drop=True)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=top_df["slice_short_label"],
            x=top_df["rate_effect"],
            name="Rate Effect (Within-Segment Performance Change)",
            orientation="h",
            marker=dict(color=COLOR_PALETTE["crimson"]),
            hovertemplate="<b>%{y}</b><br>Rate Effect: %{x:+.3f} " + delta_unit + "<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            y=top_df["slice_short_label"],
            x=top_df["mix_effect"],
