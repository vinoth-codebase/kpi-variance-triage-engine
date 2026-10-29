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
            name="Mix Effect (Volume Share Shift)",
            orientation="h",
            marker=dict(color=COLOR_PALETTE["navy"]),
            hovertemplate="<b>%{y}</b><br>Mix Effect: %{x:+.3f} " + delta_unit + "<extra></extra>",
        )
    )

    fig.add_vline(x=0.0, line_width=1.2, line_color="#475569")

    fig.update_layout(
        title=dict(
            text=f"Rate vs. Volume Mix Decomposition ({delta_unit}) — {scope_title}",
            font=dict(size=13.5, color="#0F172A", family="Inter, sans-serif"),
            x=0.0,
        ),
        barmode="relative",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=430,
        margin=dict(l=180, r=25, t=50, b=45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=10.5, color="#475569"),
        ),
        xaxis=dict(
            title=f"Contribution to Top-Level KPI Delta ({delta_unit})",
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
        ),
        yaxis=dict(
            showgrid=False,
            linecolor="#CBD5E1",
            tickfont=dict(size=10.5, color="#0F172A"),
        ),
    )
    return fig


def build_hierarchical_drilldown_chart(
    target_df: pd.DataFrame,
    kpi_key: str,
    hierarchy_cols: List[str],
    chart_type: str = "Sunburst",
) -> go.Figure:
    """Construct the Tab 3 Interactive Sunburst or Treemap multi-dimensional hierarchy chart."""
    kpi_meta = SUPPORTED_KPIS[kpi_key]
    higher_is_better = bool(kpi_meta["higher_is_better"])
    label = str(kpi_meta["label"])
    unit = str(kpi_meta["value_suffix"])

    grouped = (
        target_df.groupby(hierarchy_cols, dropna=False)
        .agg(
            execution_volume=(kpi_key, "count"),
            kpi_metric=(kpi_key, "mean"),
        )
        .reset_index()
    )

    color_scale = (
        ["#DC2626", "#FDE68A", "#0D9488"]
        if higher_is_better
        else ["#0D9488", "#FDE68A", "#DC2626"]
    )

    hierarchy_title = " -> ".join(DIMENSION_LABELS.get(c, c) for c in hierarchy_cols)

    if chart_type == "Treemap":
        fig = px.treemap(
            grouped,
            path=hierarchy_cols,
            values="execution_volume",
            color="kpi_metric",
            color_continuous_scale=color_scale,
            labels={"kpi_metric": f"{label} ({unit})", "execution_volume": "Executions"},
        )
    else:
        fig = px.sunburst(
            grouped,
            path=hierarchy_cols,
            values="execution_volume",
            color="kpi_metric",
            color_continuous_scale=color_scale,
            labels={"kpi_metric": f"{label} ({unit})", "execution_volume": "Executions"},
        )

    fig.update_layout(
        title=dict(
            text=f"Multi-Dimensional Hierarchy Drilldown ({hierarchy_title})",
            font=dict(size=13.5, color="#0F172A", family="Inter, sans-serif"),
            x=0.0,
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=470,
        margin=dict(l=15, r=15, t=48, b=15),
        coloraxis_colorbar=dict(title=f"{unit}"),
    )
    return fig


def main() -> None:
    """Render the Automated KPI Variance & Root-Cause Triage Engine interface."""
    inject_institutional_css()

    daily_df = load_daily_warehouse_series()

    # ------------------------------------------------------------------
    # Top Institutional Header
    # ------------------------------------------------------------------
    st.markdown(
        """
        <div class="portal-header">
            <div class="portal-eyebrow">ENTERPRISE OPERATIONS ANALYTICS // AUTOMATED DIAGNOSTIC WORKBENCH</div>
            <div class="portal-title">Automated KPI Variance &amp; Root-Cause Triage Engine</div>
            <div class="portal-subtitle">
                Statistical time-series surveillance, recursive Berman/Kitagawa Mix vs. Rate attribution,
                and multi-dimensional root-cause isolation across 60,000 Star-Schema workflow executions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Top Horizontal Control Bar
    # ------------------------------------------------------------------
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.35, 1.45, 1.15, 1.05])

    kpi_options = list(SUPPORTED_KPIS.keys())
    with ctrl_col1:
        selected_kpi = st.selectbox(
            "Target Operational KPI",
            options=kpi_options,
            format_func=lambda k: f"{SUPPORTED_KPIS[k]['label']} ({SUPPORTED_KPIS[k]['unit']})",
            index=0,
        )

    with ctrl_col4:
        z_threshold = st.slider(
            "Z-Score Sensitivity Threshold",
            min_value=1.5,
            max_value=3.5,
            value=float(Z_SCORE_THRESHOLD),
            step=0.1,
        )

    # Run Anomaly Detector on selected KPI to annotate dates in selector
    detector = KPIAnomalyDetector(z_threshold=z_threshold)
    surveillance_df = detector.analyze_series(daily_df, selected_kpi)

    # Track KPI selection changes to default to that KPI's primary surgical anomaly date
    default_day_idx = int(SUPPORTED_KPIS[selected_kpi]["default_anomaly_day_index"])
    date_list = surveillance_df["execution_date"].tolist()
    day_idx_list = surveillance_df["day_index"].tolist()
    default_date_idx = (
        day_idx_list.index(default_day_idx)
        if default_day_idx in day_idx_list
        else len(date_list) - 1
    )

    if "last_selected_kpi" not in st.session_state or st.session_state["last_selected_kpi"] != selected_kpi:
        st.session_state["last_selected_kpi"] = selected_kpi
        st.session_state["target_date_select"] = date_list[default_date_idx]

    date_label_map: Dict[str, str] = {}
    for _, r in surveillance_df.iterrows():
        d_str = str(r["execution_date"])
        d_idx = int(r["day_index"])
        flag_suffix = f" [ANOMALY: Z={r['z_score']:+.2f}]" if bool(r["is_anomaly"]) else ""
        date_label_map[d_str] = f"{d_str} (Day {d_idx:02d}){flag_suffix}"

    with ctrl_col2:
        selected_date = st.selectbox(
            "Target Analysis Date",
            options=date_list,
            format_func=lambda d: date_label_map.get(d, d),
            key="target_date_select",
        )

    with ctrl_col3:
        selected_baseline_mode = st.selectbox(
            "Baseline Comparison Mode",
            options=BASELINE_MODES,
            index=0,
        )

    # ------------------------------------------------------------------
    # Execute Surveillance Summary & Variance Attribution Engine
    # ------------------------------------------------------------------
    day_summary = detector.get_target_day_summary(
        surveillance_df=surveillance_df,
        target_date=selected_date,
        baseline_mode=selected_baseline_mode,
    )

    baseline_dates = day_summary["baseline_dates"]
    baseline_records_df = load_window_fact_records(tuple(baseline_dates))
    target_records_df = load_window_fact_records((selected_date,))

    engine = VarianceAttributionEngine()
    attribution = engine.run_full_attribution(
        baseline_df=baseline_records_df,
        target_df=target_records_df,
        kpi_key=selected_kpi,
        z_score=day_summary["z_score"],
        p_value=day_summary["p_value"],
    )

    kpi_meta = SUPPORTED_KPIS[selected_kpi]
    unit = str(kpi_meta["value_suffix"])
    delta_unit = str(kpi_meta["delta_unit"])
    decimals = int(kpi_meta["decimals"])
    higher_is_better = bool(kpi_meta["higher_is_better"])

    # ------------------------------------------------------------------
    # Executive Diagnostic Banner
    # ------------------------------------------------------------------
    top_2d = attribution["top_2d_driver"]
    banner_class = "exec-banner exec-banner-critical" if day_summary["is_anomaly"] else "exec-banner"
    status_badge_text = day_summary["status_badge"]
    if day_summary["severity"] == "SEVERITY_HIGH":
        badge_css = "status-badge badge-critical"
    elif day_summary["severity"] == "SEVERITY_MODERATE":
        badge_css = "status-badge badge-moderate"
    else:
        badge_css = "status-badge badge-normal"

    st.markdown(
        f"""
        <div class="{banner_class}">
            <div class="exec-banner-header">
                <span class="exec-banner-label">AUTOMATED EXECUTIVE DIAGNOSTIC BRIEF // {selected_date} (DAY {day_summary['day_index']})</span>
                <span class="{badge_css}">{status_badge_text}</span>
            </div>
            <div class="exec-banner-body">
                {attribution['executive_narrative']}
            </div>
            <div class="exec-meta-strip">
                <div class="exec-meta-item"><strong>Primary 2D Driver:</strong> {top_2d['slice_label']}</div>
                <div class="exec-meta-item"><strong>Variance Explained:</strong> {top_2d['variance_explained_pct']:.1f}%</div>
                <div class="exec-meta-item"><strong>Rate vs. Mix Split:</strong> {top_2d['rate_share_of_slice_pct']:.1f}% Rate / {top_2d['mix_share_of_slice_pct']:.1f}% Mix</div>
                <div class="exec-meta-item"><strong>Slice Significance:</strong> p = {top_2d['p_value']:.2e} (Welch t = {top_2d['t_statistic']:+.2f})</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # 4 Core KPI Cards
    # ------------------------------------------------------------------
    target_val = attribution["target_overall"]
    base_val = attribution["baseline_overall"]
    total_delta = attribution["total_delta"]
    rel_pct = (total_delta / base_val * 100.0) if base_val != 0 else 0.0

    is_adverse = (higher_is_better and total_delta < 0) or (not higher_is_better and total_delta > 0)
    delta_color_class = "kpi-card-value-crimson" if is_adverse else "kpi-card-value-teal"

    card_c1, card_c2, card_c3, card_c4 = st.columns(4)

    with card_c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-card-label">Target Date Value ({selected_date})</div>
                <div class="kpi-card-value">{target_val:.{decimals}f}{unit}</div>
                <div class="kpi-card-sub">Sample Size: {len(target_records_df):,} workflow executions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card_c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-card-label">Baseline Value ({selected_baseline_mode})</div>
                <div class="kpi-card-value">{base_val:.{decimals}f}{unit}</div>
                <div class="kpi-card-sub">Window: {len(baseline_dates)} day(s) ({len(baseline_records_df):,} executions)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card_c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-card-label">Absolute &amp; Relative Variance (Delta)</div>
                <div class="kpi-card-value {delta_color_class}">{total_delta:+.{decimals}f} {delta_unit}</div>
                <div class="kpi-card-sub">Relative Shift: {rel_pct:+.2f}% vs. baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card_c4:
        p_display = "< 0.0001" if day_summary["p_value"] < 0.0001 else f"= {day_summary['p_value']:.4f}"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-card-label">Statistical Z-Score &amp; Control Status</div>
                <div class="kpi-card-value">Z = {day_summary['z_score']:+.2f}</div>
                <div class="kpi-card-sub"><span class="{badge_css}">{status_badge_text}</span> &nbsp; p {p_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 0.85rem;'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 4 Analytical Workbench Tabs
    # ------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Tab 1: Time-Series Surveillance & Control Bands",
            "Tab 2: Variance Waterfall & Mix/Rate Decomposition",
            "Tab 3: Multi-Dimensional Root-Cause Isolation Matrix",
            "Tab 4: Record-Level Drilldown & Audit Export",
        ]
    )

    # ==================================================================
    # TAB 1: TIME-SERIES SURVEILLANCE & CONTROL BANDS
    # ==================================================================
    with tab1:
        st.plotly_chart(
            build_surveillance_chart(
                surveillance_df=surveillance_df,
                kpi_key=selected_kpi,
                selected_date=selected_date,
                z_threshold=z_threshold,
            ),
            use_container_width=True,
        )

        col_anom_left, col_anom_right = st.columns([1.05, 1.15])

        with col_anom_left:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-title">Detected Statistical Anomalies — Active KPI</div>
                    <div class="section-subtitle">
                        Dates breaching the Z-score control threshold across the 60-day surveillance window.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            flagged_df = surveillance_df[surveillance_df["is_anomaly"]].sort_values(
                "abs_z_score", ascending=False
            )
            if flagged_df.empty:
                st.info("No dates breached the current Z-score threshold for this KPI.")
            else:
                display_anom = pd.DataFrame(
                    {
                        "Date": flagged_df["execution_date"],
                        "Day": flagged_df["day_index"],
                        f"Observed ({unit})": flagged_df["kpi_value"].round(decimals),
                        f"14D Baseline ({unit})": flagged_df["prior_14d_mean"].round(decimals),
                        f"Delta ({delta_unit})": flagged_df["prior_14d_delta"].round(decimals),
                        "Z-Score": flagged_df["z_score"].round(2),
                        "P-Value": flagged_df["p_value"].map(lambda p: f"{p:.2e}"),
                        "Status": flagged_df["status_badge"],
                    }
                )
                st.dataframe(display_anom, use_container_width=True, hide_index=True)

        with col_anom_right:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-title">Warehouse Cross-KPI Anomaly Catalog (60-Day Horizon)</div>
                    <div class="section-subtitle">
                        Summary of all detected high-severity anomalies across all 3 operational KPIs.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            cross_rows: List[Dict[str, Any]] = []
            for k_name, k_cfg in SUPPORTED_KPIS.items():
                s_df = detector.analyze_series(daily_df, k_name)
                for _, a_row in s_df[s_df["is_anomaly"]].iterrows():
                    u = str(k_cfg["value_suffix"])
                    dec = int(k_cfg["decimals"])
                    cross_rows.append(
                        {
                            "KPI": str(k_cfg["label"]),
                            "Anomaly Date": f"{a_row['execution_date']} (Day {int(a_row['day_index'])})",
                            "Baseline": f"{a_row['prior_14d_mean']:.{dec}f}{u}",
                            "Observed": f"{a_row['kpi_value']:.{dec}f}{u}",
                            "Delta": f"{a_row['prior_14d_delta']:+.{dec}f} {k_cfg['delta_unit']}",
                            "Z-Score": f"{a_row['z_score']:+.2f}",
                            "Severity": str(a_row["status_badge"]),
                        }
                    )
            st.dataframe(pd.DataFrame(cross_rows), use_container_width=True, hide_index=True)

    # ==================================================================
    # TAB 2: VARIANCE WATERFALL & MIX/RATE DECOMPOSITION
    # ==================================================================
    with tab2:
        scope_options = ["Top 2D Root-Cause Interactions"] + [
            f"1D Dimension: {DIMENSION_LABELS[d]} ({d})" for d in DIMENSIONS
        ]
        wf_ctrl1, wf_ctrl2 = st.columns([1.8, 1.2])
        with wf_ctrl1:
            selected_scope = st.selectbox(
                "Decomposition View Scope",
                options=scope_options,
                index=0,
            )
        with wf_ctrl2:
            top_n_bars = st.slider(
                "Maximum Slices Displayed in Bridge",
                min_value=4,
                max_value=12,
                value=6,
                step=1,
            )

        if selected_scope == "Top 2D Root-Cause Interactions":
            # Filter to the top-ranked 2D dimension pair so the bridge steps are mutually exclusive
            top_pair_key = str(attribution["top_2d_driver"]["dimension_key"])
            top_pair_label = str(attribution["top_2d_driver"]["dimension_label"])
            active_slices_df = attribution["df_2d"][
                attribution["df_2d"]["dimension_key"] == top_pair_key
            ].copy()
            scope_title = f"Primary 2D Interaction Grid ({top_pair_label})"
        else:
            chosen_dim = [
                d for d in DIMENSIONS if f"({d})" in selected_scope
            ][0]
            active_slices_df = attribution["df_1d"][
                attribution["df_1d"]["dimension_key"] == chosen_dim
            ].copy()
            scope_title = f"Single Dimension ({DIMENSION_LABELS[chosen_dim]})"

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(
                build_waterfall_chart(
                    df_slices=active_slices_df,
                    baseline_val=base_val,
                    target_val=target_val,
                    kpi_key=selected_kpi,
                    top_n=top_n_bars,
                    scope_title=scope_title,
                ),
                use_container_width=True,
            )
        with chart_col2:
            st.plotly_chart(
                build_mix_rate_bar_chart(
                    df_slices=active_slices_df,
                    kpi_key=selected_kpi,
                    top_n=min(top_n_bars + 2, len(active_slices_df)),
                    scope_title=scope_title,
                ),
                use_container_width=True,
            )

        # Mathematical Reconciliation Proof Table across all 6 Single Dimensions
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Exact Mathematical Variance Reconciliation Check (Berman/Kitagawa Identity)</div>
                <div class="section-subtitle">
                    Verifies that across every single analytical dimension, Sum(Rate Effect) + Sum(Mix Effect) reconciles to 100.0% of the Top-Level KPI Delta.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        recon_rows: List[Dict[str, Any]] = []
        df_1d_all = attribution["df_1d"]
        for dim_col in DIMENSIONS:
            sub = df_1d_all[df_1d_all["dimension_key"] == dim_col]
            sum_rate = float(sub["rate_effect"].sum())
            sum_mix = float(sub["mix_effect"].sum())
            sum_total = float(sub["total_contribution"].sum())
            sum_pct = float(sub["variance_explained_pct"].sum())
            top_seg = sub.sort_values("variance_explained_pct", ascending=False).iloc[0]
            recon_rows.append(
                {
                    "Dimension": DIMENSION_LABELS[dim_col],
                    "Distinct Segments": len(sub),
                    "Top Contributing Segment": str(top_seg["slice_short_label"]),
                    "Top Segment Share (%)": f"{top_seg['variance_explained_pct']:.1f}%",
                    f"Sum Rate Effect ({delta_unit})": f"{sum_rate:+.4f}",
                    f"Sum Mix Effect ({delta_unit})": f"{sum_mix:+.4f}",
                    f"Reconciled Total Delta ({delta_unit})": f"{sum_total:+.4f}",
                    "Total Variance Explained (%)": f"{sum_pct:.2f}%",
                }
            )
        st.dataframe(pd.DataFrame(recon_rows), use_container_width=True, hide_index=True)

    # ==================================================================
    # TAB 3: MULTI-DIMENSIONAL ROOT-CAUSE ISOLATION MATRIX
    # ==================================================================
    with tab3:
        m_col1, m_col2, m_col3 = st.columns([1.2, 1.2, 1.0])
        with m_col1:
            dim_type_filter = st.selectbox(
                "Slice Dimensionality Filter",
                options=["All Slices (1D + 2D)", "2D Interactions Only", "1D Single Dimensions Only"],
                index=0,
            )
        with m_col2:
            hierarchy_mode = st.selectbox(
                "Drilldown Hierarchy Path",
                options=[
                    "Default Spec: Delivery Tower -> System Environment -> Shift Tier",
                    "KPI Optimal Hierarchy Path",
                ],
                index=0,
            )
        with m_col3:
            hierarchy_chart_type = st.radio(
                "Hierarchy Chart Type",
                options=["Sunburst", "Treemap"],
                horizontal=True,
            )

        leaderboard_df = attribution["combined_leaderboard"].copy()
        if dim_type_filter == "2D Interactions Only":
            leaderboard_df = leaderboard_df[leaderboard_df["dimension_type"] == "2D"]
        elif dim_type_filter == "1D Single Dimensions Only":
            leaderboard_df = leaderboard_df[leaderboard_df["dimension_type"] == "1D"]

        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Multi-Dimensional Root-Cause Isolation Matrix (Ranked by % Variance Explained)</div>
                <div class="section-subtitle">
                    Combinatorial evaluation across all 1D and 2D slices with Welch two-sample t-test p-values and exact Mix/Rate attribution.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        display_matrix = pd.DataFrame(
            {
                "Rank": range(1, len(leaderboard_df.head(25)) + 1),
                "Type": leaderboard_df.head(25)["dimension_type"],
                "Dimension(s)": leaderboard_df.head(25)["dimension_label"],
                "Segment Slice": leaderboard_df.head(25)["slice_label"],
                "Target Vol": leaderboard_df.head(25)["target_volume"],
                f"Baseline ({unit})": leaderboard_df.head(25)["baseline_metric"].round(decimals),
                f"Target ({unit})": leaderboard_df.head(25)["target_metric"].round(decimals),
                f"Delta ({delta_unit})": leaderboard_df.head(25)["metric_delta"].round(decimals),
                f"Rate Effect ({delta_unit})": leaderboard_df.head(25)["rate_effect"].round(3),
                f"Mix Effect ({delta_unit})": leaderboard_df.head(25)["mix_effect"].round(3),
                f"Contribution ({delta_unit})": leaderboard_df.head(25)["total_contribution"].round(3),
                "% Variance Explained": leaderboard_df.head(25)["variance_explained_pct"].map(
                    lambda v: f"{v:.1f}%"
                ),
                "P-Value": leaderboard_df.head(25)["p_value"].map(
                    lambda p: "< 0.0001" if p < 0.0001 else f"{p:.4f}"
                ),
            }
        )
        st.dataframe(display_matrix, use_container_width=True, hide_index=True)

        sun_col, tree_col = st.columns([1.25, 1.0])
        with sun_col:
            if hierarchy_mode.startswith("Default Spec"):
                h_cols = ["delivery_tower", "system_environment", "shift_tier"]
            else:
                h_cols = list(kpi_meta["sunburst_hierarchy"])

            st.plotly_chart(
                build_hierarchical_drilldown_chart(
                    target_df=target_records_df,
                    kpi_key=selected_kpi,
                    hierarchy_cols=h_cols,
                    chart_type=hierarchy_chart_type,
                ),
                use_container_width=True,
            )

        with tree_col:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-title">Decision-Tree Failure Cluster Rules (max_depth = 3)</div>
                    <div class="section-subtitle">
                        Automated scikit-learn DecisionTreeRegressor partition rules ranking highest-density failure pockets on the Target Date.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            tree_rules = attribution.get("decision_tree_rules", [])
            if tree_rules:
                tree_df = pd.DataFrame(
                    [
                        {
                            "Cluster Rank": idx + 1,
                            "Decision Rule Path": r["rule_path"],
                            "Executions": r["sample_count"],
                            f"Cluster {unit}": f"{r['leaf_metric']:.{decimals}f}{unit}",
                            f"vs. Day Mean ({delta_unit})": f"{r['metric_deviation']:+.{decimals}f}",
                        }
                        for idx, r in enumerate(tree_rules[:8])
                    ]
                )
                st.dataframe(tree_df, use_container_width=True, hide_index=True)

    # ==================================================================
    # TAB 4: RECORD-LEVEL DRILLDOWN & AUDIT EXPORT
    # ==================================================================
    with tab4:
        primary_filter: Dict[str, str] = top_2d["slice_filter"]
        primary_slice_str = top_2d["slice_label"]

        drill_ctrl1, drill_ctrl2 = st.columns([1.6, 1.0])
        with drill_ctrl1:
            filter_mode = st.radio(
                "Fact Record Inspection Scope",
                options=[
                    f"Auto-Isolated #1 Root-Cause Segment: [{primary_slice_str}]",
                    "All Target Date Workflow Executions (Unfiltered)",
                ],
                horizontal=False,
            )

        active_filter = primary_filter if filter_mode.startswith("Auto-Isolated") else None
        raw_drilldown_df = fetch_filtered_fact_records(
            execution_date=selected_date,
            dimension_filters=active_filter,
            limit=1000,
            db_path=DB_PATH,
        )

        with drill_ctrl2:
            st.markdown(
                """
                <div class="section-card">
                    <div class="section-title">Executive Root-Cause Audit Pack (Excel)</div>
                    <div class="section-subtitle">
                        Generate 4-sheet openpyxl report (Executive_Summary, 1D_Dimension_Breakdown, 2D_Interaction_Ranking, Anomalous_Raw_Records).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            export_path = generate_audit_pack_excel(
                kpi_key=selected_kpi,
                surveillance_summary=day_summary,
                attribution_results=attribution,
                raw_records_df=raw_drilldown_df,
                output_dir=EXPORTS_DIR,
            )
            with open(export_path, "rb") as f_excel:
                excel_bytes = f_excel.read()

            st.download_button(
                label=f"Download Excel Audit Report ({export_path.name})",
                data=excel_bytes,
                file_name=export_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        # Summary telemetry for the isolated raw records
        s_c1, s_c2, s_c3, s_c4 = st.columns(4)
        total_slice_rows = len(raw_drilldown_df)
        sla_breaches = int((raw_drilldown_df["sla_met"] == 0).sum()) if total_slice_rows > 0 else 0
        fpy_defects = int((raw_drilldown_df["first_pass_yield"] == 0).sum()) if total_slice_rows > 0 else 0
        non_none_errors = raw_drilldown_df[raw_drilldown_df["error_code"] != "NONE"]
        dominant_error = (
            str(non_none_errors["error_code"].mode().iloc[0])
            if not non_none_errors.empty
            else "NONE"
        )

        s_c1.metric("Isolated Workflow Executions", f"{total_slice_rows:,}")
        s_c2.metric("SLA Breach Count", f"{sla_breaches:,}")
        s_c3.metric("First-Pass Defect Count", f"{fpy_defects:,}")
        s_c4.metric("Dominant Diagnostic Error Code", dominant_error)

        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-title">Granular SQLite Fact Records — {selected_date}</div>
                <div class="section-subtitle">
                    Showing underlying workflow execution logs from fact_workflow_executions ordered by failure severity.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(raw_drilldown_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

