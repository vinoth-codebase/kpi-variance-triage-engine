"""Executive Audit Pack Excel Generator using openpyxl."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from src.config import EXPORTS_DIR, SUPPORTED_KPIS


def _apply_sheet_styling(ws: Any, header_row: int = 1) -> None:
    """Apply institutional table styling and auto-adjust column widths on a worksheet."""
    navy_fill = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
    zebra_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    header_font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    body_font = Font(name="Calibri", size=10, bold=False, color="0F172A")
    thin_side = Side(border_style="thin", color="E2E8F0")
    cell_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

    max_row = ws.max_row
    max_col = ws.max_column

    for col_idx in range(1, max_col + 1):
        cell = ws.cell(row=header_row, column=col_idx)
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        cell.border = cell_border

    for row_idx in range(header_row + 1, max_row + 1):
        is_even = (row_idx - header_row) % 2 == 0
        for col_idx in range(1, max_col + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.font = body_font
            if is_even:
                cell.fill = zebra_fill
            cell.border = cell_border
            if isinstance(cell.value, (int, float)):
                cell.alignment = Alignment(horizontal="right", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="center")

    for col_idx in range(1, max_col + 1):
        col_letter = get_column_letter(col_idx)
        max_len = 12
        for row_idx in range(header_row, min(max_row + 1, header_row + 200)):
            val = ws.cell(row=row_idx, column=col_idx).value
            if val is not None:
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                max_len = max(max_len, min(len(val_str) + 4, 48))
        ws.column_dimensions[col_letter].width = max_len


def generate_audit_pack_excel(
    kpi_key: str,
    surveillance_summary: Dict[str, Any],
    attribution_results: Dict[str, Any],
    raw_records_df: pd.DataFrame,
    output_dir: Path = EXPORTS_DIR,
    custom_filename: Optional[str] = None,
) -> Path:
    """
    Generate a 4-sheet formatted Excel Root-Cause Triage Audit Report.

    Sheets:
        1. Executive_Summary
        2. 1D_Dimension_Breakdown
        3. 2D_Interaction_Ranking
        4. Anomalous_Raw_Records
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    kpi_meta = SUPPORTED_KPIS[kpi_key]
    target_date = str(surveillance_summary["execution_date"])

    if custom_filename:
        file_path = output_dir / custom_filename
    else:
        safe_date = target_date.replace("-", "")
        file_path = output_dir / f"KPI_Root_Cause_Audit_{kpi_key}_{safe_date}.xlsx"

    wb = Workbook()

    # ------------------------------------------------------------------
    # Sheet 1: Executive_Summary
    # ------------------------------------------------------------------
    ws_exec = wb.active
    ws_exec.title = "Executive_Summary"

    title_font = Font(name="Calibri", size=14, bold=True, color="0F172A")
    section_font = Font(name="Calibri", size=11, bold=True, color="0F172A")
    label_font = Font(name="Calibri", size=10, bold=True, color="475569")
    val_font = Font(name="Calibri", size=10, bold=False, color="0F172A")

    ws_exec["A1"] = "AUTOMATED KPI VARIANCE & ROOT-CAUSE TRIAGE ENGINE - EXECUTIVE AUDIT REPORT"
    ws_exec["A1"].font = title_font

    ws_exec["A3"] = "1. SURVEILLANCE & BASELINE METADATA"
    ws_exec["A3"].font = section_font

    top_2d = attribution_results["top_2d_driver"]
    unit = str(kpi_meta["value_suffix"])
    decimals = int(kpi_meta["decimals"])

    summary_rows = [
        ("Report Generated Timestamp (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
        ("Target Operational KPI", f"{kpi_meta['label']} ({kpi_key})"),
        ("KPI Definition", str(kpi_meta["description"])),
        ("Target Analysis Date", f"{target_date} (Day {surveillance_summary['day_index']})"),
        ("Baseline Comparison Window", f"{surveillance_summary['baseline_mode']} ({surveillance_summary['baseline_days_count']} days)"),
        ("Baseline KPI Value", f"{attribution_results['baseline_overall']:.{decimals}f}{unit}"),
        ("Target Date KPI Value", f"{attribution_results['target_overall']:.{decimals}f}{unit}"),
        ("Top-Level Absolute Variance", f"{attribution_results['total_delta']:+.{decimals}f} {kpi_meta['delta_unit']}"),
        ("Statistical Z-Score", f"{surveillance_summary['z_score']:+.2f}"),
        ("Two-Tailed P-Value", f"{surveillance_summary['p_value']:.6f}"),
        ("Surveillance Status Classification", str(surveillance_summary["status_badge"])),
        ("Primary Root-Cause 2D Slice", str(top_2d["slice_label"])),
        ("Primary Slice Variance Explained (%)", f"{top_2d['variance_explained_pct']:.1f}%"),
        ("Primary Slice Rate Effect", f"{top_2d['rate_effect']:+.4f}"),
        ("Primary Slice Mix Effect", f"{top_2d['mix_effect']:+.4f}"),
        ("Executive Diagnostic Brief", str(attribution_results["executive_narrative"])),
    ]

    for idx, (k_label, v_text) in enumerate(summary_rows, start=4):
        c_key = ws_exec.cell(row=idx, column=1, value=k_label)
        c_val = ws_exec.cell(row=idx, column=2, value=v_text)
        c_key.font = label_font
        c_val.font = val_font
        c_val.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)

    tree_start_row = len(summary_rows) + 6
    ws_exec.cell(row=tree_start_row, column=1, value="2. DECISION TREE MICRO-CLUSTER ISOLATION RULES").font = section_font

    tree_headers = [
        "Cluster Rank",
        "Decision Tree Rule Path",
        "Cluster Sample Count",
        "Volume Share (%)",
        "Cluster Metric Value",
        "Overall Day Metric",
        "Metric Deviation",
    ]
    header_r = tree_start_row + 1
    for c_idx, h_text in enumerate(tree_headers, start=1):
        ws_exec.cell(row=header_r, column=c_idx, value=h_text)

    for r_offset, rule in enumerate(attribution_results.get("decision_tree_rules", [])[:6], start=1):
        curr_r = header_r + r_offset
        ws_exec.cell(row=curr_r, column=1, value=r_offset)
        ws_exec.cell(row=curr_r, column=2, value=str(rule["rule_path"]))
        ws_exec.cell(row=curr_r, column=3, value=int(rule["sample_count"]))
        ws_exec.cell(row=curr_r, column=4, value=round(float(rule["sample_share_pct"]), 2))
        ws_exec.cell(row=curr_r, column=5, value=round(float(rule["leaf_metric"]), 3))
        ws_exec.cell(row=curr_r, column=6, value=round(float(rule["overall_metric"]), 3))
        ws_exec.cell(row=curr_r, column=7, value=round(float(rule["metric_deviation"]), 3))

    _apply_sheet_styling(ws_exec, header_row=header_r)
    ws_exec.column_dimensions["A"].width = 36
    ws_exec.column_dimensions["B"].width = 78

    # ------------------------------------------------------------------
    # Sheet 2: 1D_Dimension_Breakdown
    # ------------------------------------------------------------------
    ws_1d = wb.create_sheet(title="1D_Dimension_Breakdown")
    cols_export = [
        ("dimension_label", "Dimension"),
        ("slice_label", "Segment Slice"),
        ("baseline_volume", "Baseline Volume"),
        ("target_volume", "Target Volume"),
        ("baseline_share", "Baseline Share"),
        ("target_share", "Target Share"),
        ("baseline_metric", "Baseline Metric"),
        ("target_metric", "Target Metric"),
        ("metric_delta", "Metric Delta"),
        ("rate_effect", "Rate Effect"),
        ("mix_effect", "Mix Effect"),
        ("total_contribution", "Total Contribution"),
        ("variance_explained_pct", "Variance Explained (%)"),
        ("p_value", "P-Value"),
        ("is_significant", "Statistically Significant"),
    ]

    df_1d = attribution_results["df_1d"]
    for c_idx, (_, col_title) in enumerate(cols_export, start=1):
        ws_1d.cell(row=1, column=c_idx, value=col_title)

    for r_idx, (_, row) in enumerate(df_1d.iterrows(), start=2):
        for c_idx, (col_key, _) in enumerate(cols_export, start=1):
            val = row[col_key]
            if isinstance(val, (bool, np.bool_)):
                val = "YES" if bool(val) else "NO"
            elif isinstance(val, (float, np.floating)):
                val = round(float(val), 6)
            elif isinstance(val, (int, np.integer)):
                val = int(val)
            ws_1d.cell(row=r_idx, column=c_idx, value=val)

    _apply_sheet_styling(ws_1d, header_row=1)

