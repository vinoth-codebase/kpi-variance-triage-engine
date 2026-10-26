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

