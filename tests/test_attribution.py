"""Comprehensive pytest suite for the Automated KPI Variance & Root-Cause Triage Engine."""

import re
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BASE_DIR,
    DB_PATH,
    EXPORTS_DIR,
    DIMENSIONS,
    SUPPORTED_KPIS,
)
from src.database import (
    fetch_daily_kpi_series,
    fetch_records_for_dates,
    fetch_filtered_fact_records,
    fetch_dimension_catalog,
    is_warehouse_populated,
)
from src.data_generator import generate_warehouse_data
from src.anomaly_detector import KPIAnomalyDetector
from src.attribution_engine import VarianceAttributionEngine
from src.exporter import generate_audit_pack_excel


# Comprehensive Unicode Emoji Regex pattern covering emoticons, dingbats, transport/map,
# supplemental symbols, flags, and pictographs
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002600-\U000026FF"  # Miscellaneous Symbols
    "\U00002700-\U000027BF"  # Dingbats
    "\U0001F1E0-\U0001F1FF"  # Flags
    "]+",
    flags=re.UNICODE,
)


@pytest.fixture(scope="module", autouse=True)
def ensure_warehouse() -> None:
    """Ensure the SQLite warehouse is built before running tests."""
    if not is_warehouse_populated(DB_PATH):
        generate_warehouse_data(DB_PATH, force_rebuild=True)


def test_star_schema_warehouse_integrity() -> None:
    """Verify the Star-Schema SQLite database has 60 days and all 6 dimension tables populated."""
    assert is_warehouse_populated(DB_PATH)
    daily_df = fetch_daily_kpi_series(DB_PATH)
    assert len(daily_df) == 60
    assert int(daily_df["total_executions"].sum()) == 60000

    dim_catalog = fetch_dimension_catalog(DB_PATH)
    assert len(dim_catalog) == 6
    for tbl_name, df_dim in dim_catalog.items():
        assert not df_dim.empty, f"Dimension table {tbl_name} should not be empty."


def test_anomaly_detector_flags_day_60_and_all_surgical_anomalies() -> None:
    """Verify KPIAnomalyDetector accurately flags Day 60 (abs(z_score) > 2.5), Day 52, and Day 45."""
    daily_df = fetch_daily_kpi_series(DB_PATH)
    detector = KPIAnomalyDetector(z_threshold=2.0)

    # 1. Day 60: SLA Compliance Drop (93.8% -> 81.4%)
    sla_surv = detector.analyze_series(daily_df, "sla_compliance_rate")
    day_60 = sla_surv[sla_surv["day_index"] == 60].iloc[0]
    assert bool(day_60["is_anomaly"]) is True
    assert day_60["anomaly_flag"] == "ANOMALY_DETECTED"
    assert day_60["severity"] == "SEVERITY_HIGH"
    assert float(day_60["abs_z_score"]) > 2.5
    assert float(day_60["prior_14d_mean"]) == pytest.approx(93.8, abs=0.1)
    assert float(day_60["kpi_value"]) == pytest.approx(81.4, abs=0.1)

    # 2. Day 52: First Pass Yield Defect Leakage Spike (95.1% -> 84.6%)
    fpy_surv = detector.analyze_series(daily_df, "first_pass_yield_rate")
    day_52 = fpy_surv[fpy_surv["day_index"] == 52].iloc[0]
    assert bool(day_52["is_anomaly"]) is True
    assert day_52["severity"] == "SEVERITY_HIGH"
    assert float(day_52["abs_z_score"]) > 2.5
    assert float(day_52["prior_14d_mean"]) == pytest.approx(95.1, abs=0.1)
    assert float(day_52["kpi_value"]) == pytest.approx(84.6, abs=0.1)

    # 3. Day 45: Mean Resolution Hours Surge (4.2h -> 7.9h)
    res_surv = detector.analyze_series(daily_df, "mean_resolution_hours")
    day_45 = res_surv[res_surv["day_index"] == 45].iloc[0]
    assert bool(day_45["is_anomaly"]) is True
    assert day_45["severity"] == "SEVERITY_HIGH"
    assert float(day_45["abs_z_score"]) > 2.5
    assert float(day_45["prior_14d_mean"]) == pytest.approx(4.2, abs=0.1)
    assert float(day_45["kpi_value"]) == pytest.approx(7.9, abs=0.1)


def test_single_dimension_variance_reconciles_to_100_percent() -> None:
