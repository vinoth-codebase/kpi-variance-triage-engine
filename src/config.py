"""Centralized configuration and constants for the KPI Variance & Root-Cause Triage Engine."""

from pathlib import Path
from typing import Dict, List, Any

# Directory and File Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
EXPORTS_DIR: Path = DATA_DIR / "exports"
DB_DIR: Path = BASE_DIR / "db"
DB_PATH: Path = DB_DIR / "kpi_triage_warehouse.db"

# Statistical & Surveillance Thresholds
Z_SCORE_THRESHOLD: float = 2.0
HIGH_SEVERITY_Z_THRESHOLD: float = 2.5
ROLLING_WINDOW_DAYS: int = 14
PRIOR_7D_WINDOW_DAYS: int = 7
TOP_N_DRIVERS: int = 10
SIGNIFICANCE_ALPHA: float = 0.05
MIN_SLICE_VOLUME: int = 10

# Synthetic Warehouse Generation Settings
SIMULATION_DAYS: int = 60
DAILY_EXECUTION_VOLUME: int = 1000
RANDOM_SEED: int = 42

# Analytical Dimensions
DIMENSIONS: List[str] = [
    "region",
    "delivery_tower",
    "process_stage",
    "shift_tier",
    "system_environment",
    "vendor_tier",
]

DIMENSION_LABELS: Dict[str, str] = {
    "region": "Geographic Region",
    "delivery_tower": "Delivery Tower",
    "process_stage": "Process Stage",
    "shift_tier": "Shift Tier",
    "system_environment": "System Environment",
    "vendor_tier": "Vendor Tier",
}

# Supported Operational & Executive KPIs
SUPPORTED_KPIS: Dict[str, Dict[str, Any]] = {
    "sla_compliance_rate": {
        "key": "sla_compliance_rate",
        "label": "SLA Compliance Rate",
        "description": "Percentage of workflows meeting target SLA",
        "unit": "%",
        "value_suffix": "%",
        "delta_unit": "pp",
        "decimals": 1,
        "higher_is_better": True,
        "target_benchmark": 93.8,
        "default_anomaly_day_index": 60,
        "sunburst_hierarchy": ["delivery_tower", "system_environment", "shift_tier"],
    },
    "first_pass_yield_rate": {
        "key": "first_pass_yield_rate",
        "label": "First Pass Yield Rate",
        "description": "Percentage of defect-free deliveries on first submission",
        "unit": "%",
        "value_suffix": "%",
        "delta_unit": "pp",
        "decimals": 1,
        "higher_is_better": True,
        "target_benchmark": 95.1,
        "default_anomaly_day_index": 52,
        "sunburst_hierarchy": ["process_stage", "vendor_tier", "delivery_tower"],
    },
    "mean_resolution_hours": {
        "key": "mean_resolution_hours",
        "label": "Mean Resolution Hours",
        "description": "Average turnaround time in hours",
        "unit": "h",
        "value_suffix": "h",
        "delta_unit": "h",
        "decimals": 2,
        "higher_is_better": False,
        "target_benchmark": 4.2,
        "default_anomaly_day_index": 45,
        "sunburst_hierarchy": ["region", "process_stage", "system_environment"],
    },
}

BASELINE_MODES: List[str] = [
    "Prior 14-Day Baseline",
    "Prior 7-Day Baseline",
    "Prior Day (DoD)",
]

# Corporate Consulting Color Palette
COLOR_PALETTE: Dict[str, str] = {
    "navy": "#0F172A",
    "slate": "#475569",
    "light_slate": "#94A3B8",
    "border": "#E2E8F0",
    "bg_subtle": "#F8FAFC",
    "white": "#FFFFFF",
    "crimson": "#DC2626",
    "crimson_light": "#FEE2E2",
    "amber": "#D97706",
    "amber_light": "#FEF3C7",
    "teal": "#0D9488",
    "teal_light": "#CCFBF1",
    "blue": "#1D4ED8",
}

