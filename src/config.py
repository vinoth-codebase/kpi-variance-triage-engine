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
