"""Synthetic Star-Schema operational warehouse generator with surgical root-cause anomalies."""

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DB_PATH,
    SIMULATION_DAYS,
    DAILY_EXECUTION_VOLUME,
    RANDOM_SEED,
)
from src.database import get_connection, init_star_schema, is_warehouse_populated


DIMENSION_METADATA: Dict[str, List[Tuple[Any, ...]]] = {
    "dim_region": [
        (1, "NA-East", "Americas", "M. Vance"),
        (2, "UK-Ireland", "EMEA-North", "S. Sterling"),
        (3, "DACH", "EMEA-Central", "H. Lindemann"),
        (4, "APAC-Singapore", "APAC", "K. Takahashi"),
    ],
    "dim_delivery_tower": [
        (1, "Core Data Engineering", "A. Subramanian", "Tier-0 Mission Critical"),
        (2, "Cloud Platform Ops", "D. Kowalski", "Tier-1 Infrastructure"),
        (3, "Enterprise Reporting", "E. Rostova", "Tier-2 Executive BI"),
        (4, "Risk & Regulatory Tech", "J. Thorne", "Tier-0 Compliance"),
    ],
    "dim_process_stage": [
        (1, "Ingestion & Staging", 1, 3.5),
        (2, "Transform & Reconciliation", 2, 4.5),
        (3, "UAT Validation", 3, 4.5),
        (4, "Security Sign-Off", 4, 5.0),
    ],
    "dim_shift_tier": [
        (1, "Shift-1 (Morning)", "06:00-14:00 UTC", "Primary Onshore Hub"),
        (2, "Shift-2 (Evening)", "14:00-22:00 UTC", "Follow-the-Sun Hybrid"),
        (3, "Shift-3 (Night)", "22:00-06:00 UTC", "Automated Batch & Offshore"),
    ],
    "dim_system_environment": [
        (1, "Snowflake-Prod-EU", "AWS-Frankfurt", "Production-Primary"),
        (2, "Snowflake-Prod-US", "AWS-Virginia", "Production-Primary"),
        (3, "Databricks-Core-EU", "Azure-WestEurope", "Production-Compute"),
        (4, "Synapse-Enterprise-Global", "Azure-EastUS", "Production-DW"),
    ],
    "dim_vendor_tier": [
        (1, "Internal FTE", "Direct Employment", "ISO-27001 Internal"),
        (2, "Tier-1 Strategic Partner", "Managed Outcome SLA", "Quarterly Audit"),
        (3, "Tier-2 Contractor", "Staff Augmentation", "Standard QA Gate"),
    ],
}


def _build_base_daily_template(n_records: int = DAILY_EXECUTION_VOLUME) -> pd.DataFrame:
    """Construct a balanced 1,000-record daily dimensional allocation template."""
    rng = np.random.default_rng(RANDOM_SEED)

    regions = ["NA-East", "UK-Ireland", "DACH", "APAC-Singapore"]
    towers = [
        "Core Data Engineering",
        "Cloud Platform Ops",
        "Enterprise Reporting",
        "Risk & Regulatory Tech",
    ]
    stages = [
        "Ingestion & Staging",
        "Transform & Reconciliation",
        "UAT Validation",
        "Security Sign-Off",
    ]
    shifts = ["Shift-1 (Morning)", "Shift-2 (Evening)", "Shift-3 (Night)"]
    envs = [
        "Snowflake-Prod-EU",
        "Snowflake-Prod-US",
        "Databricks-Core-EU",
        "Synapse-Enterprise-Global",
    ]
    vendors = ["Internal FTE", "Tier-1 Strategic Partner", "Tier-2 Contractor"]

    rows: List[Dict[str, str]] = []

    # 1. Allocate 240 records specifically to (Core Data Engineering, Snowflake-Prod-EU)
    for idx in range(240):
        shift_val = (
            "Shift-3 (Night)"
            if idx < 96
            else ("Shift-2 (Evening)" if idx < 168 else "Shift-1 (Morning)")
        )
        region_val = regions[idx % 4]
        stage_val = stages[(idx // 4) % 4]
        vendor_val = vendors[idx % 3]
        rows.append(
            {
                "region": region_val,
                "delivery_tower": "Core Data Engineering",
                "process_stage": stage_val,
                "shift_tier": shift_val,
                "system_environment": "Snowflake-Prod-EU",
                "vendor_tier": vendor_val,
            }
        )

    # 2. Allocate remaining 760 records with exact controlled slices for the other 2 anomalies
    # Need total (process_stage = UAT Validation, vendor_tier = Tier-2 Contractor) == 180
    # In first 240 rows, how many have stage == UAT Validation and vendor == Tier-2 Contractor?
    current_uat_t2 = sum(
        1
        for r in rows
        if r["process_stage"] == "UAT Validation" and r["vendor_tier"] == "Tier-2 Contractor"
    )
    needed_uat_t2 = 180 - current_uat_t2

    other_envs = ["Snowflake-Prod-US", "Databricks-Core-EU", "Synapse-Enterprise-Global"]
    other_towers = ["Cloud Platform Ops", "Enterprise Reporting", "Risk & Regulatory Tech"]

    for idx in range(needed_uat_t2):
        rows.append(
            {
                "region": regions[idx % 4],
                "delivery_tower": other_towers[idx % 3],
                "process_stage": "UAT Validation",
                "shift_tier": shifts[idx % 3],
                "system_environment": other_envs[idx % 3],
                "vendor_tier": "Tier-2 Contractor",
            }
        )

    # 3. Need total (region = DACH, process_stage = Security Sign-Off) == 160
    current_dach_sec = sum(
        1
        for r in rows
        if r["region"] == "DACH" and r["process_stage"] == "Security Sign-Off"
    )
    needed_dach_sec = 160 - current_dach_sec
    other_vendors = ["Internal FTE", "Tier-1 Strategic Partner"]

    for idx in range(needed_dach_sec):
        rows.append(
            {
                "region": "DACH",
