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
                "delivery_tower": other_towers[idx % 3],
                "process_stage": "Security Sign-Off",
                "shift_tier": shifts[idx % 3],
                "system_environment": other_envs[idx % 3],
                "vendor_tier": other_vendors[idx % 2],
            }
        )

    # 4. Fill remaining records up to n_records avoiding the 3 special 2D cells so their counts stay exact
    remaining = n_records - len(rows)
    for idx in range(remaining):
        # Cycle cleanly across combinations that do not hit the 3 locked 2D cells
        t_val = towers[idx % 4]
        e_val = envs[(idx // 4) % 4]
        if t_val == "Core Data Engineering" and e_val == "Snowflake-Prod-EU":
            e_val = "Snowflake-Prod-US"

        s_val = stages[(idx // 2) % 4]
        v_val = vendors[idx % 3]
        if s_val == "UAT Validation" and v_val == "Tier-2 Contractor":
            v_val = "Tier-1 Strategic Partner"

        r_val = regions[(idx // 3) % 4]
        if r_val == "DACH" and s_val == "Security Sign-Off":
            r_val = "UK-Ireland"

        sh_val = shifts[idx % 3]
        rows.append(
            {
                "region": r_val,
                "delivery_tower": t_val,
                "process_stage": s_val,
                "shift_tier": sh_val,
                "system_environment": e_val,
                "vendor_tier": v_val,
            }
        )

    df_template = pd.DataFrame(rows)
    # Shuffle row order deterministically
    perm = rng.permutation(len(df_template))
    return df_template.iloc[perm].reset_index(drop=True)


def generate_warehouse_data(db_path: Path = DB_PATH, force_rebuild: bool = False) -> None:
    """Populate the SQLite Star-Schema warehouse with 60 days of operational executions."""
    if not force_rebuild and is_warehouse_populated(db_path):
        return

    rng = np.random.default_rng(RANDOM_SEED)
    base_template = _build_base_daily_template(DAILY_EXECUTION_VOLUME)

    # Lookup dictionaries for dimension foreign keys
    region_map = {name: rid for rid, name, _, _ in DIMENSION_METADATA["dim_region"]}
    tower_map = {name: tid for tid, name, _, _ in DIMENSION_METADATA["dim_delivery_tower"]}
    stage_map = {name: sid for sid, name, _, _ in DIMENSION_METADATA["dim_process_stage"]}
    stage_sla_map = {name: sla for _, name, _, sla in DIMENSION_METADATA["dim_process_stage"]}
    shift_map = {name: shid for shid, name, _, _ in DIMENSION_METADATA["dim_shift_tier"]}
    env_map = {name: eid for eid, name, _, _ in DIMENSION_METADATA["dim_system_environment"]}
    vendor_map = {name: vid for vid, name, _, _ in DIMENSION_METADATA["dim_vendor_tier"]}

    # End date fixed to 2026-10-30 (60 days starting 2026-09-01)
    end_date = date(2026, 10, 30)
    start_date = end_date - timedelta(days=SIMULATION_DAYS - 1)

    # Pre-compute daily target counts for normal days so 14-day baselines match specifications
    # Normal SLA counts out of 1000:
    # Days 1..45: average ~935 (93.5%), oscillating between 930 and 940
    # Days 46..59 (14d prior to Day 60): mean must be exactly 938.0 (93.8%)
    sla_offsets_14 = [-5, +3, -2, +4, -4, +2, +1, -3, +5, -1, +2, -4, +3, -1]  # sum = 0
    fpy_offsets_14 = [-4, +3, -2, +2, -3, +4, -1, +2, -3, +1, +3, -2, +1, -1]  # sum = 0
    res_offsets_14 = [
        -0.11, +0.08, -0.05, +0.09, -0.07, +0.06, -0.03,
        +0.04, -0.08, +0.07, -0.04, +0.05, -0.03, +0.02
    ]  # sum = 0.0

    all_fact_rows: List[Tuple[Any, ...]] = []

    for day_idx in range(1, SIMULATION_DAYS + 1):
        curr_date = start_date + timedelta(days=day_idx - 1)
        date_str = curr_date.isoformat()
        date_compact = curr_date.strftime("%Y%m%d")

        df_day = base_template.copy()

        # Adjust volumes slightly on anomaly days to create realistic Mix + Rate interaction
        if day_idx == 60:
            # Add 10 extra executions into (Core Data Engineering, Snowflake-Prod-EU, Shift-3 (Night))
            non_target_mask = ~(
                (df_day["delivery_tower"] == "Core Data Engineering")
                & (df_day["system_environment"] == "Snowflake-Prod-EU")
            )
            convert_indices = df_day[non_target_mask].index[:10]
            df_day.loc[convert_indices, "delivery_tower"] = "Core Data Engineering"
            df_day.loc[convert_indices, "system_environment"] = "Snowflake-Prod-EU"
            df_day.loc[convert_indices, "shift_tier"] = "Shift-3 (Night)"

        elif day_idx == 52:
            # Add 10 extra executions into (UAT Validation, Tier-2 Contractor)
            non_target_mask = ~(
                (df_day["process_stage"] == "UAT Validation")
                & (df_day["vendor_tier"] == "Tier-2 Contractor")
            )
            convert_indices = df_day[non_target_mask].index[:10]
            df_day.loc[convert_indices, "process_stage"] = "UAT Validation"
            df_day.loc[convert_indices, "vendor_tier"] = "Tier-2 Contractor"

        elif day_idx == 45:
            # Add 10 extra executions into (DACH, Security Sign-Off)
            non_target_mask = ~(
                (df_day["region"] == "DACH")
                & (df_day["process_stage"] == "Security Sign-Off")
            )
            convert_indices = df_day[non_target_mask].index[:10]
            df_day.loc[convert_indices, "region"] = "DACH"
            df_day.loc[convert_indices, "process_stage"] = "Security Sign-Off"

        n_day = len(df_day)

        # ------------------------------------------------------------------
        # 1. SLA COMPLIANCE RATE GENERATION
        # ------------------------------------------------------------------
        sla_met_arr = np.ones(n_day, dtype=int)
        if day_idx == 60:
            # Day 60 Severe SLA Drop: 93.8% -> 81.4% (814 pass, 186 fail)
            # Primary slice: (Core Data Engineering, Snowflake-Prod-EU) has 250 records
            # Set 119 failures in primary slice (131 pass -> 52.4% SLA compliance)
            # Inside primary slice, concentrate 75 failures in Shift-3 (Night)
            primary_mask = (
                (df_day["delivery_tower"] == "Core Data Engineering")
                & (df_day["system_environment"] == "Snowflake-Prod-EU")
            )
            primary_shift3_idx = df_day[
                primary_mask & (df_day["shift_tier"] == "Shift-3 (Night)")
            ].index
            primary_other_idx = df_day[
                primary_mask & (df_day["shift_tier"] != "Shift-3 (Night)")
            ].index
            non_primary_idx = df_day[~primary_mask].index

            fail_shift3 = primary_shift3_idx[:75]
            fail_other_primary = primary_other_idx[:44]  # 75 + 44 = 119 failures in primary slice
            fail_non_primary = non_primary_idx[:67]  # 119 + 67 = 186 total failures -> 814 pass (81.4%)

            sla_met_arr[fail_shift3] = 0
            sla_met_arr[fail_other_primary] = 0
            sla_met_arr[fail_non_primary] = 0
        else:
            # Normal day SLA compliance
            if 46 <= day_idx <= 59:
                total_sla_pass = 938 + sla_offsets_14[day_idx - 46]
                primary_fails = 14  # 226 / 240 = 94.17% baseline in (Core Data Eng, Snowflake-Prod-EU)
            else:
                wave = int(np.round(5.0 * np.sin(day_idx * 0.65)))
                total_sla_pass = 935 + wave
                primary_fails = 15

            total_sla_fails = n_day - total_sla_pass
            primary_mask = (
                (df_day["delivery_tower"] == "Core Data Engineering")
                & (df_day["system_environment"] == "Snowflake-Prod-EU")
            )
            primary_idx = rng.permutation(df_day[primary_mask].index)
            non_primary_idx = rng.permutation(df_day[~primary_mask].index)

            sla_met_arr[primary_idx[:primary_fails]] = 0
            sla_met_arr[non_primary_idx[: max(0, total_sla_fails - primary_fails)]] = 0

        # ------------------------------------------------------------------
        # 2. FIRST PASS YIELD RATE GENERATION
        # ------------------------------------------------------------------
        fpy_arr = np.ones(n_day, dtype=int)
        if day_idx == 52:
            # Day 52 Defect Leakage Spike: 95.1% -> 84.6% (846 pass, 154 fail)
            # Primary slice: (UAT Validation, Tier-2 Contractor) has 190 records
