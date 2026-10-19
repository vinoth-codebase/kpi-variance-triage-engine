"""SQLite Star-Schema warehouse connection, schema initialization, and query utilities."""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from src.config import DB_PATH, DIMENSIONS


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create and return a SQLite connection with foreign keys enabled."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_star_schema(conn: sqlite3.Connection) -> None:
    """Initialize the Star-Schema dimension tables, fact table, indexes, and analytical view."""
    cursor = conn.cursor()

    cursor.executescript(
        """
        DROP VIEW IF EXISTS vw_star_workflow_executions;
        DROP TABLE IF EXISTS fact_workflow_executions;
        DROP TABLE IF EXISTS dim_region;
        DROP TABLE IF EXISTS dim_delivery_tower;
        DROP TABLE IF EXISTS dim_process_stage;
        DROP TABLE IF EXISTS dim_shift_tier;
        DROP TABLE IF EXISTS dim_system_environment;
        DROP TABLE IF EXISTS dim_vendor_tier;

        CREATE TABLE dim_region (
            region_id INTEGER PRIMARY KEY,
            region_name TEXT NOT NULL UNIQUE,
            theater_group TEXT NOT NULL,
            regional_director TEXT NOT NULL
        );

        CREATE TABLE dim_delivery_tower (
            tower_id INTEGER PRIMARY KEY,
            tower_name TEXT NOT NULL UNIQUE,
            domain_lead TEXT NOT NULL,
            criticality_class TEXT NOT NULL
        );

        CREATE TABLE dim_process_stage (
            stage_id INTEGER PRIMARY KEY,
            stage_name TEXT NOT NULL UNIQUE,
            sequence_order INTEGER NOT NULL,
            standard_sla_hours REAL NOT NULL
        );

        CREATE TABLE dim_shift_tier (
            shift_id INTEGER PRIMARY KEY,
            shift_name TEXT NOT NULL UNIQUE,
            operating_window_utc TEXT NOT NULL,
            staffing_model TEXT NOT NULL
        );

        CREATE TABLE dim_system_environment (
            env_id INTEGER PRIMARY KEY,
            env_name TEXT NOT NULL UNIQUE,
            cloud_provider TEXT NOT NULL,
            deployment_tier TEXT NOT NULL
        );

        CREATE TABLE dim_vendor_tier (
            vendor_id INTEGER PRIMARY KEY,
            vendor_name TEXT NOT NULL UNIQUE,
            contract_type TEXT NOT NULL,
            governance_regime TEXT NOT NULL
        );

        CREATE TABLE fact_workflow_executions (
            execution_id TEXT PRIMARY KEY,
            execution_date TEXT NOT NULL,
            day_index INTEGER NOT NULL,
            region_id INTEGER NOT NULL,
            tower_id INTEGER NOT NULL,
            stage_id INTEGER NOT NULL,
            shift_id INTEGER NOT NULL,
            env_id INTEGER NOT NULL,
            vendor_id INTEGER NOT NULL,
            region TEXT NOT NULL,
            delivery_tower TEXT NOT NULL,
            process_stage TEXT NOT NULL,
            shift_tier TEXT NOT NULL,
            system_environment TEXT NOT NULL,
            vendor_tier TEXT NOT NULL,
            sla_met INTEGER NOT NULL,
            sla_compliance_rate REAL NOT NULL,
            first_pass_yield INTEGER NOT NULL,
            first_pass_yield_rate REAL NOT NULL,
            mean_resolution_hours REAL NOT NULL,
            target_sla_hours REAL NOT NULL,
            defect_count INTEGER NOT NULL,
            retry_count INTEGER NOT NULL,
            error_code TEXT NOT NULL,
            pipeline_run_id TEXT NOT NULL,
            FOREIGN KEY (region_id) REFERENCES dim_region(region_id),
            FOREIGN KEY (tower_id) REFERENCES dim_delivery_tower(tower_id),
            FOREIGN KEY (stage_id) REFERENCES dim_process_stage(stage_id),
            FOREIGN KEY (shift_id) REFERENCES dim_shift_tier(shift_id),
            FOREIGN KEY (env_id) REFERENCES dim_system_environment(env_id),
            FOREIGN KEY (vendor_id) REFERENCES dim_vendor_tier(vendor_id)
        );

        CREATE INDEX idx_fact_exec_date ON fact_workflow_executions(execution_date);
        CREATE INDEX idx_fact_day_index ON fact_workflow_executions(day_index);
        CREATE INDEX idx_fact_tower_env ON fact_workflow_executions(delivery_tower, system_environment);
        CREATE INDEX idx_fact_stage_vendor ON fact_workflow_executions(process_stage, vendor_tier);
        CREATE INDEX idx_fact_region_stage ON fact_workflow_executions(region, process_stage);

        CREATE VIEW vw_star_workflow_executions AS
        SELECT
            f.execution_id,
            f.execution_date,
            f.day_index,
            f.region,
            r.theater_group,
            r.regional_director,
            f.delivery_tower,
            t.domain_lead,
            t.criticality_class,
            f.process_stage,
            s.sequence_order,
            f.shift_tier,
            sh.operating_window_utc,
            f.system_environment,
            e.cloud_provider,
            f.vendor_tier,
            v.contract_type,
            f.sla_met,
            f.sla_compliance_rate,
            f.first_pass_yield,
            f.first_pass_yield_rate,
            f.mean_resolution_hours,
            f.target_sla_hours,
            f.defect_count,
            f.retry_count,
            f.error_code,
            f.pipeline_run_id
        FROM fact_workflow_executions f
        JOIN dim_region r ON f.region_id = r.region_id
        JOIN dim_delivery_tower t ON f.tower_id = t.tower_id
        JOIN dim_process_stage s ON f.stage_id = s.stage_id
        JOIN dim_shift_tier sh ON f.shift_id = sh.shift_id
        JOIN dim_system_environment e ON f.env_id = e.env_id
        JOIN dim_vendor_tier v ON f.vendor_id = v.vendor_id;
        """
    )
    conn.commit()


def is_warehouse_populated(db_path: Path = DB_PATH) -> bool:
    """Check whether the SQLite warehouse exists and contains the 60-day dataset."""
    if not db_path.exists():
        return False
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_workflow_executions';"
            )
            if cursor.fetchone() is None:
                return False
            cursor.execute("SELECT COUNT(*), COUNT(DISTINCT execution_date) FROM fact_workflow_executions;")
            row = cursor.fetchone()
            if row is None:
                return False
            total_rows, distinct_days = row
            return total_rows >= 50000 and distinct_days == 60
    except sqlite3.Error:
        return False


def fetch_daily_kpi_series(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Aggregate daily KPI metrics across the 60-day warehouse horizon."""
    query = """
        SELECT
            execution_date,
            day_index,
            COUNT(*) AS total_executions,
            AVG(sla_compliance_rate) AS sla_compliance_rate,
            AVG(first_pass_yield_rate) AS first_pass_yield_rate,
            AVG(mean_resolution_hours) AS mean_resolution_hours,
            SUM(CASE WHEN sla_met = 0 THEN 1 ELSE 0 END) AS sla_breach_count,
            SUM(CASE WHEN first_pass_yield = 0 THEN 1 ELSE 0 END) AS fpy_defect_count
        FROM fact_workflow_executions
        GROUP BY execution_date, day_index
        ORDER BY day_index ASC;
    """
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    return df


def fetch_records_for_dates(
    dates: List[str],
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
