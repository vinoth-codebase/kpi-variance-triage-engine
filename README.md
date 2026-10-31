# Automated KPI Variance & Root-Cause Triage Engine

An enterprise-grade operational analytics and automated root-cause diagnostic workbench built in Python, SQLite, SciPy, scikit-learn, Plotly, and Streamlit.

When an operational or executive KPI suddenly degrades, analysts typically spend hours writing manual, repetitive SQL `GROUP BY` queries across dozens of dimensional combinations to isolate the underlying driver. The **Automated KPI Variance & Root-Cause Triage Engine** eliminates manual slicing by combining statistical time-series surveillance, exact mathematical **Mix vs. Rate variance decomposition** (Berman/Kitagawa identity), shallow decision-tree micro-cluster extraction, and one-click Excel audit reporting.

---

## 1. Architectural Capabilities

1. **Statistical Time-Series Surveillance (`src/anomaly_detector.py`):**
   - Evaluates 60 days of daily operational telemetry (`60,000` workflow executions across a 6-dimension Star Schema).
   - Computes 14-day rolling means, rolling standard deviations, Bollinger-style upper and lower control bands, Z-scores, two-tailed normal P-values, and Interquartile Range (IQR) bounds.
   - Supports three comparison baseline modes: **Prior 14-Day Baseline**, **Prior 7-Day Baseline**, and **Prior Day (DoD)**.

2. **Combinatorial Mix vs. Rate Variance Attribution (`src/attribution_engine.py`):**
   - Evaluates all single dimensions (`1D`) and all two-way dimensional interactions (`2D` via `itertools.combinations`).
   - Decomposes every segment's contribution into:
     - **Rate Effect (Within-Segment Performance Change):**
       $$\text{Rate Effect}_i = W_{i,0} \times (M_{i,1} - M_{i,0})$$
     - **Mix Effect (Volume Shift Across Segments):**
       $$\text{Mix Effect}_i = (W_{i,1} - W_{i,0}) \times (M_{i,1} - \bar{M}_0)$$
     - **Total Variance Contribution ($\Delta C_i$):**
       $$\Delta C_i = \text{Rate Effect}_i + \text{Mix Effect}_i, \quad \sum_i \Delta C_i = \bar{M}_1 - \bar{M}_0$$
   - Executes Welch's two-sample t-test (`scipy.stats.ttest_ind`) per slice to filter out small-sample noise and rank statistically significant root causes.
   - Trains a shallow `DecisionTreeRegressor` (`max_depth=3`) on target-day records to isolate human-readable decision path rules for the highest-density failure pockets.

3. **Star-Schema Operational Warehouse & Surgical Anomalies (`src/database.py`, `src/data_generator.py`):**
   - Populates `db/kpi_triage_warehouse.db` with `fact_workflow_executions`, 6 dimension tables (`dim_region`, `dim_delivery_tower`, `dim_process_stage`, `dim_shift_tier`, `dim_system_environment`, `dim_vendor_tier`), and an analytical view `vw_star_workflow_executions`.
   - Includes three calibrated operational root-cause anomalies:
     - **Day 60 (`2026-10-30`) — Severe SLA Compliance Drop (`93.8%` to `81.4%`):** Driven `84.2%` by `delivery_tower = "Core Data Engineering"` and `system_environment = "Snowflake-Prod-EU"`, concentrated in `shift_tier = "Shift-3 (Night)"` (`ERR_SF_TIMEOUT_504`).
     - **Day 52 (`2026-10-22`) — Defect Leakage Spike (`95.1%` to `84.6%` First Pass Yield):** Driven `79.0%` by `process_stage = "UAT Validation"` and `vendor_tier = "Tier-2 Contractor"` (`ERR_UAT_SCHEMA_MISMATCH`).
     - **Day 45 (`2026-10-15`) — Turnaround Time Surge (`4.20h` to `7.90h` Mean Resolution Hours):** Driven `81.3%` by `region = "DACH"` and `process_stage = "Security Sign-Off"` (`ERR_SEC_POLICY_HOLD`).

4. **Executive Audit Pack Exporter (`src/exporter.py`):**
   - Generates a formatted 4-sheet Excel workbook (`Executive_Summary`, `1D_Dimension_Breakdown`, `2D_Interaction_Ranking`, `Anomalous_Raw_Records`) in `data/exports/`.

---

## 2. Repository Structure

```text
kpi-variance-triage-engine/
├── .gitignore
├── README.md
├── requirements.txt
├── run_app.sh
├── data/
│   └── exports/
├── db/
│   └── kpi_triage_warehouse.db
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── database.py
│   ├── data_generator.py
│   ├── anomaly_detector.py
│   ├── attribution_engine.py
│   └── exporter.py
├── app.py
└── tests/
    └── test_attribution.py
```

---
