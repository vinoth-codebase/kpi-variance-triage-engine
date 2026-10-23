"""Statistical Time-Series Surveillance and Anomaly Detection Engine."""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    Z_SCORE_THRESHOLD,
    HIGH_SEVERITY_Z_THRESHOLD,
    ROLLING_WINDOW_DAYS,
    PRIOR_7D_WINDOW_DAYS,
    SUPPORTED_KPIS,
)


class KPIAnomalyDetector:
    """Detect statistical anomalies across daily operational KPI time series."""

    def __init__(
        self,
        z_threshold: float = Z_SCORE_THRESHOLD,
        high_severity_threshold: float = HIGH_SEVERITY_Z_THRESHOLD,
        rolling_window: int = ROLLING_WINDOW_DAYS,
    ) -> None:
        self.z_threshold = float(z_threshold)
        self.high_severity_threshold = max(float(high_severity_threshold), self.z_threshold)
        self.rolling_window = int(rolling_window)

    def analyze_series(
        self,
        daily_df: pd.DataFrame,
        kpi_key: str,
    ) -> pd.DataFrame:
        """
        Compute rolling control statistics, Z-scores, IQR bounds, and baseline windows.

        Parameters
        ----------
        daily_df : pd.DataFrame
            Daily aggregated warehouse metrics ordered by day_index.
        kpi_key : str
            Target KPI column name (e.g., 'sla_compliance_rate').

        Returns
        -------
        pd.DataFrame
            Enriched surveillance DataFrame with statistical bounds and anomaly flags.
        """
        if kpi_key not in SUPPORTED_KPIS:
            raise ValueError(f"Unsupported KPI key: {kpi_key}")
        if kpi_key not in daily_df.columns:
            raise KeyError(f"Column '{kpi_key}' not present in daily DataFrame.")

        df = daily_df.sort_values("day_index").reset_index(drop=True).copy()
        series = df[kpi_key].astype(float)

        # Rolling window statistics (14-day window)
        df["kpi_value"] = series
        df["rolling_mean"] = series.rolling(
            window=self.rolling_window, min_periods=1
        ).mean()
        rolling_std = (
            series.rolling(window=self.rolling_window, min_periods=2)
            .std(ddof=1)
            .bfill()
            .fillna(1e-6)
        )
        df["rolling_std"] = rolling_std.replace(0.0, 1e-6)

        # Baseline windows: Prior Day (DoD), Prior 7-Day Moving Average, Prior 14-Day Baseline
        df["prior_day_value"] = series.shift(1).bfill()
        df["prior_7d_mean"] = (
            series.shift(1)
            .rolling(window=PRIOR_7D_WINDOW_DAYS, min_periods=1)
            .mean()
            .bfill()
        )
        df["prior_14d_mean"] = (
            series.shift(1)
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .bfill()
        )

        # Deltas and relative changes across all 3 baseline windows
        df["dod_delta"] = df["kpi_value"] - df["prior_day_value"]
        df["dod_pct_change"] = np.where(
            df["prior_day_value"] != 0,
            (df["dod_delta"] / df["prior_day_value"]) * 100.0,
            0.0,
        )

        df["prior_7d_delta"] = df["kpi_value"] - df["prior_7d_mean"]
        df["prior_7d_pct_change"] = np.where(
            df["prior_7d_mean"] != 0,
            (df["prior_7d_delta"] / df["prior_7d_mean"]) * 100.0,
            0.0,
        )

        df["prior_14d_delta"] = df["kpi_value"] - df["prior_14d_mean"]
        df["prior_14d_pct_change"] = np.where(
            df["prior_14d_mean"] != 0,
            (df["prior_14d_delta"] / df["prior_14d_mean"]) * 100.0,
            0.0,
        )

        # Statistical Z-score comparing daily observation against prior 14-day baseline level
        # scaled by the 14-day rolling standard deviation
        df["z_score"] = (df["kpi_value"] - df["prior_14d_mean"]) / df["rolling_std"]
        df["abs_z_score"] = df["z_score"].abs()
        df["p_value"] = stats.norm.sf(df["abs_z_score"]) * 2.0

        # Bollinger-style upper and lower control bands around the prior 14-day baseline mean
        df["upper_band"] = df["prior_14d_mean"] + (self.z_threshold * df["rolling_std"])
        df["lower_band"] = df["prior_14d_mean"] - (self.z_threshold * df["rolling_std"])

        # Interquartile Range (IQR) bounds across the 60-day surveillance horizon
        q25 = float(np.percentile(series, 25))
        q75 = float(np.percentile(series, 75))
        iqr = max(q75 - q25, 1e-6)
        df["iqr_lower"] = q25 - 1.5 * iqr
        df["iqr_upper"] = q75 + 1.5 * iqr
        df["is_iqr_outlier"] = (df["kpi_value"] < df["iqr_lower"]) | (
            df["kpi_value"] > df["iqr_upper"]
        )

        # Anomaly classification and severity grading
        df["is_anomaly"] = df["abs_z_score"] >= self.z_threshold
        df["anomaly_flag"] = np.where(
            df["is_anomaly"], "ANOMALY_DETECTED", "NORMAL"
        )
        df["severity"] = np.select(
            [
                df["abs_z_score"] >= self.high_severity_threshold,
                df["abs_z_score"] >= self.z_threshold,
            ],
            [
                "SEVERITY_HIGH",
                "SEVERITY_MODERATE",
            ],
            default="NORMAL",
        )
        df["status_badge"] = np.select(
            [
                df["abs_z_score"] >= self.high_severity_threshold,
                df["abs_z_score"] >= self.z_threshold,
            ],
            [
                "CRITICAL ANOMALY",
                "MODERATE ANOMALY",
            ],
            default="WITHIN CONTROL BOUNDS",
        )

        return df

    @staticmethod
    def get_baseline_dates(
        surveillance_df: pd.DataFrame,
        target_date: str,
        baseline_mode: str = "Prior 14-Day Baseline",
    ) -> List[str]:
        """Return the list of execution dates composing the comparison baseline window."""
        df = surveillance_df.sort_values("day_index").reset_index(drop=True)
        match = df[df["execution_date"] == target_date]
        if match.empty:
            raise ValueError(f"Target date '{target_date}' not found in surveillance series.")

        target_pos = int(match.index[0])
        if target_pos == 0:
            # Edge case if Day 1 is selected: fallback to first available normal days
            return df.iloc[1 : min(15, len(df))]["execution_date"].tolist()

        if baseline_mode == "Prior Day (DoD)":
            window_size = 1
        elif baseline_mode == "Prior 7-Day Baseline":
            window_size = PRIOR_7D_WINDOW_DAYS
