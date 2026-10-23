"""Mathematical Root-Cause Attribution, Mix/Rate Decomposition, and Decision Tree Rule Extraction."""

from itertools import combinations
from typing import Dict, List, Optional, Tuple, Any
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    DIMENSIONS,
    DIMENSION_LABELS,
    SUPPORTED_KPIS,
    SIGNIFICANCE_ALPHA,
    MIN_SLICE_VOLUME,
    RANDOM_SEED,
)


class VarianceAttributionEngine:
    """
    Perform 1D and 2D combinatorial Mix vs. Rate variance decomposition,
    hypothesis testing, decision-tree failure cluster extraction, and narrative synthesis.
    """

    def __init__(
        self,
        dimensions: Optional[List[str]] = None,
        alpha: float = SIGNIFICANCE_ALPHA,
        min_slice_volume: int = MIN_SLICE_VOLUME,
    ) -> None:
        self.dimensions = dimensions if dimensions is not None else list(DIMENSIONS)
        self.alpha = float(alpha)
        self.min_slice_volume = int(min_slice_volume)

    def _compute_slice_attribution(
        self,
        baseline_df: pd.DataFrame,
        target_df: pd.DataFrame,
        group_cols: List[str],
        kpi_key: str,
        baseline_overall: float,
        target_overall: float,
    ) -> pd.DataFrame:
        """
        Compute exact Mix vs. Rate decomposition across all segments of group_cols.

        Mathematical Identity:
            Rate Effect_i = W_{i,0} * (M_{i,1} - M_{i,0})
            Mix Effect_i  = (W_{i,1} - W_{i,0}) * (M_{i,1} - M_0_bar)
            Total Contribution_i = Rate Effect_i + Mix Effect_i
            Sum_i (Total Contribution_i) == M_1_bar - M_0_bar
        """
        total_delta = target_overall - baseline_overall
        n_base_total = max(len(baseline_df), 1)
        n_targ_total = max(len(target_df), 1)

        base_grouped = (
            baseline_df.groupby(group_cols, dropna=False)[kpi_key]
            .agg(baseline_volume="count", baseline_metric="mean", baseline_std="std")
            .reset_index()
        )
        targ_grouped = (
            target_df.groupby(group_cols, dropna=False)[kpi_key]
            .agg(target_volume="count", target_metric="mean", target_std="std")
            .reset_index()
        )

        merged = pd.merge(base_grouped, targ_grouped, on=group_cols, how="outer")
        merged["baseline_volume"] = merged["baseline_volume"].fillna(0).astype(int)
        merged["target_volume"] = merged["target_volume"].fillna(0).astype(int)
        merged["baseline_std"] = merged["baseline_std"].fillna(0.0).astype(float)
        merged["target_std"] = merged["target_std"].fillna(0.0).astype(float)

        # If a segment is empty in one window, inherit the other window's mean so algebra holds
        merged["baseline_metric"] = merged["baseline_metric"].fillna(merged["target_metric"]).fillna(baseline_overall)
        merged["target_metric"] = merged["target_metric"].fillna(merged["baseline_metric"]).fillna(target_overall)

        merged["baseline_share"] = merged["baseline_volume"] / float(n_base_total)
        merged["target_share"] = merged["target_volume"] / float(n_targ_total)

        merged["metric_delta"] = merged["target_metric"] - merged["baseline_metric"]
        merged["share_delta"] = merged["target_share"] - merged["baseline_share"]

        # Exact Berman/Kitagawa Rate vs Mix decomposition
        merged["rate_effect"] = merged["baseline_share"] * (
            merged["target_metric"] - merged["baseline_metric"]
        )
        merged["mix_effect"] = (merged["target_share"] - merged["baseline_share"]) * (
            merged["target_metric"] - baseline_overall
        )
        merged["total_contribution"] = merged["rate_effect"] + merged["mix_effect"]

        if abs(total_delta) > 1e-9:
            merged["variance_explained_pct"] = (
                merged["total_contribution"] / total_delta
            ) * 100.0
        else:
            merged["variance_explained_pct"] = 0.0

        # Share of absolute effect within slice attributable to Rate vs Mix
        abs_rate = merged["rate_effect"].abs()
        abs_mix = merged["mix_effect"].abs()
        denom_abs = (abs_rate + abs_mix).replace(0.0, 1.0)
        merged["rate_share_of_slice_pct"] = np.where(
            (abs_rate + abs_mix) > 1e-12, (abs_rate / denom_abs) * 100.0, 100.0
        )
        merged["mix_share_of_slice_pct"] = np.where(
            (abs_rate + abs_mix) > 1e-12, (abs_mix / denom_abs) * 100.0, 0.0
        )

        # Statistical significance (Welch's two-sample t-test per slice)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t_res = stats.ttest_ind_from_stats(
                mean1=merged["target_metric"].to_numpy(dtype=float),
                std1=merged["target_std"].to_numpy(dtype=float),
                nobs1=np.maximum(merged["target_volume"].to_numpy(dtype=float), 2.0),
                mean2=merged["baseline_metric"].to_numpy(dtype=float),
