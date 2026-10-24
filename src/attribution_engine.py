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
                std2=merged["baseline_std"].to_numpy(dtype=float),
                nobs2=np.maximum(merged["baseline_volume"].to_numpy(dtype=float), 2.0),
                equal_var=False,
            )

        valid_mask = (merged["baseline_volume"] >= 2) & (merged["target_volume"] >= 2)
        nonzero_std_mask = (merged["baseline_std"] > 1e-12) | (merged["target_std"] > 1e-12)

        t_stat_arr = np.where(
            valid_mask & nonzero_std_mask,
            np.nan_to_num(t_res.statistic, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
        )
        p_val_arr = np.where(
            valid_mask & nonzero_std_mask,
            np.nan_to_num(t_res.pvalue, nan=1.0, posinf=1.0, neginf=1.0),
            1.0,
        )

        merged["t_statistic"] = t_stat_arr
        merged["p_value"] = p_val_arr
        merged["is_significant"] = (merged["p_value"] < self.alpha) & (
            merged["target_volume"] >= self.min_slice_volume
        )

        # Metadata columns for unified leaderboard ranking
        dim_type = "1D" if len(group_cols) == 1 else "2D"
        dim_key = " | ".join(group_cols)
        dim_label = " x ".join(DIMENSION_LABELS.get(c, c) for c in group_cols)

        slice_labels: List[str] = []
        slice_short_labels: List[str] = []
        slice_filters: List[Dict[str, str]] = []

        for _, row in merged.iterrows():
            parts = [f"{c} = {row[c]}" for c in group_cols]
            short_parts = [str(row[c]) for c in group_cols]
            slice_labels.append(" | ".join(parts))
            slice_short_labels.append(" + ".join(short_parts))
            slice_filters.append({c: str(row[c]) for c in group_cols})

        merged["dimension_type"] = dim_type
        merged["dimension_key"] = dim_key
        merged["dimension_label"] = dim_label
        merged["slice_label"] = slice_labels
        merged["slice_short_label"] = slice_short_labels
        merged["slice_filter"] = slice_filters

        return merged

    def extract_decision_tree_rules(
        self,
        target_df: pd.DataFrame,
        kpi_key: str,
        max_depth: int = 3,
        min_samples_leaf: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Train a shallow scikit-learn DecisionTreeRegressor on the target day's records
        to extract human-readable decision path rules for the highest-density failure clusters.
        """
        if target_df.empty:
            return []

        kpi_meta = SUPPORTED_KPIS[kpi_key]
        higher_is_better = bool(kpi_meta["higher_is_better"])
        overall_mean = float(target_df[kpi_key].mean())

        X_df = pd.get_dummies(target_df[self.dimensions], prefix_sep=" == ", dtype=float)
        feature_names = list(X_df.columns)
        y = target_df[kpi_key].to_numpy(dtype=float)

        tree_model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_SEED,
        )
        tree_model.fit(X_df, y)

        tree_ = tree_model.tree_
        leaf_rules: List[Dict[str, Any]] = []

        def _recurse(node_id: int, path_conditions: List[str]) -> None:
            left_child = int(tree_.children_left[node_id])
            right_child = int(tree_.children_right[node_id])

            if left_child == right_child:  # Leaf node
                leaf_val = float(tree_.value[node_id][0][0])
                leaf_count = int(tree_.n_node_samples[node_id])
                deviation = leaf_val - overall_mean
                adverse_score = -deviation if higher_is_better else deviation

                # Simplify path conditions: prioritize equality conditions if present
                eq_conds = [c for c in path_conditions if " == " in c]
                neq_conds = [c for c in path_conditions if " != " in c]
                ordered_conds = eq_conds + neq_conds
                rule_str = " AND ".join(ordered_conds) if ordered_conds else "All Executions"

                leaf_rules.append(
                    {
                        "rule_path": rule_str,
                        "equality_conditions": eq_conds,
                        "sample_count": leaf_count,
                        "sample_share_pct": (leaf_count / max(len(target_df), 1)) * 100.0,
                        "leaf_metric": leaf_val,
                        "overall_metric": overall_mean,
                        "metric_deviation": deviation,
                        "adverse_score": adverse_score,
                    }
                )
                return

            feat_idx = int(tree_.feature[node_id])
            feat_name = feature_names[feat_idx]
            # Left child corresponds to feature <= 0.5 (not equal)
            _recurse(left_child, path_conditions + [feat_name.replace(" == ", " != ")])
            # Right child corresponds to feature > 0.5 (equal)
            _recurse(right_child, path_conditions + [feat_name])

        _recurse(0, [])
        leaf_rules.sort(key=lambda r: r["adverse_score"], reverse=True)
        return leaf_rules

    def generate_executive_narrative(
        self,
        kpi_key: str,
        baseline_overall: float,
        target_overall: float,
        z_score: float,
        p_value: float,
        top_driver_row: pd.Series,
        tree_rules: List[Dict[str, Any]],
    ) -> str:
        """Synthesize a plain-English consulting diagnostic brief."""
        kpi_meta = SUPPORTED_KPIS[kpi_key]
        kpi_label = str(kpi_meta["label"])
        unit = str(kpi_meta["value_suffix"])
        decimals = int(kpi_meta["decimals"])
        higher_is_better = bool(kpi_meta["higher_is_better"])

        total_delta = target_overall - baseline_overall
        direction_word = "decreased" if total_delta < 0 else "increased"
        abs_delta = abs(total_delta)

        if unit == "%":
