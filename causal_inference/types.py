"""
型定義モジュール

因果推論モジュールで使用する型を定義します。
"""

from typing import TypedDict, Protocol, Literal
import numpy as np
import pandas as pd


class RosenbaumBoundsResult(TypedDict):
    """Rosenbaum Bounds の結果型"""
    Gamma: float
    P_value_upper: float
    P_value_lower: float
    Significant_at_0_05_upper: bool
    Significant_at_0_05_lower: bool


class EValueResult(TypedDict):
    """E-value の結果型"""
    point_estimate: float
    ci_lower: float | None
    interpretation: str


class SensitivityAnalysisReport(TypedDict):
    """感度分析レポートの型"""
    rosenbaum_bounds: pd.DataFrame
    e_value: EValueResult
    summary: str
    recommendation: str


class ClusterRobustResult(TypedDict):
    """クラスター頑健標準誤差の結果型"""
    coefficients: np.ndarray
    se_regular: np.ndarray
    se_cluster: np.ndarray
    t_statistics: np.ndarray
    p_values: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    n_clusters: int
    n_observations: int
    df: int
    icc: float
    design_effect: np.ndarray
    vcov_cluster: np.ndarray


class CovariateBalanceRow(TypedDict):
    """共変量バランステーブルの行型"""
    Covariate: str
    Mean_Treated_Before: float
    Mean_Control_Before: float
    SMD_Before: float
    Balance_Before: str
    Mean_Treated_After: float
    Mean_Control_After: float
    SMD_After: float
    Balance_After: str
    SMD_Improvement_Percent: float


class OverlapResult(TypedDict):
    """傾向スコアの重なり評価結果型"""
    overlap_range: tuple[float, float]
    n_treated_in_range: int
    n_control_in_range: int
    percentage_treated: float
    percentage_control: float
    recommendation: str
    range_width: float


class PSMQualityReport(TypedDict):
    """PSM品質レポートの型"""
    overall_quality: Literal["Excellent", "Good", "Acceptable", "Poor"]
    overall_score: float
    balance_score: float
    overlap_score: float
    matching_rate_treated: float
    matching_rate_control: float
    summary: str
    recommendations: list[str]


class DIDResult(TypedDict):
    """DID推定結果の型"""
    did_estimate: float
    did_coefficient: float
    se: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_treated: int
    n_control: int
    n_observations: int
    means: dict[str, float]
    parallel_trends_test: dict
    model_summary: object


class ParallelTrendsTestResult(TypedDict):
    """平行トレンド検定の結果型"""
    test_statistic: float
    p_value: float
    result: Literal["Pass", "Fail", "Inconclusive", "Error"]
    interpretation: str


# Protocol for models
class PredictiveModel(Protocol):
    """予測モデルのプロトコル"""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """モデルを学習"""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測を実行"""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """確率予測を実行"""
        ...


EffectType = Literal["odds_ratio", "risk_ratio", "mean_difference"]
OverlapMethod = Literal["minmax", "percentile"]
ClusteringMethod = Literal["kmeans", "hierarchical"]
CVMethod = Literal["stratified", "loocv", "nested"]
