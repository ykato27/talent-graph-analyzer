"""
因果推論モジュール

このパッケージには以下のモジュールが含まれます：
- sensitivity_analysis: 感度分析（Rosenbaum bounds, E-value）
- cluster_robust: クラスター頑健標準誤差
- psm_diagnostics: 傾向スコアマッチングの診断指標
- did_analysis: 差分の差分法（Difference-in-Differences）
- causal_forest: Causal Forest による異質効果推定
"""

from .sensitivity_analysis import (
    rosenbaum_bounds,
    calculate_e_value,
    sensitivity_analysis_report
)

from .cluster_robust import (
    cluster_robust_se,
    cluster_robust_inference
)

from .psm_diagnostics import (
    calculate_smd,
    covariate_balance_table,
    check_overlap
)

from .did_analysis import (
    did_estimation,
    parallel_trends_test,
    did_with_covariates
)

from .causal_forest import (
    fit_causal_forest,
    get_heterogeneous_effects,
    identify_subgroups
)

__all__ = [
    # Sensitivity Analysis
    'rosenbaum_bounds',
    'calculate_e_value',
    'sensitivity_analysis_report',

    # Cluster Robust
    'cluster_robust_se',
    'cluster_robust_inference',

    # PSM Diagnostics
    'calculate_smd',
    'covariate_balance_table',
    'check_overlap',

    # DID Analysis
    'did_estimation',
    'parallel_trends_test',
    'did_with_covariates',

    # Causal Forest
    'fit_causal_forest',
    'get_heterogeneous_effects',
    'identify_subgroups',
]
