"""
モデル評価モジュール

このパッケージには以下のモジュールが含まれます：
- baseline_models: ベースラインモデル（LR, RF, XGBoost）
- cross_validation: クロスバリデーションフレームワーク
- performance_metrics: 評価指標の計算
"""

from .baseline_models import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
    compare_models
)

from .cross_validation import (
    stratified_cv,
    leave_one_out_cv,
    nested_cv,
    cv_performance_summary
)

from .performance_metrics import (
    calculate_all_metrics,
    calibration_curve,
    roc_analysis,
    confusion_matrix_analysis
)

__all__ = [
    # Baseline Models
    'LogisticRegressionBaseline',
    'RandomForestBaseline',
    'XGBoostBaseline',
    'compare_models',

    # Cross Validation
    'stratified_cv',
    'leave_one_out_cv',
    'nested_cv',
    'cv_performance_summary',

    # Performance Metrics
    'calculate_all_metrics',
    'calibration_curve',
    'roc_analysis',
    'confusion_matrix_analysis',
]
