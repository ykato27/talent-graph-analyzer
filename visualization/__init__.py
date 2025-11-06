"""
可視化モジュール

このパッケージには以下のモジュールが含まれます：
- love_plot: Love plot（共変量バランスの可視化）
- calibration_plot: キャリブレーションプロット
- did_plots: DID分析の可視化
"""

from .love_plot import (
    plot_love,
    plot_smd_comparison
)

from .calibration_plot import (
    plot_calibration,
    plot_reliability_diagram
)

from .did_plots import (
    plot_parallel_trends,
    plot_treatment_effect_over_time,
    plot_did_coefficients
)

__all__ = [
    # Love Plot
    'plot_love',
    'plot_smd_comparison',

    # Calibration Plot
    'plot_calibration',
    'plot_reliability_diagram',

    # DID Plots
    'plot_parallel_trends',
    'plot_treatment_effect_over_time',
    'plot_did_coefficients',
]
