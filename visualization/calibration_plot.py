"""
キャリブレーションプロットモジュール

予測確率の信頼性を可視化します。
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def plot_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
    figsize: Tuple = (8, 8)
) -> None:
    """
    キャリブレーションプロット

    Parameters
    ----------
    y_true : np.ndarray
        真のラベル
    y_proba : np.ndarray
        予測確率
    n_bins : int
        ビン数
    save_path : str, optional
        保存先パス
    figsize : Tuple
        図のサイズ
    """
    from sklearn.calibration import calibration_curve

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins
    )

    ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

    ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
    ax.set_title('Calibration Plot', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration plot saved to {save_path}")

    plt.close()


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None
):
    """信頼性ダイアグラム（ヒストグラム付き）"""
    from sklearn.calibration import calibration_curve

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins
    )

    ax1.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax1.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
    ax1.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Histogram
    ax2.hist(y_proba, bins=n_bins, range=(0, 1), alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()
