"""
Love Plotモジュール

共変量バランスを可視化するLove plotを生成します。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def plot_love(
    balance_table: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple = (10, 6)
) -> None:
    """
    Love plot（共変量バランスの可視化）

    Parameters
    ----------
    balance_table : pd.DataFrame
        covariate_balance_table()の結果
    save_path : str, optional
        保存先パス
    figsize : Tuple
        図のサイズ
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    covariates = balance_table['Covariate'].values
    smd_before = balance_table['SMD_Before'].abs().values
    smd_after = balance_table['SMD_After'].abs().values

    y_pos = np.arange(len(covariates))

    ax.scatter(smd_before, y_pos, marker='o', s=100, label='Before Matching', alpha=0.7)
    ax.scatter(smd_after, y_pos, marker='s', s=100, label='After Matching', alpha=0.7)

    for i in range(len(covariates)):
        ax.plot([smd_before[i], smd_after[i]], [y_pos[i], y_pos[i]], 'k-', alpha=0.3)

    ax.axvline(x=0.1, color='red', linestyle='--', linewidth=2, label='SMD=0.1 threshold')
    ax.axvline(x=0.2, color='orange', linestyle='--', linewidth=2, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates)
    ax.set_xlabel('Absolute Standardized Mean Difference', fontsize=12, fontweight='bold')
    ax.set_title('Love Plot: Covariate Balance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Love plot saved to {save_path}")

    plt.close()


def plot_smd_comparison(balance_table: pd.DataFrame, save_path: Optional[str] = None):
    """SMDの改善を棒グラフで可視化"""
    fig, ax = plt.subplots(figsize=(10, 6))

    covariates = balance_table['Covariate'].values
    improvement = balance_table['SMD_Improvement_%'].values

    colors = ['green' if x > 0 else 'red' for x in improvement]
    ax.barh(covariates, improvement, color=colors, alpha=0.7)

    ax.set_xlabel('SMD Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Covariate Balance Improvement', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()
