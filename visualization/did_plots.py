"""
DID分析の可視化モジュール

差分の差分法の結果を可視化します。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def plot_parallel_trends(
    df: pd.DataFrame,
    outcome_col: str,
    time_col: str,
    group_col: str,
    save_path: Optional[str] = None,
    figsize: Tuple = (10, 6)
) -> None:
    """
    平行トレンドのプロット

    Parameters
    ----------
    df : pd.DataFrame
        分析データ
    outcome_col : str
        アウトカム変数
    time_col : str
        時間変数
    group_col : str
        グループ変数（0=対照群, 1=処置群）
    save_path : str, optional
        保存先パス
    figsize : Tuple
        図のサイズ
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    # グループごとの時系列平均
    grouped = df.groupby([time_col, group_col])[outcome_col].mean().reset_index()

    for group_id in [0, 1]:
        group_data = grouped[grouped[group_col] == group_id]
        label = 'Treatment Group' if group_id == 1 else 'Control Group'
        ax.plot(group_data[time_col], group_data[outcome_col], 'o-', label=label, linewidth=2, markersize=8)

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Mean {outcome_col}', fontsize=12, fontweight='bold')
    ax.set_title('Parallel Trends Check', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Parallel trends plot saved to {save_path}")

    plt.close()


def plot_treatment_effect_over_time(
    did_results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple = (10, 6)
) -> None:
    """
    処置効果の時系列プロット

    Parameters
    ----------
    did_results : Dict
        did_estimation()の結果
    save_path : str, optional
        保存先パス
    figsize : Tuple
        図のサイズ
    """
    fig, ax = plt.subplots(figsize=figsize)

    means = did_results['means']
    periods = ['Pre', 'Post']

    treated = [means['treated_pre'], means['treated_post']]
    control = [means['control_pre'], means['control_post']]

    ax.plot(periods, treated, 'o-', label='Treatment Group', linewidth=2, markersize=10)
    ax.plot(periods, control, 's-', label='Control Group', linewidth=2, markersize=10)

    # DID効果を矢印で表示
    did_estimate = did_results['did_estimate']
    ax.annotate(
        f'DID = {did_estimate:.3f}',
        xy=(1, means['treated_post']),
        xytext=(1.2, means['treated_post'] + did_estimate/2),
        arrowprops=dict(arrowstyle='->', color='red', lw=2),
        fontsize=12,
        fontweight='bold',
        color='red'
    )

    ax.set_ylabel('Outcome', fontsize=12, fontweight='bold')
    ax.set_title('Difference-in-Differences Effect', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_did_coefficients(
    did_results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple = (8, 6)
) -> None:
    """
    DID推定値と信頼区間のプロット

    Parameters
    ----------
    did_results : Dict
        did_estimation()の結果
    save_path : str, optional
        保存先パス
    figsize : Tuple
        図のサイズ
    """
    fig, ax = plt.subplots(figsize=figsize)

    estimate = did_results['did_estimate']
    ci_lower = did_results['ci_lower']
    ci_upper = did_results['ci_upper']

    ax.errorbar(0, estimate, yerr=[[estimate - ci_lower], [ci_upper - estimate]],
                fmt='o', markersize=12, capsize=10, capthick=2, linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)

    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([0])
    ax.set_xticklabels(['DID Estimate'])
    ax.set_ylabel('Treatment Effect', fontsize=12, fontweight='bold')
    ax.set_title('DID Estimate with 95% CI', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 推定値をテキストで表示
    ax.text(0.1, estimate, f'{estimate:.3f}\n[{ci_lower:.3f}, {ci_upper:.3f}]',
            fontsize=10, verticalalignment='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()
