"""
感度分析モジュール

隠れた交絡因子（観測されていない変数）の影響を評価する手法を提供します。

主な機能：
1. Rosenbaum Bounds: 隠れた交絡因子の影響範囲を計算
2. E-value: 結果を無効にするために必要な最小の交絡の強さを計算
3. 感度分析レポート生成
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, binom
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def rosenbaum_bounds(
    treated_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    gamma_values: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Rosenbaum bounds を計算

    Rosenbaumの感度分析は、傾向スコアマッチング後に隠れた交絡因子が
    どの程度存在しても結果が頑健かを評価します。

    Parameters
    ----------
    treated_outcomes : np.ndarray
        処置群（スキル保有者）のアウトカム（優秀度）
    control_outcomes : np.ndarray
        対照群（スキル非保有者）のアウトカム（優秀度）
    gamma_values : List[float], optional
        評価するGamma値のリスト。None の場合は [1.0, 1.5, 2.0, 2.5, 3.0]

    Returns
    -------
    pd.DataFrame
        各Gamma値に対する上限・下限のp値を含むデータフレーム

    Notes
    -----
    Gamma値の解釈:
    - Gamma=1: 隠れた交絡なし（マッチングが完璧）
    - Gamma=2: 隠れた交絡により傾向スコアが2倍異なる可能性
    - Gamma=3: 隠れた交絡により傾向スコアが3倍異なる可能性

    結果の頑健性:
    - 上限p値が有意水準以下 → その程度の隠れた交絡があっても結果は有意
    - 下限p値が有意水準を超える → その程度の隠れた交絡で結果が非有意になる

    References
    ----------
    Rosenbaum, P. R. (2002). "Observational Studies" (2nd ed.). Springer.
    """
    if gamma_values is None:
        gamma_values = [1.0, 1.5, 2.0, 2.5, 3.0]

    n_treated = len(treated_outcomes)
    n_control = len(control_outcomes)

    # Wilcoxon符号順位検定の統計量を計算
    all_outcomes = np.concatenate([treated_outcomes, control_outcomes])
    all_ranks = np.argsort(np.argsort(all_outcomes)) + 1  # 順位

    treated_ranks = all_ranks[:n_treated]
    T_plus = np.sum(treated_ranks)  # 処置群の順位和

    results = []

    for gamma in gamma_values:
        # Gamma調整後の期待値と分散
        # 上限（最も不利なシナリオ）
        p_upper = gamma / (1 + gamma)
        E_plus_upper = n_treated * (n_treated + n_control + 1) * p_upper / 2
        Var_plus_upper = n_treated * n_control * (n_treated + n_control + 1) * gamma / (12 * (1 + gamma)**2)

        # 下限（最も有利なシナリオ）
        p_lower = 1 / (1 + gamma)
        E_plus_lower = n_treated * (n_treated + n_control + 1) * p_lower / 2
        Var_plus_lower = n_treated * n_control * (n_treated + n_control + 1) / (12 * gamma * (1 + gamma)**2)

        # 標準化統計量とp値
        if Var_plus_upper > 0:
            z_upper = (T_plus - E_plus_upper) / np.sqrt(Var_plus_upper)
            p_val_upper = 1 - norm.cdf(z_upper)  # 片側検定
        else:
            p_val_upper = np.nan

        if Var_plus_lower > 0:
            z_lower = (T_plus - E_plus_lower) / np.sqrt(Var_plus_lower)
            p_val_lower = 1 - norm.cdf(z_lower)  # 片側検定
        else:
            p_val_lower = np.nan

        results.append({
            'Gamma': gamma,
            'P_value_upper': p_val_upper,
            'P_value_lower': p_val_lower,
            'Significant_at_0.05_upper': p_val_upper < 0.05 if not np.isnan(p_val_upper) else False,
            'Significant_at_0.05_lower': p_val_lower < 0.05 if not np.isnan(p_val_lower) else False
        })

        logger.info(f"Rosenbaum bounds - Gamma={gamma}: p_upper={p_val_upper:.4f}, p_lower={p_val_lower:.4f}")

    return pd.DataFrame(results)


def calculate_e_value(
    effect_estimate: float,
    effect_se: Optional[float] = None,
    effect_type: str = "odds_ratio"
) -> Dict[str, float]:
    """
    E-value を計算

    E-valueは、観測された因果効果を無効にするために必要な
    最小の交絡因子の強さを示します。

    Parameters
    ----------
    effect_estimate : float
        観測された効果の推定値（オッズ比、リスク比、平均差など）
    effect_se : float, optional
        効果推定値の標準誤差。Noneの場合は信頼区間のE-valueは計算されない
    effect_type : str
        効果の種類: "odds_ratio", "risk_ratio", "mean_difference"

    Returns
    -------
    Dict[str, float]
        'point_estimate': 点推定値のE-value
        'ci_lower': 信頼区間下限のE-value（effect_seが提供された場合）
        'interpretation': 解釈用の値

    Notes
    -----
    E-valueの解釈:
    - E-value=1.5: RR=1.5の交絡因子が必要
    - E-value=2.0: RR=2.0の交絡因子が必要
    - E-value=3.0: RR=3.0の交絡因子が必要（かなり強い交絡）

    高いE-value → 結果は頑健（強い隠れた交絡が必要）
    低いE-value → 結果は脆弱（弱い隠れた交絡で無効化される）

    References
    ----------
    VanderWeele, T. J., & Ding, P. (2017). "Sensitivity Analysis in Observational Research:
    Introducing the E-Value." Annals of Internal Medicine, 167(4), 268-274.
    """

    if effect_type == "odds_ratio" or effect_type == "risk_ratio":
        if effect_estimate < 1:
            # 保護効果の場合は逆数を取る
            effect_estimate = 1 / effect_estimate

        # E-value = RR + sqrt(RR * (RR - 1))
        e_value_point = effect_estimate + np.sqrt(effect_estimate * (effect_estimate - 1))

    elif effect_type == "mean_difference":
        # 平均差の場合は標準化する必要がある
        # ここでは簡易的に実装（実際にはSDが必要）
        raise NotImplementedError("mean_difference type requires standardization. Use effect_estimate as Cohen's d.")

    else:
        raise ValueError(f"Unknown effect_type: {effect_type}")

    result = {
        'point_estimate': e_value_point,
        'interpretation': _interpret_e_value(e_value_point)
    }

    # 信頼区間下限のE-valueを計算
    if effect_se is not None:
        # 95% CI の下限を計算（対数スケールで）
        log_effect = np.log(effect_estimate)
        ci_lower_log = log_effect - 1.96 * effect_se
        ci_lower = np.exp(ci_lower_log)

        if ci_lower > 1:
            e_value_ci = ci_lower + np.sqrt(ci_lower * (ci_lower - 1))
        else:
            e_value_ci = 1.0  # 信頼区間が1を跨ぐ場合

        result['ci_lower'] = e_value_ci

    logger.info(f"E-value calculated - Point: {e_value_point:.3f}, Interpretation: {result['interpretation']}")

    return result


def _interpret_e_value(e_value: float) -> str:
    """E-valueの解釈を提供"""
    if e_value < 1.5:
        return "非常に脆弱（弱い交絡で無効化される）"
    elif e_value < 2.0:
        return "やや脆弱（中程度の交絡で無効化される）"
    elif e_value < 3.0:
        return "中程度の頑健性（強い交絡が必要）"
    elif e_value < 4.0:
        return "頑健（非常に強い交絡が必要）"
    else:
        return "非常に頑健（極めて強い交絡が必要）"


def sensitivity_analysis_report(
    treated_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    effect_estimate: float,
    effect_se: Optional[float] = None,
    effect_type: str = "odds_ratio",
    gamma_values: Optional[List[float]] = None
) -> Dict:
    """
    包括的な感度分析レポートを生成

    Parameters
    ----------
    treated_outcomes : np.ndarray
        処置群のアウトカム
    control_outcomes : np.ndarray
        対照群のアウトカム
    effect_estimate : float
        観測された効果の推定値
    effect_se : float, optional
        効果推定値の標準誤差
    effect_type : str
        効果の種類
    gamma_values : List[float], optional
        Rosenbaum bounds で評価するGamma値

    Returns
    -------
    Dict
        'rosenbaum_bounds': Rosenbaum bounds の結果
        'e_value': E-value の結果
        'summary': テキストサマリー
        'recommendation': 推奨事項
    """
    logger.info("Generating comprehensive sensitivity analysis report...")

    # Rosenbaum bounds
    rosenbaum_results = rosenbaum_bounds(treated_outcomes, control_outcomes, gamma_values)

    # E-value
    e_value_results = calculate_e_value(effect_estimate, effect_se, effect_type)

    # サマリーテキスト生成
    summary = _generate_summary(rosenbaum_results, e_value_results)

    # 推奨事項
    recommendation = _generate_recommendation(rosenbaum_results, e_value_results)

    report = {
        'rosenbaum_bounds': rosenbaum_results,
        'e_value': e_value_results,
        'summary': summary,
        'recommendation': recommendation
    }

    logger.info("Sensitivity analysis report generated successfully")

    return report


def _generate_summary(rosenbaum_results: pd.DataFrame, e_value_results: Dict) -> str:
    """サマリーテキストを生成"""

    # Rosenbaumの結果サマリー
    max_gamma_significant = 1.0
    for _, row in rosenbaum_results.iterrows():
        if row['Significant_at_0.05_upper']:
            max_gamma_significant = row['Gamma']

    rosenbaum_summary = f"Rosenbaum bounds分析: Gamma={max_gamma_significant}までは結果が有意に保たれます。"

    # E-valueの結果サマリー
    e_val = e_value_results['point_estimate']
    e_summary = f"E-value: {e_val:.2f}（{e_value_results['interpretation']}）"

    full_summary = f"""
【感度分析サマリー】

{rosenbaum_summary}
{e_summary}

これは、隠れた交絡因子が傾向スコアを{max_gamma_significant}倍変化させる程度であれば、
観測された因果効果は統計的に有意なままであることを意味します。

E-valueは、観測された効果を無効にするために、リスク比{e_val:.2f}の
隠れた交絡因子が必要であることを示しています。
"""

    return full_summary.strip()


def _generate_recommendation(rosenbaum_results: pd.DataFrame, e_value_results: Dict) -> str:
    """推奨事項を生成"""

    e_val = e_value_results['point_estimate']

    if e_val >= 3.0:
        rec = """
【推奨事項】
✅ 結果は隠れた交絡に対して頑健です。
✅ 因果的解釈に高い信頼性があります。
✅ ビジネス施策への活用を推奨します。
"""
    elif e_val >= 2.0:
        rec = """
【推奨事項】
⚠️ 結果は中程度の頑健性を持ちます。
⚠️ 可能であれば追加のデータ収集や検証を推奨します。
⚠️ ビジネス施策への活用は慎重に行ってください。
⚠️ 他の証拠と組み合わせて判断することを推奨します。
"""
    else:
        rec = """
【推奨事項】
❌ 結果は隠れた交絡に対して脆弱です。
❌ 因果的解釈には慎重さが必要です。
❌ 追加のデータ収集や別の分析手法の検討を強く推奨します。
❌ ビジネス施策への活用は時期尚早です。
"""

    return rec.strip()


def plot_sensitivity_analysis(
    rosenbaum_results: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    感度分析の結果をプロット

    Parameters
    ----------
    rosenbaum_results : pd.DataFrame
        rosenbaum_bounds()の結果
    save_path : str, optional
        保存先のパス
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")

        fig, ax = plt.subplots(figsize=(10, 6))

        gamma_vals = rosenbaum_results['Gamma'].values
        p_upper = rosenbaum_results['P_value_upper'].values
        p_lower = rosenbaum_results['P_value_lower'].values

        ax.plot(gamma_vals, p_upper, 'o-', label='Upper bound (worst case)', linewidth=2, markersize=8)
        ax.plot(gamma_vals, p_lower, 's-', label='Lower bound (best case)', linewidth=2, markersize=8)
        ax.axhline(y=0.05, color='r', linestyle='--', label='α=0.05', linewidth=2)

        ax.set_xlabel('Gamma (隠れた交絡の強さ)', fontsize=12, fontweight='bold')
        ax.set_ylabel('P-value', fontsize=12, fontweight='bold')
        ax.set_title('Rosenbaum Bounds: 感度分析', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sensitivity analysis plot saved to {save_path}")

        plt.close()

    except ImportError:
        logger.warning("matplotlib/seaborn not available for plotting")
