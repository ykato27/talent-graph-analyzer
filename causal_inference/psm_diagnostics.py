"""
傾向スコアマッチング診断モジュール

PSMの品質を評価するための診断指標を提供します。

主な機能：
1. 標準化平均差（SMD）の計算
2. 共変量バランステーブルの生成
3. 傾向スコアの重なり度合いの評価
4. マッチング品質の評価
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_smd(
    treated: np.ndarray,
    control: np.ndarray,
    continuous: bool = True
) -> float:
    """
    標準化平均差（Standardized Mean Difference, SMD）を計算

    SMDは共変量のバランスを評価する標準的な指標です。
    絶対値が0.1未満であれば良好なバランスとされます。

    Parameters
    ----------
    treated : np.ndarray
        処置群の共変量
    control : np.ndarray
        対照群の共変量
    continuous : bool
        連続変数かどうか（Falseの場合は二値変数）

    Returns
    -------
    float
        SMD値

    Notes
    -----
    SMD の解釈:
    - |SMD| < 0.1: 優れたバランス
    - 0.1 <= |SMD| < 0.2: 良好なバランス
    - 0.2 <= |SMD| < 0.3: 許容可能なバランス
    - |SMD| >= 0.3: 不十分なバランス（マッチング改善が必要）

    連続変数の場合:
    SMD = (mean_treated - mean_control) / sqrt((var_treated + var_control) / 2)

    二値変数の場合:
    SMD = (p_treated - p_control) / sqrt((p_treated*(1-p_treated) + p_control*(1-p_control)) / 2)

    References
    ----------
    Austin, P. C. (2011). "An introduction to propensity score methods for reducing the effects
    of confounding in observational studies." Multivariate Behavioral Research, 46(3), 399-424.
    """

    if len(treated) == 0 or len(control) == 0:
        logger.warning("Empty array provided to calculate_smd")
        return np.nan

    if continuous:
        # 連続変数の場合
        mean_treated = np.mean(treated)
        mean_control = np.mean(control)
        var_treated = np.var(treated, ddof=1)
        var_control = np.var(control, ddof=1)

        pooled_std = np.sqrt((var_treated + var_control) / 2)

        if pooled_std == 0:
            return 0.0

        smd = (mean_treated - mean_control) / pooled_std

    else:
        # 二値変数の場合
        p_treated = np.mean(treated)
        p_control = np.mean(control)

        var_treated = p_treated * (1 - p_treated)
        var_control = p_control * (1 - p_control)

        pooled_std = np.sqrt((var_treated + var_control) / 2)

        if pooled_std == 0:
            return 0.0

        smd = (p_treated - p_control) / pooled_std

    return smd


def covariate_balance_table(
    X_treated_before: np.ndarray,
    X_control_before: np.ndarray,
    X_treated_after: np.ndarray,
    X_control_after: np.ndarray,
    covariate_names: List[str],
    continuous_vars: Optional[List[bool]] = None
) -> pd.DataFrame:
    """
    共変量バランステーブルを生成

    マッチング前後での共変量のバランスを比較します。

    Parameters
    ----------
    X_treated_before : np.ndarray
        マッチング前の処置群の共変量
        Shape: (n_treated, n_covariates)
    X_control_before : np.ndarray
        マッチング前の対照群の共変量
        Shape: (n_control, n_covariates)
    X_treated_after : np.ndarray
        マッチング後の処置群の共変量
    X_control_after : np.ndarray
        マッチング後の対照群の共変量
    covariate_names : List[str]
        共変量の名前
    continuous_vars : List[bool], optional
        各変数が連続変数かどうか（Noneの場合は全て連続変数と仮定）

    Returns
    -------
    pd.DataFrame
        バランステーブル

    Examples
    --------
    >>> balance_table = covariate_balance_table(
    ...     X_treated_before, X_control_before,
    ...     X_treated_after, X_control_after,
    ...     ['Years', 'Grade', 'Position']
    ... )
    >>> print(balance_table)
    """

    n_covariates = X_treated_before.shape[1]

    if continuous_vars is None:
        continuous_vars = [True] * n_covariates

    results = []

    for i, (name, is_continuous) in enumerate(zip(covariate_names, continuous_vars)):
        # マッチング前
        treated_before = X_treated_before[:, i]
        control_before = X_control_before[:, i]

        mean_treated_before = np.mean(treated_before)
        mean_control_before = np.mean(control_before)
        smd_before = calculate_smd(treated_before, control_before, is_continuous)

        # マッチング後
        treated_after = X_treated_after[:, i]
        control_after = X_control_after[:, i]

        mean_treated_after = np.mean(treated_after)
        mean_control_after = np.mean(control_after)
        smd_after = calculate_smd(treated_after, control_after, is_continuous)

        # SMDの改善率
        improvement = ((abs(smd_before) - abs(smd_after)) / abs(smd_before) * 100) if abs(smd_before) > 0 else 0

        # バランス評価
        balance_before = _evaluate_balance(smd_before)
        balance_after = _evaluate_balance(smd_after)

        results.append({
            'Covariate': name,
            'Mean_Treated_Before': mean_treated_before,
            'Mean_Control_Before': mean_control_before,
            'SMD_Before': smd_before,
            'Balance_Before': balance_before,
            'Mean_Treated_After': mean_treated_after,
            'Mean_Control_After': mean_control_after,
            'SMD_After': smd_after,
            'Balance_After': balance_after,
            'SMD_Improvement_%': improvement
        })

    balance_df = pd.DataFrame(results)

    # サマリー統計
    logger.info(f"Balance table generated for {n_covariates} covariates")
    logger.info(f"Mean |SMD| before: {balance_df['SMD_Before'].abs().mean():.3f}")
    logger.info(f"Mean |SMD| after: {balance_df['SMD_After'].abs().mean():.3f}")

    return balance_df


def _evaluate_balance(smd: float) -> str:
    """SMD値からバランスを評価"""
    abs_smd = abs(smd)
    if abs_smd < 0.1:
        return "Excellent"
    elif abs_smd < 0.2:
        return "Good"
    elif abs_smd < 0.3:
        return "Acceptable"
    else:
        return "Poor"


def check_overlap(
    ps_treated: np.ndarray,
    ps_control: np.ndarray,
    method: str = "minmax"
) -> Dict:
    """
    傾向スコアの重なり度合いを評価

    共通サポート（common support）の条件を満たしているかを確認します。

    Parameters
    ----------
    ps_treated : np.ndarray
        処置群の傾向スコア
    ps_control : np.ndarray
        対照群の傾向スコア
    method : str
        評価方法: "minmax", "percentile", "visual"

    Returns
    -------
    Dict
        'overlap_range': 重なり範囲 (min, max)
        'n_treated_in_range': 範囲内の処置群数
        'n_control_in_range': 範囲内の対照群数
        'percentage_treated': 範囲内の処置群の割合
        'percentage_control': 範囲内の対照群の割合
        'recommendation': 推奨事項

    Notes
    -----
    共通サポートの条件:
    - 処置群と対照群の傾向スコア分布が十分に重なっている
    - 両群の傾向スコアの範囲が類似している
    - 極端な傾向スコア（0に近い、1に近い）が少ない

    重なりが不十分な場合:
    - マッチングの質が低下
    - 外挿に依存した推定になる
    - 因果効果の推定が不安定

    対策:
    - トリミング（極端なスコアを除外）
    - カリパーを狭める
    - 共変量の選択を見直す
    """

    if method == "minmax":
        # Min-max方式: 両群が存在する範囲
        min_treated = np.min(ps_treated)
        max_treated = np.max(ps_treated)
        min_control = np.min(ps_control)
        max_control = np.max(ps_control)

        overlap_min = max(min_treated, min_control)
        overlap_max = min(max_treated, max_control)

    elif method == "percentile":
        # パーセンタイル方式: 5-95パーセンタイル
        p5_treated = np.percentile(ps_treated, 5)
        p95_treated = np.percentile(ps_treated, 95)
        p5_control = np.percentile(ps_control, 5)
        p95_control = np.percentile(ps_control, 95)

        overlap_min = max(p5_treated, p5_control)
        overlap_max = min(p95_treated, p95_control)

    else:
        raise ValueError(f"Unknown method: {method}")

    # 範囲内のサンプル数をカウント
    n_treated_in_range = np.sum((ps_treated >= overlap_min) & (ps_treated <= overlap_max))
    n_control_in_range = np.sum((ps_control >= overlap_min) & (ps_control <= overlap_max))

    percentage_treated = n_treated_in_range / len(ps_treated) * 100
    percentage_control = n_control_in_range / len(ps_control) * 100

    # 推奨事項
    recommendation = _generate_overlap_recommendation(
        percentage_treated, percentage_control, overlap_min, overlap_max
    )

    logger.info(f"Overlap check - Range: [{overlap_min:.3f}, {overlap_max:.3f}], "
                f"Treated: {percentage_treated:.1f}%, Control: {percentage_control:.1f}%")

    return {
        'overlap_range': (overlap_min, overlap_max),
        'n_treated_in_range': n_treated_in_range,
        'n_control_in_range': n_control_in_range,
        'percentage_treated': percentage_treated,
        'percentage_control': percentage_control,
        'recommendation': recommendation,
        'range_width': overlap_max - overlap_min
    }


def _generate_overlap_recommendation(
    pct_treated: float,
    pct_control: float,
    overlap_min: float,
    overlap_max: float
) -> str:
    """重なり度合いに基づく推奨事項"""

    min_percentage = min(pct_treated, pct_control)

    if min_percentage >= 90:
        rec = f"""
✅ 優れた重なり: 両群の{min_percentage:.1f}%以上が共通サポート範囲内です。
  - 因果効果の推定は信頼できます
  - 外挿のリスクは最小限です
"""
    elif min_percentage >= 75:
        rec = f"""
⚠️ 良好な重なり: 両群の{min_percentage:.1f}%が共通サポート範囲内です。
  - 因果効果の推定は概ね信頼できます
  - 一部のサンプルでは外挿に依存します
"""
    elif min_percentage >= 50:
        rec = f"""
⚠️ 中程度の重なり: 両群の{min_percentage:.1f}%のみが共通サポート範囲内です。
  - 因果効果の推定には注意が必要です
  - トリミング（極端なスコアの除外）を検討してください
  - カリパーを狭めることを推奨します
"""
    else:
        rec = f"""
❌ 不十分な重なり: 両群の{min_percentage:.1f}%しか共通サポート範囲内にありません。
  - 因果効果の推定は信頼できません
  - 以下の対策を強く推奨します:
    1. 共変量の選択を見直す
    2. より厳しいトリミング（5-95パーセンタイル）
    3. 別の因果推論手法の検討（IV法、DIDなど）
"""

    if overlap_max - overlap_min < 0.3:
        rec += f"""
⚠️ 重なり範囲が狭い（幅={overlap_max - overlap_min:.3f}）:
  - 推定される因果効果は限定的な集団にのみ適用可能
  - 一般化可能性に注意してください
"""

    return rec.strip()


def psm_quality_report(
    balance_table: pd.DataFrame,
    overlap_results: Dict,
    n_matched_pairs: int,
    n_treated_total: int,
    n_control_total: int
) -> Dict:
    """
    PSMの総合的な品質レポートを生成

    Parameters
    ----------
    balance_table : pd.DataFrame
        covariate_balance_table() の結果
    overlap_results : Dict
        check_overlap() の結果
    n_matched_pairs : int
        マッチングされたペア数
    n_treated_total : int
        処置群の総数
    n_control_total : int
        対照群の総数

    Returns
    -------
    Dict
        'overall_quality': 総合評価 ("Excellent", "Good", "Acceptable", "Poor")
        'balance_score': バランススコア (0-100)
        'overlap_score': 重なりスコア (0-100)
        'matching_rate': マッチング率
        'summary': サマリーテキスト
        'recommendations': 推奨事項リスト
    """

    # バランススコア計算
    mean_smd_after = balance_table['SMD_After'].abs().mean()
    max_smd_after = balance_table['SMD_After'].abs().max()
    n_poor_balance = (balance_table['SMD_After'].abs() >= 0.2).sum()

    balance_score = max(0, 100 - mean_smd_after * 500)  # 0.2以下なら90点以上

    # 重なりスコア計算
    min_overlap_pct = min(
        overlap_results['percentage_treated'],
        overlap_results['percentage_control']
    )
    overlap_score = min_overlap_pct

    # マッチング率
    matching_rate_treated = n_matched_pairs / n_treated_total * 100
    matching_rate_control = n_matched_pairs / n_control_total * 100

    # 総合評価
    overall_score = (balance_score * 0.5 + overlap_score * 0.3 + matching_rate_treated * 0.2)

    if overall_score >= 80:
        overall_quality = "Excellent"
    elif overall_score >= 60:
        overall_quality = "Good"
    elif overall_score >= 40:
        overall_quality = "Acceptable"
    else:
        overall_quality = "Poor"

    # サマリー生成
    summary = f"""
【PSM品質レポート】

総合評価: {overall_quality} (スコア: {overall_score:.1f}/100)

■ バランス評価
- 平均|SMD|（マッチング後）: {mean_smd_after:.3f}
- 最大|SMD|（マッチング後）: {max_smd_after:.3f}
- 不十分なバランスの変数数: {n_poor_balance}/{len(balance_table)}
- バランススコア: {balance_score:.1f}/100

■ 重なり評価
- 処置群の範囲内割合: {overlap_results['percentage_treated']:.1f}%
- 対照群の範囲内割合: {overlap_results['percentage_control']:.1f}%
- 重なりスコア: {overlap_score:.1f}/100

■ マッチング統計
- マッチングされたペア数: {n_matched_pairs}
- 処置群マッチング率: {matching_rate_treated:.1f}%
- 対照群マッチング率: {matching_rate_control:.1f}%
"""

    # 推奨事項
    recommendations = []

    if mean_smd_after > 0.1:
        recommendations.append("⚠️ 平均SMDが0.1を超えています。カリパーを狭めるか、共変量を追加してください。")

    if n_poor_balance > 0:
        poor_vars = balance_table[balance_table['SMD_After'].abs() >= 0.2]['Covariate'].tolist()
        recommendations.append(f"⚠️ バランスが不十分な変数: {', '.join(poor_vars)}")

    if min_overlap_pct < 75:
        recommendations.append("⚠️ 重なりが不十分です。トリミングを検討してください。")

    if matching_rate_treated < 70:
        recommendations.append("⚠️ マッチング率が低いです。カリパーを広げるか、サンプルサイズを増やしてください。")

    if len(recommendations) == 0:
        recommendations.append("✅ PSMの品質は良好です。因果推論を進めて問題ありません。")

    report = {
        'overall_quality': overall_quality,
        'overall_score': overall_score,
        'balance_score': balance_score,
        'overlap_score': overlap_score,
        'matching_rate_treated': matching_rate_treated,
        'matching_rate_control': matching_rate_control,
        'summary': summary.strip(),
        'recommendations': recommendations
    }

    logger.info(f"PSM quality report generated - Overall quality: {overall_quality} ({overall_score:.1f}/100)")

    return report
