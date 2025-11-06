"""
クラスター頑健標準誤差モジュール

部門やグループ内の相関を考慮した統計的推論を提供します。

主な機能：
1. クラスター頑健標準誤差の計算
2. クラスター調整されたp値と信頼区間
3. クラスター内相関係数（ICC）の計算
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional, Union
import logging
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

logger = logging.getLogger(__name__)


def cluster_robust_se(
    y: np.ndarray,
    X: np.ndarray,
    clusters: np.ndarray,
    add_intercept: bool = True
) -> Dict:
    """
    クラスター頑健標準誤差を計算

    部門や組織単位でのクラスタリングを考慮した標準誤差を計算します。
    同じクラスター内の観測値は相関している可能性があるため、
    通常の標準誤差は過小評価されます。

    Parameters
    ----------
    y : np.ndarray
        アウトカム変数（優秀度など）
    X : np.ndarray
        説明変数（スキル保有、共変量など）
        Shape: (n_samples, n_features)
    clusters : np.ndarray
        クラスターID（部門コードなど）
        Shape: (n_samples,)
    add_intercept : bool
        切片を追加するか

    Returns
    -------
    Dict
        'coefficients': 回帰係数
        'se_regular': 通常の標準誤差
        'se_cluster': クラスター頑健標準誤差
        'p_values': クラスター調整されたp値
        'ci_lower': 95%信頼区間下限
        'ci_upper': 95%信頼区間上限
        'n_clusters': クラスター数
        'icc': クラスター内相関係数

    Notes
    -----
    クラスター頑健標準誤差は以下の場合に重要：
    - 同じ部門の社員は似た特性を持つ
    - 同じマネージャーの下で働く社員は相関する
    - 地理的に近い拠点の社員は類似する

    標準誤差の比較:
    - SE_cluster > SE_regular の場合が多い
    - 比率 = SE_cluster / SE_regular は設計効果（design effect）
    - 設計効果 > 1.5 の場合、クラスタリングの影響が大きい

    References
    ----------
    Cameron, A. C., & Miller, D. L. (2015). "A practitioner's guide to cluster-robust inference."
    Journal of Human Resources, 50(2), 317-372.
    """

    # データ検証
    if len(y) != len(X) or len(y) != len(clusters):
        raise ValueError("y, X, clusters must have the same length")

    # 切片を追加
    if add_intercept:
        X = add_constant(X)

    n_samples, n_features = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    logger.info(f"Calculating cluster-robust SE with {n_clusters} clusters, {n_samples} observations")

    # OLS推定
    model = OLS(y, X)
    results = model.fit()

    # 通常の標準誤差
    se_regular = results.bse.values
    coefficients = results.params.values

    # クラスター頑健標準誤差を手動計算
    # Meat of sandwich estimator
    X_demeaned = X - X.mean(axis=0)
    residuals = y - results.fittedvalues

    # クラスターごとのスコアを計算
    cluster_scores = np.zeros((n_clusters, n_features))

    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = clusters == cluster_id
        cluster_X = X[cluster_mask]
        cluster_residuals = residuals[cluster_mask]

        # Score = X' * e for this cluster
        cluster_scores[i] = cluster_X.T @ cluster_residuals

    # Meat matrix: sum of outer products of cluster scores
    meat = cluster_scores.T @ cluster_scores

    # Bread matrix: (X'X)^{-1}
    bread = results.normalized_cov_params.values

    # Small sample adjustment
    small_sample_adj = n_clusters / (n_clusters - 1) * (n_samples - 1) / (n_samples - n_features)

    # Cluster-robust variance-covariance matrix
    vcov_cluster = small_sample_adj * bread @ meat @ bread

    # クラスター頑健標準誤差
    se_cluster = np.sqrt(np.diag(vcov_cluster))

    # t統計量とp値（自由度はクラスター数-1を使用）
    t_stats = coefficients / se_cluster
    df = n_clusters - 1
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))

    # 95%信頼区間
    t_critical = stats.t.ppf(0.975, df)
    ci_lower = coefficients - t_critical * se_cluster
    ci_upper = coefficients + t_critical * se_cluster

    # クラスター内相関係数（ICC）を計算
    icc = calculate_icc(y, clusters)

    # 設計効果
    design_effect = se_cluster / se_regular

    results_dict = {
        'coefficients': coefficients,
        'se_regular': se_regular,
        'se_cluster': se_cluster,
        't_statistics': t_stats,
        'p_values': p_values,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_clusters': n_clusters,
        'n_observations': n_samples,
        'df': df,
        'icc': icc,
        'design_effect': design_effect,
        'vcov_cluster': vcov_cluster
    }

    logger.info(f"Cluster-robust SE calculated - ICC: {icc:.4f}, Mean design effect: {design_effect.mean():.3f}")

    return results_dict


def calculate_icc(y: np.ndarray, clusters: np.ndarray) -> float:
    """
    クラスター内相関係数（Intraclass Correlation Coefficient, ICC）を計算

    ICCは、全体の分散のうちクラスター間の分散が占める割合を示します。

    Parameters
    ----------
    y : np.ndarray
        アウトカム変数
    clusters : np.ndarray
        クラスターID

    Returns
    -------
    float
        ICC（0から1の値）

    Notes
    -----
    ICC の解釈:
    - ICC < 0.05: クラスタリング効果は小さい
    - 0.05 <= ICC < 0.15: 中程度のクラスタリング効果
    - ICC >= 0.15: 大きなクラスタリング効果

    ICC = σ²_between / (σ²_between + σ²_within)
    """

    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # 全体平均
    grand_mean = np.mean(y)

    # クラスターごとの平均とサイズ
    cluster_means = []
    cluster_sizes = []

    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        cluster_y = y[cluster_mask]
        cluster_means.append(np.mean(cluster_y))
        cluster_sizes.append(len(cluster_y))

    cluster_means = np.array(cluster_means)
    cluster_sizes = np.array(cluster_sizes)

    # Between-cluster variance
    ss_between = np.sum(cluster_sizes * (cluster_means - grand_mean) ** 2)
    ms_between = ss_between / (n_clusters - 1)

    # Within-cluster variance
    ss_within = 0
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        cluster_y = y[cluster_mask]
        cluster_mean = cluster_means[unique_clusters == cluster_id][0]
        ss_within += np.sum((cluster_y - cluster_mean) ** 2)

    df_within = len(y) - n_clusters
    ms_within = ss_within / df_within if df_within > 0 else 0

    # Average cluster size
    n_bar = len(y) / n_clusters

    # ICC calculation
    if ms_within > 0:
        icc = (ms_between - ms_within) / (ms_between + (n_bar - 1) * ms_within)
        icc = max(0, min(1, icc))  # Bound to [0, 1]
    else:
        icc = 0

    return icc


def cluster_robust_inference(
    y: np.ndarray,
    treatment: np.ndarray,
    covariates: Optional[np.ndarray],
    clusters: np.ndarray,
    treatment_name: str = "Treatment"
) -> pd.DataFrame:
    """
    クラスター頑健な因果推論

    処置効果（例：スキルの効果）をクラスタリングを考慮して推定します。

    Parameters
    ----------
    y : np.ndarray
        アウトカム変数（優秀度）
    treatment : np.ndarray
        処置変数（スキル保有: 0 or 1）
    covariates : np.ndarray, optional
        共変量（勤続年数、等級など）
    clusters : np.ndarray
        クラスターID（部門コード）
    treatment_name : str
        処置変数の名前

    Returns
    -------
    pd.DataFrame
        推定結果の表（係数、標準誤差、p値、信頼区間）

    Examples
    --------
    >>> import numpy as np
    >>> y = np.random.binomial(1, 0.5, 100)
    >>> treatment = np.random.binomial(1, 0.3, 100)
    >>> covariates = np.random.randn(100, 2)
    >>> clusters = np.repeat(np.arange(10), 10)
    >>> results = cluster_robust_inference(y, treatment, covariates, clusters)
    >>> print(results)
    """

    # データ構築
    if covariates is not None:
        X = np.column_stack([treatment.reshape(-1, 1), covariates])
        var_names = [treatment_name] + [f"Covariate_{i+1}" for i in range(covariates.shape[1])]
    else:
        X = treatment.reshape(-1, 1)
        var_names = [treatment_name]

    # クラスター頑健推定
    results = cluster_robust_se(y, X, clusters, add_intercept=True)

    # 結果を表形式に整理
    results_df = pd.DataFrame({
        'Variable': ['Intercept'] + var_names,
        'Coefficient': results['coefficients'],
        'SE_Regular': results['se_regular'],
        'SE_Cluster': results['se_cluster'],
        'Design_Effect': results['design_effect'],
        'T_Statistic': results['t_statistics'],
        'P_Value': results['p_values'],
        'CI_Lower': results['ci_lower'],
        'CI_Upper': results['ci_upper']
    })

    # 有意性マーカー
    results_df['Significance'] = results_df['P_Value'].apply(_significance_stars)

    logger.info(f"Cluster-robust inference completed for {treatment_name}")
    logger.info(f"Treatment effect: {results['coefficients'][1]:.4f} (SE: {results['se_cluster'][1]:.4f}, p={results['p_values'][1]:.4f})")

    return results_df


def _significance_stars(p_value: float) -> str:
    """有意性のマーカーを返す"""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value < 0.1:
        return "."
    else:
        return ""


def cluster_summary_statistics(clusters: np.ndarray) -> pd.DataFrame:
    """
    クラスター構造のサマリー統計を計算

    Parameters
    ----------
    clusters : np.ndarray
        クラスターID

    Returns
    -------
    pd.DataFrame
        クラスターごとのサイズ、割合などの統計
    """

    unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
    n_total = len(clusters)

    summary = pd.DataFrame({
        'Cluster_ID': unique_clusters,
        'Size': cluster_counts,
        'Percentage': 100 * cluster_counts / n_total
    })

    summary = summary.sort_values('Size', ascending=False)

    logger.info(f"Cluster summary - {len(unique_clusters)} clusters, "
                f"Mean size: {cluster_counts.mean():.1f}, "
                f"Median size: {np.median(cluster_counts):.1f}")

    return summary


def recommend_clustering_approach(n_clusters: int, n_observations: int, icc: float) -> str:
    """
    クラスタリングアプローチの推奨事項を提供

    Parameters
    ----------
    n_clusters : int
        クラスター数
    n_observations : int
        観測数
    icc : float
        クラスター内相関係数

    Returns
    -------
    str
        推奨事項のテキスト
    """

    avg_cluster_size = n_observations / n_clusters

    recommendation = f"""
【クラスター分析の推奨事項】

クラスター数: {n_clusters}
平均クラスターサイズ: {avg_cluster_size:.1f}
ICC: {icc:.4f}

"""

    if n_clusters < 30:
        recommendation += """
⚠️ クラスター数が少ない（<30）ため、以下に注意してください：
  - Wild cluster bootstrap を使用することを推奨
  - クラスター頑健標準誤差は過小評価される可能性があります
  - 可能であればより多くのクラスターを含むデータを収集してください
"""

    if icc < 0.05:
        recommendation += """
✅ ICC が低い（<0.05）ため、クラスタリング効果は小さいです。
  - 通常の標準誤差でも大きな問題はありません
  - ただし、安全のためクラスター頑健SEの使用を推奨します
"""
    elif icc < 0.15:
        recommendation += """
⚠️ ICC が中程度（0.05-0.15）のため、クラスタリング効果を考慮すべきです。
  - クラスター頑健標準誤差の使用を強く推奨します
  - 通常のSEを使うと過度に楽観的な結果になります
"""
    else:
        recommendation += """
❌ ICC が高い（>=0.15）ため、クラスタリング効果が大きいです。
  - クラスター頑健標準誤差の使用が必須です
  - 固定効果モデルや階層モデルの使用も検討してください
  - 通常のSEを使うと深刻に誤った推論になります
"""

    if avg_cluster_size < 5:
        recommendation += """
⚠️ 平均クラスターサイズが小さい（<5）ため、推定が不安定になる可能性があります。
"""

    return recommendation.strip()
