"""
差分の差分法（Difference-in-Differences, DID）分析モジュール

時系列データを活用した因果推論手法を提供します。
資格取得日のデータを使用して、スキル習得の因果効果を推定します。

主な機能：
1. 標準的なDID推定
2. 平行トレンド仮定の検証
3. 共変量調整付きDID
4. 多時点DID（スタガード採用デザイン）
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def did_estimation(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    time_col: str,
    unit_col: str,
    treatment_date_col: Optional[str] = None,
    pre_period: Optional[Tuple] = None,
    post_period: Optional[Tuple] = None
) -> Dict:
    """
    差分の差分法（DID）による因果効果の推定

    DIDは、処置群と対照群の時系列変化を比較することで、
    時間不変の交絡因子（個人の能力、性格など）を除去できます。

    Parameters
    ----------
    df : pd.DataFrame
        分析データ（パネルデータ形式）
    outcome_col : str
        アウトカム変数の列名（例: 'is_excellent'）
    treatment_col : str
        処置変数の列名（例: 'has_skill'）
    time_col : str
        時点を示す列名（例: 'year', 'date'）
    unit_col : str
        個体IDの列名（例: 'member_id'）
    treatment_date_col : str, optional
        処置（スキル取得）が発生した日付の列名
    pre_period : Tuple, optional
        処置前期間の範囲（例: (2020, 2021)）
    post_period : Tuple, optional
        処置後期間の範囲（例: (2022, 2023)）

    Returns
    -------
    Dict
        'did_estimate': DID推定値（処置効果）
        'se': 標準誤差
        'p_value': p値
        'ci_lower': 95%信頼区間下限
        'ci_upper': 95%信頼区間上限
        'n_treated': 処置群サイズ
        'n_control': 対照群サイズ
        'parallel_trends_test': 平行トレンド検定の結果

    Notes
    -----
    DIDの仮定:
    1. 平行トレンド仮定: 処置がない場合、両群のトレンドは平行
    2. 共通ショック: 処置以外の時間効果は両群で同じ
    3. 処置の外生性: 処置のタイミングは結果に影響されない

    DIDの利点:
    - 時間不変の交絡因子を除去
    - 観測されない個人特性（能力、性格）をコントロール
    - PSMより強い因果推論が可能

    推定式:
    Y_it = β0 + β1*Treated_i + β2*Post_t + β3*(Treated_i * Post_t) + ε_it

    DID推定値 = β3 = (Y_treated_post - Y_treated_pre) - (Y_control_post - Y_control_pre)

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'member_id': [1, 1, 2, 2, 3, 3, 4, 4],
    ...     'year': [2020, 2022, 2020, 2022, 2020, 2022, 2020, 2022],
    ...     'has_skill': [0, 1, 0, 1, 0, 0, 0, 0],
    ...     'is_excellent': [0, 1, 0, 0, 0, 0, 0, 0]
    ... })
    >>> results = did_estimation(df, 'is_excellent', 'has_skill', 'year', 'member_id')
    """

    logger.info(f"Starting DID estimation for outcome: {outcome_col}, treatment: {treatment_col}")

    # データの前処理
    df = df.copy()

    # 処置群と対照群を識別
    # 処置群: 期間中に処置を受けた個体
    treated_units = df[df[treatment_col] == 1][unit_col].unique()
    df['treated_group'] = df[unit_col].isin(treated_units).astype(int)

    # 処置前後の期間を定義
    if pre_period is None or post_period is None:
        # 自動的に期間を設定
        time_values = sorted(df[time_col].unique())
        mid_point = len(time_values) // 2
        pre_period = (time_values[0], time_values[mid_point - 1])
        post_period = (time_values[mid_point], time_values[-1])
        logger.info(f"Auto-detected periods - Pre: {pre_period}, Post: {post_period}")

    # Pre/Post フラグ
    df['post'] = ((df[time_col] >= post_period[0]) & (df[time_col] <= post_period[1])).astype(int)
    df['pre'] = ((df[time_col] >= pre_period[0]) & (df[time_col] <= pre_period[1])).astype(int)

    # 分析対象データ（pre or post期間のみ）
    df_analysis = df[(df['pre'] == 1) | (df['post'] == 1)].copy()

    # 各グループ・期間の平均アウトカムを計算
    means = df_analysis.groupby(['treated_group', 'post'])[outcome_col].mean()

    try:
        y_treated_pre = means.loc[(1, 0)]
        y_treated_post = means.loc[(1, 1)]
        y_control_pre = means.loc[(0, 0)]
        y_control_post = means.loc[(0, 1)]
    except KeyError:
        logger.error("Insufficient data for DID estimation (missing group-period combinations)")
        raise ValueError("Cannot compute DID: missing data for some group-period combinations")

    # DID推定値
    did_estimate = (y_treated_post - y_treated_pre) - (y_control_post - y_control_pre)

    logger.info(f"DID estimate: {did_estimate:.4f}")
    logger.info(f"  Treated: {y_treated_pre:.3f} -> {y_treated_post:.3f} (Δ={y_treated_post - y_treated_pre:.3f})")
    logger.info(f"  Control: {y_control_pre:.3f} -> {y_control_post:.3f} (Δ={y_control_post - y_control_pre:.3f})")

    # 回帰による推定（標準誤差を得るため）
    from statsmodels.formula.api import ols

    df_analysis['interaction'] = df_analysis['treated_group'] * df_analysis['post']

    formula = f"{outcome_col} ~ treated_group + post + interaction"
    model = ols(formula, data=df_analysis).fit()

    # DID推定値は interaction 項の係数
    did_coef = model.params['interaction']
    did_se = model.bse['interaction']
    did_pval = model.pvalues['interaction']
    did_ci = model.conf_int().loc['interaction']

    # サンプルサイズ
    n_treated = df_analysis[df_analysis['treated_group'] == 1][unit_col].nunique()
    n_control = df_analysis[df_analysis['treated_group'] == 0][unit_col].nunique()

    # 平行トレンド検定
    parallel_trends_result = parallel_trends_test(df, outcome_col, unit_col, time_col,
                                                    'treated_group', pre_period)

    results = {
        'did_estimate': did_estimate,
        'did_coefficient': did_coef,
        'se': did_se,
        'p_value': did_pval,
        'ci_lower': did_ci[0],
        'ci_upper': did_ci[1],
        'n_treated': n_treated,
        'n_control': n_control,
        'n_observations': len(df_analysis),
        'means': {
            'treated_pre': y_treated_pre,
            'treated_post': y_treated_post,
            'control_pre': y_control_pre,
            'control_post': y_control_post
        },
        'parallel_trends_test': parallel_trends_result,
        'model_summary': model.summary()
    }

    logger.info(f"DID estimation completed - Estimate: {did_estimate:.4f}, p-value: {did_pval:.4f}")

    return results


def parallel_trends_test(
    df: pd.DataFrame,
    outcome_col: str,
    unit_col: str,
    time_col: str,
    treated_col: str,
    pre_period: Tuple,
    test_method: str = "regression"
) -> Dict:
    """
    平行トレンド仮定の検定

    DIDの重要な仮定である「平行トレンド」が満たされているかを検証します。
    処置前期間において、処置群と対照群のトレンドが平行であることを確認します。

    Parameters
    ----------
    df : pd.DataFrame
        分析データ
    outcome_col : str
        アウトカム変数
    unit_col : str
        個体ID
    time_col : str
        時点
    treated_col : str
        処置群フラグ（0 or 1）
    pre_period : Tuple
        処置前期間
    test_method : str
        検定方法: "regression" or "visual"

    Returns
    -------
    Dict
        'test_statistic': 検定統計量
        'p_value': p値
        'result': "Pass" or "Fail"
        'interpretation': 解釈テキスト

    Notes
    -----
    平行トレンド仮定:
    - 処置がない場合、両群のアウトカムのトレンドは平行
    - 処置前期間で検証する
    - 満たされない場合、DID推定値はバイアスを持つ

    検定方法（回帰法）:
    Y_it = β0 + β1*Treated_i + β2*Time_t + β3*(Treated_i * Time_t) + ε_it

    H0: β3 = 0（平行トレンド）
    H1: β3 ≠ 0（非平行）

    p > 0.05 であれば平行トレンド仮定は棄却されない（良い）
    """

    logger.info("Testing parallel trends assumption...")

    # 処置前期間のデータのみ
    df_pre = df[(df[time_col] >= pre_period[0]) & (df[time_col] <= pre_period[1])].copy()

    if len(df_pre) == 0:
        logger.warning("No data in pre-treatment period for parallel trends test")
        return {
            'test_statistic': np.nan,
            'p_value': np.nan,
            'result': 'Inconclusive',
            'interpretation': 'データ不足のため検定できません'
        }

    if test_method == "regression":
        # 時間変数を数値化
        df_pre['time_numeric'] = pd.to_datetime(df_pre[time_col]).astype(int) / 10**9 if df_pre[time_col].dtype == 'object' else df_pre[time_col]

        # 標準化
        df_pre['time_numeric'] = (df_pre['time_numeric'] - df_pre['time_numeric'].min())

        # 回帰分析
        from statsmodels.formula.api import ols

        df_pre['interaction_time'] = df_pre[treated_col] * df_pre['time_numeric']

        formula = f"{outcome_col} ~ {treated_col} + time_numeric + interaction_time"

        try:
            model = ols(formula, data=df_pre).fit()

            # 交互作用項の係数とp値
            coef = model.params['interaction_time']
            pval = model.pvalues['interaction_time']

            result = "Pass" if pval > 0.05 else "Fail"

            if result == "Pass":
                interpretation = f"""
✅ 平行トレンド仮定は棄却されません（p={pval:.3f} > 0.05）
  - 処置前期間で両群のトレンドは統計的に有意な差がありません
  - DID推定は適切に使用できます
"""
            else:
                interpretation = f"""
❌ 平行トレンド仮定が棄却されます（p={pval:.3f} < 0.05）
  - 処置前期間で両群のトレンドに有意な差があります
  - DID推定値にはバイアスがある可能性があります
  - 以下の対策を検討してください：
    1. 共変量調整DID（did_with_covariates）を使用
    2. より短い比較期間を選択
    3. 合成コントロール法など別の手法を検討
"""

            logger.info(f"Parallel trends test - Coefficient: {coef:.4f}, p-value: {pval:.4f}, Result: {result}")

            return {
                'test_statistic': coef,
                'p_value': pval,
                'result': result,
                'interpretation': interpretation.strip()
            }

        except Exception as e:
            logger.error(f"Parallel trends test failed: {str(e)}")
            return {
                'test_statistic': np.nan,
                'p_value': np.nan,
                'result': 'Error',
                'interpretation': f'検定中にエラーが発生しました: {str(e)}'
            }

    else:
        raise ValueError(f"Unknown test method: {test_method}")


def did_with_covariates(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    time_col: str,
    unit_col: str,
    covariate_cols: List[str],
    pre_period: Optional[Tuple] = None,
    post_period: Optional[Tuple] = None
) -> Dict:
    """
    共変量調整付きDID

    共変量（勤続年数、等級など）を含めたDID分析を実行します。
    平行トレンド仮定が厳密に満たされない場合に有用です。

    Parameters
    ----------
    df : pd.DataFrame
        分析データ
    outcome_col : str
        アウトカム変数
    treatment_col : str
        処置変数
    time_col : str
        時点
    unit_col : str
        個体ID
    covariate_cols : List[str]
        共変量の列名リスト
    pre_period : Tuple, optional
        処置前期間
    post_period : Tuple, optional
        処置後期間

    Returns
    -------
    Dict
        DID推定結果（共変量調整済み）

    Notes
    -----
    推定式:
    Y_it = β0 + β1*Treated_i + β2*Post_t + β3*(Treated_i * Post_t) + Σ(γ_k * X_k) + ε_it

    共変量を含めることで:
    - 時間変動する交絡因子をコントロール
    - 平行トレンド仮定を緩和
    - より正確な因果効果の推定

    Examples
    --------
    >>> results = did_with_covariates(
    ...     df, 'is_excellent', 'has_skill', 'year', 'member_id',
    ...     covariate_cols=['years_of_service', 'grade', 'position']
    ... )
    """

    logger.info(f"Starting DID with covariates: {covariate_cols}")

    # 基本的なDIDと同様の前処理
    df = df.copy()

    treated_units = df[df[treatment_col] == 1][unit_col].unique()
    df['treated_group'] = df[unit_col].isin(treated_units).astype(int)

    if pre_period is None or post_period is None:
        time_values = sorted(df[time_col].unique())
        mid_point = len(time_values) // 2
        pre_period = (time_values[0], time_values[mid_point - 1])
        post_period = (time_values[mid_point], time_values[-1])

    df['post'] = ((df[time_col] >= post_period[0]) & (df[time_col] <= post_period[1])).astype(int)
    df['pre'] = ((df[time_col] >= pre_period[0]) & (df[time_col] <= pre_period[1])).astype(int)

    df_analysis = df[(df['pre'] == 1) | (df['post'] == 1)].copy()
    df_analysis['interaction'] = df_analysis['treated_group'] * df_analysis['post']

    # 共変量を含む回帰式を構築
    from statsmodels.formula.api import ols

    covariate_formula = " + ".join(covariate_cols)
    formula = f"{outcome_col} ~ treated_group + post + interaction + {covariate_formula}"

    model = ols(formula, data=df_analysis).fit()

    # DID推定値（共変量調整済み）
    did_coef = model.params['interaction']
    did_se = model.bse['interaction']
    did_pval = model.pvalues['interaction']
    did_ci = model.conf_int().loc['interaction']

    # サンプルサイズ
    n_treated = df_analysis[df_analysis['treated_group'] == 1][unit_col].nunique()
    n_control = df_analysis[df_analysis['treated_group'] == 0][unit_col].nunique()

    # 共変量の効果も報告
    covariate_effects = {}
    for cov in covariate_cols:
        if cov in model.params:
            covariate_effects[cov] = {
                'coefficient': model.params[cov],
                'p_value': model.pvalues[cov]
            }

    results = {
        'did_estimate': did_coef,
        'se': did_se,
        'p_value': did_pval,
        'ci_lower': did_ci[0],
        'ci_upper': did_ci[1],
        'n_treated': n_treated,
        'n_control': n_control,
        'n_observations': len(df_analysis),
        'covariates_used': covariate_cols,
        'covariate_effects': covariate_effects,
        'model_r_squared': model.rsquared,
        'model_summary': model.summary()
    }

    logger.info(f"DID with covariates completed - Estimate: {did_coef:.4f}, p-value: {did_pval:.4f}")

    return results


def staggered_did(
    df: pd.DataFrame,
    outcome_col: str,
    unit_col: str,
    time_col: str,
    treatment_date_col: str,
    method: str = "callaway_santanna"
) -> Dict:
    """
    スタガード採用デザインDID（異なる時点で処置を受ける場合）

    個体ごとに異なる時点でスキルを取得する場合の因果効果を推定します。

    Parameters
    ----------
    df : pd.DataFrame
        分析データ
    outcome_col : str
        アウトカム変数
    unit_col : str
        個体ID
    time_col : str
        時点
    treatment_date_col : str
        処置が発生した日付
    method : str
        推定方法: "callaway_santanna" or "simple_average"

    Returns
    -------
    Dict
        ATT（Average Treatment Effect on the Treated）推定値

    Notes
    -----
    スタガードDIDの課題:
    - 異なる処置タイミングが存在
    - 標準的なTwo-way Fixed Effectsはバイアスを持つ
    - Callaway & Sant'Anna (2021) の手法を推奨

    この実装は簡易版です。厳密な分析には専用パッケージ（did, DIDmultiplegtなど）を推奨します。
    """

    logger.info(f"Starting staggered DID estimation with method: {method}")

    df = df.copy()
    df['treatment_date'] = pd.to_datetime(df[treatment_date_col])
    df['time'] = pd.to_datetime(df[time_col])

    # 各個体の処置時点を特定
    treatment_times = df.groupby(unit_col)['treatment_date'].min()

    # Never-treated（対照群）を特定
    never_treated = treatment_times[treatment_times.isna()].index

    # コホートごとにATTを推定
    cohort_atts = []

    for cohort_date in treatment_times.dropna().unique():
        cohort_units = treatment_times[treatment_times == cohort_date].index

        # このコホートと対照群を比較
        cohort_df = df[df[unit_col].isin(cohort_units) | df[unit_col].isin(never_treated)].copy()

        # Post indicator
        cohort_df['post'] = (cohort_df['time'] >= cohort_date).astype(int)
        cohort_df['treated'] = cohort_df[unit_col].isin(cohort_units).astype(int)
        cohort_df['interaction'] = cohort_df['treated'] * cohort_df['post']

        # 回帰推定
        from statsmodels.formula.api import ols
        model = ols(f"{outcome_col} ~ treated + post + interaction", data=cohort_df).fit()

        att_cohort = model.params['interaction']
        se_cohort = model.bse['interaction']

        cohort_atts.append({
            'cohort_date': cohort_date,
            'att': att_cohort,
            'se': se_cohort,
            'n_treated': len(cohort_units)
        })

    # 全体のATT（加重平均）
    total_treated = sum([c['n_treated'] for c in cohort_atts])
    overall_att = sum([c['att'] * c['n_treated'] for c in cohort_atts]) / total_treated if total_treated > 0 else np.nan

    logger.info(f"Staggered DID completed - Overall ATT: {overall_att:.4f}")

    return {
        'overall_att': overall_att,
        'cohort_results': cohort_atts,
        'n_cohorts': len(cohort_atts),
        'n_never_treated': len(never_treated),
        'method': method
    }
