"""
Pytestの共通設定とフィクスチャ
"""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple


@pytest.fixture
def sample_binary_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    サンプルの二値データを生成

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (treated_outcomes, control_outcomes)
    """
    np.random.seed(42)
    treated = np.random.binomial(1, 0.6, 50)
    control = np.random.binomial(1, 0.4, 50)
    return treated, control


@pytest.fixture
def sample_continuous_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    サンプルの連続データを生成

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (treated_outcomes, control_outcomes)
    """
    np.random.seed(42)
    treated = np.random.normal(1.0, 0.5, 50)
    control = np.random.normal(0.5, 0.5, 50)
    return treated, control


@pytest.fixture
def sample_regression_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    サンプルの回帰データを生成

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (y, X, clusters)
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 3

    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([1.5, -0.8, 0.3])
    y = X @ true_coef + np.random.randn(n_samples) * 0.5

    # 10個のクラスターを作成
    clusters = np.repeat(np.arange(10), 10)

    return y, X, clusters


@pytest.fixture
def sample_panel_data() -> pd.DataFrame:
    """
    サンプルのパネルデータを生成（DID分析用）

    Returns
    -------
    pd.DataFrame
        パネルデータ
    """
    np.random.seed(42)

    # 50人のメンバー、4年分のデータ
    member_ids = np.repeat(np.arange(50), 4)
    years = np.tile(np.arange(2020, 2024), 50)

    # 25人が2022年にスキルを取得
    treatment_group = member_ids < 25
    post_period = years >= 2022
    has_skill = treatment_group & post_period

    # DID効果 = 0.3
    baseline = np.random.binomial(1, 0.3, len(member_ids))
    treatment_effect = 0.3 * has_skill
    is_excellent = (baseline + treatment_effect + np.random.randn(len(member_ids)) * 0.1 > 0.5).astype(int)

    df = pd.DataFrame({
        'member_id': member_ids,
        'year': years,
        'has_skill': has_skill.astype(int),
        'is_excellent': is_excellent,
        'treated_group': treatment_group.astype(int)
    })

    return df


@pytest.fixture
def sample_propensity_scores() -> Tuple[np.ndarray, np.ndarray]:
    """
    サンプルの傾向スコアを生成

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (ps_treated, ps_control)
    """
    np.random.seed(42)
    ps_treated = np.random.beta(5, 2, 50)
    ps_control = np.random.beta(2, 5, 100)
    return ps_treated, ps_control


@pytest.fixture
def sample_covariates() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    サンプルの共変量データを生成（マッチング前後）

    Returns
    -------
    Tuple
        (X_treated_before, X_control_before, X_treated_after, X_control_after)
    """
    np.random.seed(42)

    # マッチング前
    X_treated_before = np.random.randn(50, 3) + np.array([1.0, 0.5, -0.3])
    X_control_before = np.random.randn(100, 3)

    # マッチング後（バランスが改善）
    X_treated_after = np.random.randn(40, 3) + np.array([0.2, 0.1, -0.1])
    X_control_after = np.random.randn(40, 3)

    return X_treated_before, X_control_before, X_treated_after, X_control_after
