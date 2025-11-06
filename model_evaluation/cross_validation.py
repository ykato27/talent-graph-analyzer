"""
クロスバリデーションモジュール

モデルの汎化性能を評価するためのCV手法を提供します。

主な機能：
1. Stratified K-Fold CV
2. Leave-One-Out CV（少数サンプル用）
3. Nested CV（ハイパーパラメータ調整付き）
4. CV結果のサマリー
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, LeaveOneOut, cross_val_score,
    cross_validate
)
from sklearn.metrics import make_scorer, roc_auc_score
from typing import Dict, List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def stratified_cv(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    metrics: Optional[List[str]] = None,
    return_predictions: bool = False
) -> Dict:
    """
    層化K分割交差検証

    クラス比率を保ったままデータを分割して評価します。
    不均衡データに適しています。

    Parameters
    ----------
    model : object
        scikit-learn互換のモデル
    X : np.ndarray
        特徴量
    y : np.ndarray
        ラベル
    n_splits : int
        分割数（デフォルト: 5）
    metrics : List[str], optional
        評価指標のリスト: ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    return_predictions : bool
        CVの予測値を返すかどうか

    Returns
    -------
    Dict
        'scores': 各指標のスコア
        'mean_scores': 各指標の平均
        'std_scores': 各指標の標準偏差
        'predictions': CVの予測値（return_predictions=Trueの場合）

    Notes
    -----
    層化CVの利点:
    - クラス比率を各foldで保持
    - 不均衡データでも安定した評価
    - 小サンプルでもバイアスが少ない

    推奨される使用法:
    - サンプル数 >= 50: 5-fold or 10-fold
    - サンプル数 < 50: 3-fold or LOOCV

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> model = LogisticRegression()
    >>> results = stratified_cv(model, X, y, n_splits=5)
    >>> print(f"Mean AUC: {results['mean_scores']['roc_auc']:.3f}")
    """

    logger.info(f"Running Stratified {n_splits}-Fold CV on {len(X)} samples")

    # デフォルトの評価指標
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # スコアリング関数
    scoring = {metric: metric for metric in metrics}

    # クロスバリデーション実行
    cv_results = cross_validate(
        model, X, y,
        cv=skf,
        scoring=scoring,
        return_train_score=True,
        return_estimator=return_predictions
    )

    # 結果を整理
    results = {
        'scores': {},
        'mean_scores': {},
        'std_scores': {},
        'train_scores': {},
        'mean_train_scores': {},
        'std_train_scores': {}
    }

    for metric in metrics:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']

        results['scores'][metric] = test_scores
        results['mean_scores'][metric] = np.mean(test_scores)
        results['std_scores'][metric] = np.std(test_scores)

        results['train_scores'][metric] = train_scores
        results['mean_train_scores'][metric] = np.mean(train_scores)
        results['std_train_scores'][metric] = np.std(train_scores)

        logger.info(f"  {metric}: {results['mean_scores'][metric]:.4f} ± {results['std_scores'][metric]:.4f}")

    # 予測値を返す場合
    if return_predictions:
        predictions = np.zeros(len(y))
        probabilities = np.zeros(len(y))

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            estimator = cv_results['estimator'][fold_idx]
            predictions[test_idx] = estimator.predict(X[test_idx])

            if hasattr(estimator, 'predict_proba'):
                probabilities[test_idx] = estimator.predict_proba(X[test_idx])[:, 1]

        results['predictions'] = predictions
        results['probabilities'] = probabilities

    logger.info(f"Stratified CV completed - Mean AUC: {results['mean_scores'].get('roc_auc', np.nan):.4f}")

    return results


def leave_one_out_cv(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    metric: str = 'accuracy'
) -> Dict:
    """
    Leave-One-Out交差検証

    サンプルを1つずつ除外して評価します。
    少数サンプル（n<30）の場合に推奨されます。

    Parameters
    ----------
    model : object
        scikit-learn互換のモデル
    X : np.ndarray
        特徴量
    y : np.ndarray
        ラベル
    metric : str
        評価指標

    Returns
    -------
    Dict
        'scores': 各サンプルのスコア
        'mean_score': 平均スコア
        'accuracy': 正解率（分類の場合）

    Notes
    -----
    LOOCV の特徴:
    - ほぼ不偏な推定
    - 計算コストが高い（n回学習）
    - 分散が大きい
    - n < 30 の場合に推奨

    利点:
    - 全データを学習に使用（n-1個）
    - データの無駄がない

    欠点:
    - 計算時間がO(n)
    - 分散が大きい

    Examples
    --------
    >>> results = leave_one_out_cv(model, X, y)
    >>> print(f"LOOCV Accuracy: {results['accuracy']:.3f}")
    """

    n_samples = len(X)
    logger.info(f"Running Leave-One-Out CV on {n_samples} samples (this may take a while...)")

    # Leave-One-Out
    loo = LeaveOneOut()

    # 各サンプルの予測を記録
    predictions = np.zeros(n_samples)
    probabilities = np.zeros(n_samples)

    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # モデル学習
        model_clone = clone_model(model)
        model_clone.fit(X_train, y_train)

        # 予測
        predictions[test_idx] = model_clone.predict(X_test)

        if hasattr(model_clone, 'predict_proba'):
            probabilities[test_idx] = model_clone.predict_proba(X_test)[:, 1]

        if (fold_idx + 1) % 10 == 0:
            logger.info(f"  Completed {fold_idx + 1}/{n_samples} folds")

    # 精度計算
    accuracy = np.mean(predictions == y)

    # AUC計算（クラスが2つ以上ある場合）
    if len(np.unique(y)) > 1:
        auc = roc_auc_score(y, probabilities)
    else:
        auc = np.nan

    results = {
        'predictions': predictions,
        'probabilities': probabilities,
        'accuracy': accuracy,
        'auc': auc,
        'n_correct': np.sum(predictions == y),
        'n_incorrect': np.sum(predictions != y)
    }

    logger.info(f"LOOCV completed - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

    return results


def clone_model(model: object):
    """モデルのクローンを作成"""
    from sklearn.base import clone
    return clone(model)


def nested_cv(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict,
    n_outer_splits: int = 5,
    n_inner_splits: int = 3,
    metric: str = 'roc_auc'
) -> Dict:
    """
    Nested交差検証（ハイパーパラメータ調整付き）

    外側のループで汎化性能を評価し、内側のループでハイパーパラメータを選択します。

    Parameters
    ----------
    model : object
        scikit-learn互換のモデル
    X : np.ndarray
        特徴量
    y : np.ndarray
        ラベル
    param_grid : Dict
        ハイパーパラメータのグリッド
    n_outer_splits : int
        外側のfold数（汎化性能評価用）
    n_inner_splits : int
        内側のfold数（ハイパーパラメータ選択用）
    metric : str
        評価指標

    Returns
    -------
    Dict
        'outer_scores': 外側CVのスコア
        'mean_score': 平均スコア
        'std_score': 標準偏差
        'best_params_per_fold': 各foldの最適パラメータ

    Notes
    -----
    Nested CVの目的:
    - ハイパーパラメータ調整のバイアスを除去
    - 真の汎化性能を推定
    - データリークを防止

    構造:
    - 外側CV: 汎化性能の推定
    - 内側CV: ハイパーパラメータの選択

    注意:
    - 計算コストが高い（outer × inner × grid_size 回学習）
    - サンプル数が十分（n > 100）な場合に推奨

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier()
    >>> param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    >>> results = nested_cv(model, X, y, param_grid)
    >>> print(f"Nested CV Score: {results['mean_score']:.3f}")
    """

    from sklearn.model_selection import GridSearchCV

    logger.info(f"Running Nested CV - Outer: {n_outer_splits}, Inner: {n_inner_splits}")
    logger.info(f"Parameter grid: {param_grid}")

    # 外側のStratified K-Fold
    outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=42)

    outer_scores = []
    best_params_per_fold = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 内側のグリッドサーチCV
        inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            model, param_grid,
            cv=inner_cv,
            scoring=metric,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        # 最適パラメータで外側のテストデータを評価
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        if metric == 'roc_auc':
            y_proba = best_model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_proba)
        else:
            score = best_model.score(X_test, y_test)

        outer_scores.append(score)
        best_params_per_fold.append(best_params)

        logger.info(f"  Fold {fold_idx + 1}: Score={score:.4f}, Best params={best_params}")

    mean_score = np.mean(outer_scores)
    std_score = np.std(outer_scores)

    results = {
        'outer_scores': outer_scores,
        'mean_score': mean_score,
        'std_score': std_score,
        'best_params_per_fold': best_params_per_fold,
        'n_outer_splits': n_outer_splits,
        'n_inner_splits': n_inner_splits
    }

    logger.info(f"Nested CV completed - Mean {metric}: {mean_score:.4f} ± {std_score:.4f}")

    return results


def cv_performance_summary(
    cv_results: Dict,
    model_name: str = "Model"
) -> pd.DataFrame:
    """
    CV結果のサマリーを生成

    Parameters
    ----------
    cv_results : Dict
        stratified_cv()の結果
    model_name : str
        モデル名

    Returns
    -------
    pd.DataFrame
        サマリー表

    Examples
    --------
    >>> summary = cv_performance_summary(cv_results, "Random Forest")
    >>> print(summary)
    """

    summary_data = []

    for metric, mean_val in cv_results['mean_scores'].items():
        std_val = cv_results['std_scores'][metric]
        train_mean = cv_results['mean_train_scores'].get(metric, np.nan)
        train_std = cv_results['std_train_scores'].get(metric, np.nan)

        overfit = train_mean - mean_val if not np.isnan(train_mean) else np.nan

        summary_data.append({
            'Model': model_name,
            'Metric': metric.upper(),
            'Train_Mean': train_mean,
            'Train_Std': train_std,
            'Test_Mean': mean_val,
            'Test_Std': std_val,
            'Overfit': overfit
        })

    summary_df = pd.DataFrame(summary_data)

    return summary_df


def compare_cv_results(
    cv_results_dict: Dict[str, Dict],
    metric: str = 'roc_auc'
) -> pd.DataFrame:
    """
    複数モデルのCV結果を比較

    Parameters
    ----------
    cv_results_dict : Dict[str, Dict]
        {モデル名: CV結果} の辞書
    metric : str
        比較する評価指標

    Returns
    -------
    pd.DataFrame
        比較表

    Examples
    --------
    >>> cv_results_dict = {
    ...     'LR': lr_cv_results,
    ...     'RF': rf_cv_results,
    ...     'XGB': xgb_cv_results
    ... }
    >>> comparison = compare_cv_results(cv_results_dict, 'roc_auc')
    >>> print(comparison)
    """

    comparison_data = []

    for model_name, cv_results in cv_results_dict.items():
        mean_score = cv_results['mean_scores'].get(metric, np.nan)
        std_score = cv_results['std_scores'].get(metric, np.nan)
        train_mean = cv_results['mean_train_scores'].get(metric, np.nan)

        comparison_data.append({
            'Model': model_name,
            'Test_Mean': mean_score,
            'Test_Std': std_score,
            'Train_Mean': train_mean,
            'Overfit': train_mean - mean_score if not np.isnan(train_mean) else np.nan
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_Mean', ascending=False)
    comparison_df['Rank'] = range(1, len(comparison_df) + 1)

    return comparison_df
