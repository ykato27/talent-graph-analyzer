"""
ベースラインモデルモジュール

GNNと比較するためのベースラインモデルを提供します。

主な機能：
1. Logistic Regression
2. Random Forest
3. XGBoost
4. モデル比較フレームワーク
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LogisticRegressionBaseline:
    """
    ロジスティック回帰ベースライン

    最もシンプルな線形モデル。解釈性が高く、少数サンプルでも安定。
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs : dict
            sklearn.LogisticRegressionのパラメータ
        """
        default_params = {
            'max_iter': 1000,
            'random_state': 42,
            'class_weight': 'balanced'  # 不均衡データ対策
        }
        default_params.update(kwargs)

        self.model = LogisticRegression(**default_params)
        self.feature_names = None
        logger.info(f"Initialized Logistic Regression with params: {default_params}")

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        モデルを学習

        Parameters
        ----------
        X : np.ndarray
            特徴量行列
        y : np.ndarray
            ラベル（0 or 1）
        feature_names : List[str], optional
            特徴量の名前
        """
        self.feature_names = feature_names
        logger.info(f"Training Logistic Regression on {len(X)} samples")

        self.model.fit(X, y)

        # 学習データでの性能
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        train_acc = accuracy_score(y, y_pred)
        train_auc = roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else np.nan

        logger.info(f"Training completed - Accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測（0 or 1）"""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """確率予測（0-1）"""
        return self.model.predict_proba(X)[:, 1]

    def get_coefficients(self) -> pd.DataFrame:
        """
        回帰係数を取得（解釈性）

        Returns
        -------
        pd.DataFrame
            特徴量と係数、重要度
        """
        if self.feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(len(self.model.coef_[0]))]

        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_[0],
            'Abs_Coefficient': np.abs(self.model.coef_[0])
        }).sort_values('Abs_Coefficient', ascending=False)

        return coef_df


class RandomForestBaseline:
    """
    ランダムフォレストベースライン

    非線形関係を捉え、特徴量の重要度を提供。過学習に強い。
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs : dict
            sklearn.RandomForestClassifierのパラメータ
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1
        }
        default_params.update(kwargs)

        self.model = RandomForestClassifier(**default_params)
        self.feature_names = None
        logger.info(f"Initialized Random Forest with params: {default_params}")

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """モデルを学習"""
        self.feature_names = feature_names
        logger.info(f"Training Random Forest on {len(X)} samples")

        self.model.fit(X, y)

        # 学習データでの性能
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        train_acc = accuracy_score(y, y_pred)
        train_auc = roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else np.nan

        logger.info(f"Training completed - Accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測（0 or 1）"""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """確率予測（0-1）"""
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> pd.DataFrame:
        """
        特徴量の重要度を取得

        Returns
        -------
        pd.DataFrame
            特徴量と重要度
        """
        if self.feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(len(self.model.feature_importances_))]

        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        return importance_df


class XGBoostBaseline:
    """
    XGBoostベースライン

    勾配ブースティング。競技では最強クラスの性能。
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs : dict
            xgboost.XGBClassifierのパラメータ
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("xgboost not installed. Install with: pip install xgboost")
            raise ImportError("xgboost is required")

        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        default_params.update(kwargs)

        self.model = xgb.XGBClassifier(**default_params)
        self.feature_names = None
        logger.info(f"Initialized XGBoost with params: {default_params}")

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """モデルを学習"""
        self.feature_names = feature_names
        logger.info(f"Training XGBoost on {len(X)} samples")

        # クラス不均衡の処理
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

        self.model.set_params(scale_pos_weight=scale_pos_weight)

        self.model.fit(X, y)

        # 学習データでの性能
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        train_acc = accuracy_score(y, y_pred)
        train_auc = roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else np.nan

        logger.info(f"Training completed - Accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測（0 or 1）"""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """確率予測（0-1）"""
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> pd.DataFrame:
        """特徴量の重要度を取得"""
        if self.feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(len(self.model.feature_importances_))]

        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        return importance_df


def compare_models(
    models: Dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    複数のモデルを比較

    Parameters
    ----------
    models : Dict[str, object]
        モデル名とモデルオブジェクトの辞書
    X_train : np.ndarray
        学習用特徴量
    y_train : np.ndarray
        学習用ラベル
    X_test : np.ndarray
        テスト用特徴量
    y_test : np.ndarray
        テスト用ラベル
    feature_names : List[str], optional
        特徴量の名前

    Returns
    -------
    pd.DataFrame
        モデル比較結果

    Examples
    --------
    >>> models = {
    ...     'Logistic Regression': LogisticRegressionBaseline(),
    ...     'Random Forest': RandomForestBaseline(),
    ...     'XGBoost': XGBoostBaseline()
    ... }
    >>> comparison = compare_models(models, X_train, y_train, X_test, y_test)
    >>> print(comparison)
    """

    logger.info(f"Comparing {len(models)} models")

    results = []

    for model_name, model in models.items():
        logger.info(f"Evaluating model: {model_name}")

        # 学習
        model.fit(X_train, y_train, feature_names)

        # 学習データでの評価
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)

        train_metrics = _calculate_metrics(y_train, y_train_pred, y_train_proba)

        # テストデータでの評価
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)

        test_metrics = _calculate_metrics(y_test, y_test_pred, y_test_proba)

        # 過学習度合い
        overfit_auc = train_metrics['AUC'] - test_metrics['AUC']
        overfit_f1 = train_metrics['F1'] - test_metrics['F1']

        results.append({
            'Model': model_name,
            'Train_AUC': train_metrics['AUC'],
            'Test_AUC': test_metrics['AUC'],
            'Train_Accuracy': train_metrics['Accuracy'],
            'Test_Accuracy': test_metrics['Accuracy'],
            'Train_Precision': train_metrics['Precision'],
            'Test_Precision': test_metrics['Precision'],
            'Train_Recall': train_metrics['Recall'],
            'Test_Recall': test_metrics['Recall'],
            'Train_F1': train_metrics['F1'],
            'Test_F1': test_metrics['F1'],
            'Overfit_AUC': overfit_auc,
            'Overfit_F1': overfit_f1
        })

        logger.info(f"  {model_name} - Test AUC: {test_metrics['AUC']:.4f}, "
                    f"Test F1: {test_metrics['F1']:.4f}, "
                    f"Overfit (AUC): {overfit_auc:.4f}")

    comparison_df = pd.DataFrame(results)

    # ランキング
    comparison_df['Rank_Test_AUC'] = comparison_df['Test_AUC'].rank(ascending=False)
    comparison_df['Rank_Test_F1'] = comparison_df['Test_F1'].rank(ascending=False)

    # ベストモデル
    best_model = comparison_df.loc[comparison_df['Test_AUC'].idxmax(), 'Model']
    logger.info(f"Best model by Test AUC: {best_model}")

    return comparison_df


def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
    """評価指標を計算"""

    # ラベルが1クラスしかない場合の処理
    if len(np.unique(y_true)) < 2:
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': np.nan,
            'Recall': np.nan,
            'F1': np.nan,
            'AUC': np.nan
        }

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_true, y_proba)
    }

    return metrics


def benchmark_against_gnn(
    gnn_predictions: np.ndarray,
    gnn_probabilities: np.ndarray,
    baseline_models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    GNNとベースラインモデルをベンチマーク

    Parameters
    ----------
    gnn_predictions : np.ndarray
        GNNの予測（0 or 1）
    gnn_probabilities : np.ndarray
        GNNの確率予測
    baseline_models : Dict[str, object]
        学習済みベースラインモデル
    X_test : np.ndarray
        テストデータ特徴量
    y_test : np.ndarray
        テストデータラベル

    Returns
    -------
    pd.DataFrame
        ベンチマーク結果（GNN vs ベースライン）

    Examples
    --------
    >>> benchmark = benchmark_against_gnn(
    ...     gnn_pred, gnn_proba,
    ...     {'LR': lr_model, 'RF': rf_model},
    ...     X_test, y_test
    ... )
    """

    logger.info("Benchmarking GNN against baseline models")

    # GNNの評価
    gnn_metrics = _calculate_metrics(y_test, gnn_predictions, gnn_probabilities)

    results = [{
        'Model': 'GNN',
        'AUC': gnn_metrics['AUC'],
        'Accuracy': gnn_metrics['Accuracy'],
        'Precision': gnn_metrics['Precision'],
        'Recall': gnn_metrics['Recall'],
        'F1': gnn_metrics['F1']
    }]

    # ベースラインモデルの評価
    for model_name, model in baseline_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = _calculate_metrics(y_test, y_pred, y_proba)

        results.append({
            'Model': model_name,
            'AUC': metrics['AUC'],
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1': metrics['F1']
        })

    benchmark_df = pd.DataFrame(results)

    # GNNとの差分
    gnn_auc = gnn_metrics['AUC']
    benchmark_df['AUC_vs_GNN'] = benchmark_df['AUC'] - gnn_auc
    benchmark_df['F1_vs_GNN'] = benchmark_df['F1'] - gnn_metrics['F1']

    # ランキング
    benchmark_df['Rank_AUC'] = benchmark_df['AUC'].rank(ascending=False)

    logger.info(f"GNN AUC: {gnn_auc:.4f}")
    for _, row in benchmark_df.iterrows():
        if row['Model'] != 'GNN':
            logger.info(f"  {row['Model']} AUC: {row['AUC']:.4f} (vs GNN: {row['AUC_vs_GNN']:+.4f})")

    return benchmark_df
