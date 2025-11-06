"""
評価指標モジュール

モデルの性能を多角的に評価する指標を提供します。
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, brier_score_loss
)
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
    """
    全ての評価指標を計算

    Parameters
    ----------
    y_true : np.ndarray
        真のラベル
    y_pred : np.ndarray
        予測ラベル
    y_proba : np.ndarray
        予測確率

    Returns
    -------
    Dict
        全評価指標
    """
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true")
        return {'error': 'Single class'}

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba),
        'brier_score': brier_score_loss(y_true, y_proba)
    }


def calibration_curve(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> Tuple:
    """
    キャリブレーションカーブを計算

    Parameters
    ----------
    y_true : np.ndarray
        真のラベル
    y_proba : np.ndarray
        予測確率
    n_bins : int
        ビン数

    Returns
    -------
    Tuple
        (bin_centers, observed_frequencies, bin_counts)
    """
    from sklearn.calibration import calibration_curve as sklearn_calibration_curve
    return sklearn_calibration_curve(y_true, y_proba, n_bins=n_bins)


def roc_analysis(y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
    """
    ROC曲線の分析

    Returns
    -------
    Dict
        'fpr', 'tpr', 'thresholds', 'auc'
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    # 最適閾値（Youden index）
    youden_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[youden_idx]

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': auc,
        'optimal_threshold': optimal_threshold,
        'optimal_tpr': tpr[youden_idx],
        'optimal_fpr': fpr[youden_idx]
    }


def confusion_matrix_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    混同行列の分析

    Returns
    -------
    Dict
        TN, FP, FN, TP, sensitivity, specificity, etc.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0
    }
