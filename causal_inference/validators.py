"""
入力検証モジュール

データの妥当性を検証する関数を提供します。
"""

import numpy as np
from typing import Any
from .exceptions import InvalidInputError, InsufficientDataError


def validate_array_lengths(*arrays: np.ndarray, names: list[str] | None = None) -> None:
    """
    配列の長さが一致することを検証

    Parameters
    ----------
    *arrays : np.ndarray
        検証する配列
    names : list[str], optional
        配列の名前（エラーメッセージ用）

    Raises
    ------
    InvalidInputError
        配列の長さが一致しない場合
    """
    if not arrays:
        return

    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        if names:
            length_info = ", ".join([f"{name}={length}" for name, length in zip(names, lengths)])
        else:
            length_info = ", ".join([str(length) for length in lengths])

        raise InvalidInputError(
            f"Array length mismatch: {length_info}. "
            f"All input arrays must have the same number of samples."
        )


def validate_positive_integer(value: int, name: str, min_value: int = 1) -> None:
    """
    正の整数であることを検証

    Parameters
    ----------
    value : int
        検証する値
    name : str
        パラメータ名
    min_value : int
        最小値

    Raises
    ------
    InvalidInputError
        負の値または最小値未満の場合
    """
    if not isinstance(value, (int, np.integer)):
        raise InvalidInputError(
            f"{name} must be an integer, got {type(value).__name__}"
        )

    if value < min_value:
        raise InvalidInputError(
            f"{name} must be >= {min_value}, got {value}"
        )


def validate_probability(value: float, name: str) -> None:
    """
    確率値（0-1の範囲）であることを検証

    Parameters
    ----------
    value : float
        検証する値
    name : str
        パラメータ名

    Raises
    ------
    InvalidInputError
        範囲外の値の場合
    """
    if not isinstance(value, (float, int, np.floating, np.integer)):
        raise InvalidInputError(
            f"{name} must be numeric, got {type(value).__name__}"
        )

    if not 0.0 <= value <= 1.0:
        raise InvalidInputError(
            f"{name} must be in [0, 1], got {value}"
        )


def validate_array_no_nan(array: np.ndarray, name: str) -> None:
    """
    配列にNaNが含まれないことを検証

    Parameters
    ----------
    array : np.ndarray
        検証する配列
    name : str
        配列名

    Raises
    ------
    InvalidInputError
        NaNが含まれる場合
    """
    if np.any(np.isnan(array)):
        n_nan = np.sum(np.isnan(array))
        raise InvalidInputError(
            f"{name} contains {n_nan} NaN value(s). "
            f"Please remove or impute missing values before analysis."
        )


def validate_array_no_inf(array: np.ndarray, name: str) -> None:
    """
    配列に無限大が含まれないことを検証

    Parameters
    ----------
    array : np.ndarray
        検証する配列
    name : str
        配列名

    Raises
    ------
    InvalidInputError
        無限大が含まれる場合
    """
    if np.any(np.isinf(array)):
        n_inf = np.sum(np.isinf(array))
        raise InvalidInputError(
            f"{name} contains {n_inf} infinite value(s). "
            f"Please check your data for numerical overflow."
        )


def validate_sufficient_data(
    n_samples: int,
    min_samples: int,
    context: str = "analysis"
) -> None:
    """
    十分なサンプル数があることを検証

    Parameters
    ----------
    n_samples : int
        サンプル数
    min_samples : int
        最小サンプル数
    context : str
        コンテキスト情報

    Raises
    ------
    InsufficientDataError
        サンプル数が不足している場合
    """
    if n_samples < min_samples:
        raise InsufficientDataError(
            f"Insufficient data for {context}: "
            f"got {n_samples} samples, need at least {min_samples}. "
            f"Please collect more data or use a different analysis method."
        )


def validate_binary_array(array: np.ndarray, name: str) -> None:
    """
    二値配列（0 or 1）であることを検証

    Parameters
    ----------
    array : np.ndarray
        検証する配列
    name : str
        配列名

    Raises
    ------
    InvalidInputError
        0と1以外の値が含まれる場合
    """
    unique_values = np.unique(array)
    valid_values = {0, 1, 0.0, 1.0}

    if not all(val in valid_values for val in unique_values):
        raise InvalidInputError(
            f"{name} must be binary (0 or 1), "
            f"but contains values: {sorted(unique_values)}"
        )


def validate_gamma_values(gamma_values: list[float]) -> None:
    """
    Gamma値のリストを検証

    Parameters
    ----------
    gamma_values : list[float]
        Gamma値のリスト

    Raises
    ------
    InvalidInputError
        不正な値が含まれる場合
    """
    if not gamma_values:
        raise InvalidInputError("gamma_values must not be empty")

    for gamma in gamma_values:
        if gamma < 1.0:
            raise InvalidInputError(
                f"All gamma values must be >= 1.0, got {gamma}. "
                f"Gamma represents the strength of hidden confounding."
            )


def validate_2d_array(array: np.ndarray, name: str) -> None:
    """
    2次元配列であることを検証

    Parameters
    ----------
    array : np.ndarray
        検証する配列
    name : str
        配列名

    Raises
    ------
    InvalidInputError
        2次元でない場合
    """
    if array.ndim != 2:
        raise InvalidInputError(
            f"{name} must be a 2D array, got {array.ndim}D. "
            f"Shape: {array.shape}"
        )


def validate_clusters(clusters: np.ndarray, min_clusters: int = 3) -> None:
    """
    クラスター配列を検証

    Parameters
    ----------
    clusters : np.ndarray
        クラスターID配列
    min_clusters : int
        最小クラスター数

    Raises
    ------
    InvalidInputError
        クラスター数が不足している場合
    """
    n_clusters = len(np.unique(clusters))

    if n_clusters < min_clusters:
        raise InvalidInputError(
            f"Insufficient number of clusters: got {n_clusters}, "
            f"need at least {min_clusters}. "
            f"Cluster-robust inference requires multiple clusters."
        )
