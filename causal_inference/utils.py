"""
ユーティリティモジュール

共通のヘルパー関数を提供します。
"""

import time
import logging
from contextlib import contextmanager
from typing import Generator, Any
from functools import wraps


@contextmanager
def log_execution_time(
    logger: logging.Logger,
    operation_name: str,
    level: int = logging.INFO
) -> Generator[dict[str, Any], None, None]:
    """
    処理時間をログに記録するコンテキストマネージャー

    Parameters
    ----------
    logger : logging.Logger
        ロガーインスタンス
    operation_name : str
        操作名
    level : int
        ログレベル

    Yields
    ------
    dict
        メタデータを格納する辞書

    Examples
    --------
    >>> with log_execution_time(logger, "model training") as metadata:
    ...     train_model()
    ...     metadata['n_samples'] = 1000
    """
    metadata: dict[str, Any] = {}
    start_time = time.time()

    logger.log(level, f"Starting: {operation_name}")

    try:
        yield metadata
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"Failed: {operation_name} after {elapsed:.2f}s - {type(e).__name__}: {e}",
            exc_info=True
        )
        raise
    else:
        elapsed = time.time() - start_time
        metadata_str = ", ".join([f"{k}={v}" for k, v in metadata.items()])
        if metadata_str:
            logger.log(level, f"Completed: {operation_name} in {elapsed:.2f}s ({metadata_str})")
        else:
            logger.log(level, f"Completed: {operation_name} in {elapsed:.2f}s")


def timing_decorator(func):
    """
    関数の実行時間を計測するデコレーター

    Examples
    --------
    >>> @timing_decorator
    ... def slow_function():
    ...     time.sleep(1)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


def deprecated(reason: str):
    """
    非推奨の関数をマークするデコレーター

    Parameters
    ----------
    reason : str
        非推奨の理由

    Examples
    --------
    >>> @deprecated("Use new_function() instead")
    ... def old_function():
    ...     pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            logger.warning(
                f"{func.__name__} is deprecated: {reason}"
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    ゼロ除算を安全に処理する除算

    Parameters
    ----------
    numerator : float
        分子
    denominator : float
        分母
    default : float
        分母がゼロの場合の返り値

    Returns
    -------
    float
        除算結果またはデフォルト値
    """
    if denominator == 0 or denominator == 0.0:
        return default
    return numerator / denominator


def format_pvalue(p_value: float, threshold: float = 0.001) -> str:
    """
    p値を見やすくフォーマット

    Parameters
    ----------
    p_value : float
        p値
    threshold : float
        閾値（これより小さい場合は"<"表記）

    Returns
    -------
    str
        フォーマットされたp値

    Examples
    --------
    >>> format_pvalue(0.0001)
    'p < 0.001'
    >>> format_pvalue(0.045)
    'p = 0.045'
    """
    if p_value < threshold:
        return f"p < {threshold}"
    else:
        return f"p = {p_value:.3f}"


def format_ci(lower: float, upper: float, decimals: int = 3) -> str:
    """
    信頼区間を見やすくフォーマット

    Parameters
    ----------
    lower : float
        下限
    upper : float
        上限
    decimals : int
        小数点以下の桁数

    Returns
    -------
    str
        フォーマットされた信頼区間

    Examples
    --------
    >>> format_ci(0.123, 0.456)
    '[0.123, 0.456]'
    """
    fmt = f"{{:.{decimals}f}}"
    return f"[{fmt.format(lower)}, {fmt.format(upper)}]"


def get_significance_stars(p_value: float) -> str:
    """
    p値から有意性マーカーを取得

    Parameters
    ----------
    p_value : float
        p値

    Returns
    -------
    str
        有意性マーカー

    Examples
    --------
    >>> get_significance_stars(0.0001)
    '***'
    >>> get_significance_stars(0.02)
    '*'
    >>> get_significance_stars(0.1)
    ''
    """
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
