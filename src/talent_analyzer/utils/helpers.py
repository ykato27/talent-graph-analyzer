"""
ユーティリティ関数モジュール

ロギング、ファイルI/O、データ処理など、システム全体で使用される
補助関数を提供します。
"""

import logging
import pandas as pd
from pathlib import Path
from logging.handlers import RotatingFileHandler
from ..config.loader import get_config


def setup_logging():
    """
    ロギングの設定を行う

    Returns:
    --------
    logger: logging.Logger
        設定済みのロガーインスタンス
    """
    log_config = get_config('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # ロガーの設定
    logger = logging.getLogger('TalentAnalyzer')
    logger.setLevel(log_level)

    # 既存のハンドラを削除（重複登録を防止）
    logger.handlers.clear()

    # コンソールハンドラ
    if log_config.get('console_logging', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)

    # ファイルハンドラ
    if log_config.get('file_logging', True):
        log_dir = Path(get_config('versioning.log_dir', './logs'))
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / log_config.get('log_file', 'talent_analyzer.log')

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=log_config.get('max_bytes', 10485760),
            backupCount=log_config.get('backup_count', 5)
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    return logger


def load_csv_files(member_path, acquired_path, skill_path, education_path, license_path):
    """
    5つのCSVファイルを読み込む

    Parameters:
    -----------
    member_path: str
        社員マスタファイルのパス
    acquired_path: str
        スキル習得データファイルのパス
    skill_path: str
        スキルマスタファイルのパス
    education_path: str
        教育マスタファイルのパス
    license_path: str
        資格マスタファイルのパス

    Returns:
    --------
    tuple of pd.DataFrame
        (member_df, acquired_df, skill_df, education_df, license_df)

    Raises:
    -------
    FileNotFoundError
        指定されたファイルが見つからない場合
    pd.errors.EmptyDataError
        ファイルが空の場合
    """
    encoding = get_config('files.encoding', 'utf-8-sig')
    logger = logging.getLogger('TalentAnalyzer')

    try:
        logger.info(f"Loading CSV files...")
        member_df = pd.read_csv(member_path, encoding=encoding)
        acquired_df = pd.read_csv(acquired_path, encoding=encoding)
        skill_df = pd.read_csv(skill_path, encoding=encoding)
        education_df = pd.read_csv(education_path, encoding=encoding)
        license_df = pd.read_csv(license_path, encoding=encoding)

        logger.info(f"CSV files loaded successfully")
        logger.debug(f"  member: {len(member_df)} rows")
        logger.debug(f"  acquired: {len(acquired_df)} rows")
        logger.debug(f"  skill: {len(skill_df)} rows")
        logger.debug(f"  education: {len(education_df)} rows")
        logger.debug(f"  license: {len(license_df)} rows")

        return member_df, acquired_df, skill_df, education_df, license_df

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty CSV file: {e}")
        raise


def format_time(seconds):
    """
    秒数を人間が読める形式にフォーマット

    Parameters:
    -----------
    seconds: float
        秒数

    Returns:
    --------
    str
        フォーマット済み時間文字列（例: "1.5秒", "2.3分", "1.2時"）
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分"
    else:
        return f"{seconds/3600:.1f}時"


def safe_divide(numerator, denominator, default=0):
    """
    ゼロ除算を防いで除算を実行

    Parameters:
    -----------
    numerator: float
        分子
    denominator: float
        分母
    default: float
        分母がゼロの場合のデフォルト値

    Returns:
    --------
    float
        計算結果またはデフォルト値
    """
    from constants import NumericalConfig
    epsilon = NumericalConfig.get_epsilon()

    if abs(denominator) < epsilon:
        return default
    return numerator / denominator


def ensure_dir_exists(dir_path):
    """
    ディレクトリが存在することを確認、ない場合は作成

    Parameters:
    -----------
    dir_path: str or Path
        ディレクトリパス

    Returns:
    --------
    Path
        ディレクトリパスオブジェクト
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logger(name='TalentAnalyzer'):
    """
    ロガーを取得

    Parameters:
    -----------
    name: str
        ロガー名

    Returns:
    --------
    logging.Logger
        ロガーインスタンス
    """
    return logging.getLogger(name)
