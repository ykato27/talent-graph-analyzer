"""
設定ファイル読み込みユーティリティ
"""

import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Any

# ロギング設定
logger = logging.getLogger('ConfigLoader')


class ConfigLoader:
    """設定ファイルを読み込んで管理するクラス"""

    _instance = None
    _config = None

    def __new__(cls):
        """シングルトンパターン"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初期化"""
        if self._config is None:
            self._load_config()

    def _load_config(self):
        """設定ファイルを読み込む

        Raises:
            FileNotFoundError: 設定ファイルが見つからない場合
            yaml.YAMLError: YAML解析に失敗した場合
            ValueError: 設定が無効な場合
        """
        config_path = Path(__file__).parent / "config.yaml"

        # ファイルの存在確認
        if not config_path.exists():
            logger.error(f"設定ファイルが見つかりません: {config_path}")
            raise FileNotFoundError(
                f"設定ファイルが見つかりません: {config_path}\n"
                f"expected path: {config_path.absolute()}"
            )

        # ファイルの読み込みと解析
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)

                # 設定が None またはリスト（不正な形式）でないか確認
                if self._config is None:
                    logger.warning(f"設定ファイルが空です: {config_path}")
                    self._config = {}
                elif not isinstance(self._config, dict):
                    logger.error(f"設定ファイルの形式が無効です（辞書である必要があります）")
                    raise ValueError(
                        f"設定ファイルは YAML 辞書である必要があります。"
                        f"受け取った型: {type(self._config).__name__}"
                    )

                logger.info(f"設定ファイルを読み込みました: {config_path}")

        except yaml.YAMLError as e:
            logger.error(f"YAML解析エラー: {e}", exc_info=True)
            raise yaml.YAMLError(
                f"設定ファイルのYAML形式が無効です: {config_path}\n"
                f"エラー: {e}"
            ) from e
        except IOError as e:
            logger.error(f"ファイル読み込みエラー: {e}", exc_info=True)
            raise IOError(
                f"設定ファイルの読み込みに失敗しました: {config_path}\n"
                f"エラー: {e}"
            ) from e

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        ドット記法で設定値を取得

        Parameters:
        -----------
        key_path: str
            設定のキーパス (例: "model.hidden_dim")
        default: Any
            デフォルト値

        Returns:
        --------
        value: Any
            設定値
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_all(self) -> Dict[str, Any]:
        """すべての設定を取得"""
        return self._config.copy()

    def reload(self):
        """設定ファイルを再読み込み"""
        self._config = None
        self._load_config()


# グローバルインスタンス
config = ConfigLoader()


def get_config(key_path: str = None, default: Any = None) -> Any:
    """
    設定値を取得する便利関数

    Parameters:
    -----------
    key_path: str, optional
        設定のキーパス。Noneの場合は全設定を返す
    default: Any
        デフォルト値

    Returns:
    --------
    value: Any
        設定値
    """
    if key_path is None:
        return config.get_all()
    return config.get(key_path, default)
