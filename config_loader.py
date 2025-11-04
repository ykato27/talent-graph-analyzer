"""
設定ファイル読み込みユーティリティ
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


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
        """設定ファイルを読み込む"""
        config_path = Path(__file__).parent / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

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
