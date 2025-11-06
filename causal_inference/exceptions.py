"""
カスタム例外クラス

因果推論モジュール用の例外階層を定義します。
"""


class CausalInferenceError(Exception):
    """因果推論モジュールの基底例外クラス"""
    pass


class InvalidInputError(CausalInferenceError):
    """入力データの検証エラー"""
    pass


class InsufficientDataError(CausalInferenceError):
    """データ不足エラー"""
    pass


class ConvergenceError(CausalInferenceError):
    """収束失敗エラー"""
    pass


class ConfigurationError(CausalInferenceError):
    """設定エラー"""
    pass


class MatchingError(CausalInferenceError):
    """マッチング失敗エラー"""
    pass


class EstimationError(CausalInferenceError):
    """推定失敗エラー"""
    pass
