"""
定数定義モジュール

GNNベース優秀人材分析システムで使用されるすべての定数を管理します。
config.yaml から設定値を読み込み、デフォルト値をここで定義します。
"""

from config_loader import get_config


class ModelConfig:
    """GNNモデルのパラメータ"""

    # デフォルト値
    DEFAULT_N_LAYERS = 3
    DEFAULT_HIDDEN_DIM = 128
    DEFAULT_OUTPUT_DIM = 128
    DEFAULT_DROPOUT = 0.3
    DEFAULT_LEARNING_RATE = 0.01
    DEFAULT_K_NEIGHBORS = 10

    # config.yamlから読み込み
    @staticmethod
    def get_n_layers():
        return get_config('model.n_layers', ModelConfig.DEFAULT_N_LAYERS)

    @staticmethod
    def get_hidden_dim():
        return get_config('model.hidden_dim', ModelConfig.DEFAULT_HIDDEN_DIM)

    @staticmethod
    def get_dropout():
        return get_config('model.dropout', ModelConfig.DEFAULT_DROPOUT)

    @staticmethod
    def get_learning_rate():
        return get_config('model.learning_rate', ModelConfig.DEFAULT_LEARNING_RATE)

    @staticmethod
    def get_k_neighbors():
        return get_config('model.k_neighbors', ModelConfig.DEFAULT_K_NEIGHBORS)


class TrainingConfig:
    """学習パラメータ"""

    # デフォルト値
    DEFAULT_EPOCHS = 100
    DEFAULT_EARLY_STOPPING_PATIENCE = 20
    DEFAULT_BATCH_SIZE_EDGES = 1000

    # config.yamlから読み込み
    @staticmethod
    def get_epochs():
        return get_config('training.default_epochs', TrainingConfig.DEFAULT_EPOCHS)

    @staticmethod
    def get_early_stopping_patience():
        return get_config('training.early_stopping_patience', TrainingConfig.DEFAULT_EARLY_STOPPING_PATIENCE)

    @staticmethod
    def get_batch_size_edges():
        return get_config('training.batch_size_edges', TrainingConfig.DEFAULT_BATCH_SIZE_EDGES)


class StatisticalConfig:
    """統計的検定パラメータ"""

    # デフォルト値
    DEFAULT_SIGNIFICANCE_LEVEL = 0.05
    DEFAULT_CONFIDENCE_LEVEL = 0.95

    # config.yamlから読み込み
    @staticmethod
    def get_significance_level():
        return get_config('statistical_tests.significance_level', StatisticalConfig.DEFAULT_SIGNIFICANCE_LEVEL)

    @staticmethod
    def get_confidence_level():
        return get_config('statistical_tests.confidence_level', StatisticalConfig.DEFAULT_CONFIDENCE_LEVEL)

    @staticmethod
    def get_correction_method():
        return get_config('statistical_tests.multiple_testing_correction', 'fdr_bh')


class CausalInferenceConfig:
    """因果推論パラメータ"""

    # デフォルト値
    DEFAULT_PROPENSITY_CALIPER = 0.1
    DEFAULT_MIN_MATCHED_PAIRS = 5

    # config.yamlから読み込み
    @staticmethod
    def get_propensity_caliper():
        return get_config('causal_inference.propensity_score.caliper', CausalInferenceConfig.DEFAULT_PROPENSITY_CALIPER)

    @staticmethod
    def get_min_matched_pairs():
        return get_config('causal_inference.propensity_score.min_matched_pairs', CausalInferenceConfig.DEFAULT_MIN_MATCHED_PAIRS)


class SkillInteractionConfig:
    """スキル相互作用パラメータ"""

    # デフォルト値
    DEFAULT_SYNERGY_THRESHOLD = 0.1
    DEFAULT_MIN_BOTH_RATE = 0.7
    DEFAULT_MIN_GROUP_SIZE = 3
    DEFAULT_MIN_SKILL_SAMPLES = 5

    # config.yamlから読み込み
    @staticmethod
    def get_synergy_threshold():
        return get_config('skill_interaction.synergy_threshold', SkillInteractionConfig.DEFAULT_SYNERGY_THRESHOLD)

    @staticmethod
    def get_min_both_rate():
        return get_config('skill_interaction.min_both_rate', SkillInteractionConfig.DEFAULT_MIN_BOTH_RATE)

    @staticmethod
    def get_min_group_size():
        return get_config('skill_interaction.min_group_size', SkillInteractionConfig.DEFAULT_MIN_GROUP_SIZE)

    @staticmethod
    def get_min_skill_samples():
        return get_config('skill_interaction.min_skill_samples', SkillInteractionConfig.DEFAULT_MIN_SKILL_SAMPLES)


class NumericalConfig:
    """数値計算パラメータ"""

    # デフォルト値
    DEFAULT_EPSILON = 1e-8

    # config.yamlから読み込み
    @staticmethod
    def get_epsilon():
        return get_config('numerical.epsilon', NumericalConfig.DEFAULT_EPSILON)


class AnalysisConfig:
    """分析パラメータ"""

    @staticmethod
    def get_min_excellent_members():
        return get_config('analysis.min_excellent_members', 3)

    @staticmethod
    def get_max_excellent_members():
        return get_config('analysis.max_excellent_members_recommended', 20)

    @staticmethod
    def get_essential_skill_threshold():
        return get_config('analysis.essential_skill_threshold', 0.8)

    @staticmethod
    def get_important_skill_diff_threshold():
        return get_config('analysis.important_skill_diff_threshold', 0.3)
