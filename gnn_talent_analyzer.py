"""旧モジュール名の移行用入口。

v3では talent_graph.data / analysis / graph を直接使用してください。
旧Streamlit用TalentAnalyzer・疑似HTE APIは廃止しました。
"""

from talent_graph.analysis import AnalysisResult, analyze
from talent_graph.data import DataValidationError, SkillnoteData, load_directory, prepare_data
from talent_graph.graph import GraphEncoder

__all__ = [
    'AnalysisResult', 'DataValidationError', 'SkillnoteData', 'GraphEncoder',
    'analyze', 'load_directory', 'prepare_data',
]
