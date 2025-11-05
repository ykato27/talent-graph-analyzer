"""
talent_analyzer パッケージ

GNN優秀人材分析システムのメインパッケージ
"""

from .core.analyzer import TalentAnalyzer
from .core.gnn_models import SimpleGNN

__all__ = ['TalentAnalyzer', 'SimpleGNN']
