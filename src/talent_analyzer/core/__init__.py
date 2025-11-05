"""
core モジュール

GNNモデルと分析エンジンの実装
"""

from .analyzer import TalentAnalyzer
from .gnn_models import SimpleGNN

__all__ = ['TalentAnalyzer', 'SimpleGNN']
