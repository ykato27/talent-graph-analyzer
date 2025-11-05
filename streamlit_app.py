#!/usr/bin/env python
"""
GNN優秀人材分析システム - Streamlit Cloud 用エントリーポイント

このファイルは Streamlit Cloud がアプリケーションを実行するときに
最初に読み込むファイルです。
"""

import sys
from pathlib import Path

# src/ ディレクトリを Python パスに追加
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# app.py をインポートして実行
from talent_analyzer.ui.app import *
