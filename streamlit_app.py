#!/usr/bin/env python
"""
GNN優秀人材分析システム - Streamlit Cloud 用エントリーポイント

このファイルは Streamlit Cloud がアプリケーションを実行するときに
最初に読み込むファイルです。
"""

import sys
import os
from pathlib import Path

# プロジェクトルートを取得
project_root = Path(__file__).parent
src_dir = project_root / "src"
config_dir = project_root / "config"
config_file = config_dir / "config.yaml"

# sys.path に追加
sys.path.insert(0, str(src_dir))

# 環境変数で config path を設定（loader.py が使用）
os.environ['TALENT_ANALYZER_CONFIG'] = str(config_file)

# app.py をインポートして実行
from talent_analyzer.ui.app import *
