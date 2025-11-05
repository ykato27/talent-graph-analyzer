#!/usr/bin/env python
"""
GNN優秀人材分析システム - Streamlit エントリーポイント

このスクリプトは Streamlit アプリケーションを起動します。
使用方法: streamlit run run.py
または: python run.py (streamlit コマンドが PATH にない場合)
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Streamlit アプリケーションを起動"""
    # プロジェクトルートを取得
    project_root = Path(__file__).parent
    app_path = project_root / "src" / "talent_analyzer" / "ui" / "app.py"

    if not app_path.exists():
        print(f"エラー: app.py が見つかりません: {app_path}")
        sys.exit(1)

    # Streamlit を起動
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            cwd=str(project_root)
        )
    except FileNotFoundError:
        print("エラー: streamlit がインストールされていません")
        print("インストール: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
