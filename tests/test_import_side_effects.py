"""import 時の副作用に関する回帰テスト。

gnn_talent_analyzer は import 時に setup_logging() を実行するが、
ログディレクトリ（./logs）を import しただけで作ってはいけない
（初回ログ出力時に作るのが正しい）。
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_import_does_not_create_logs_dir(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    # pytest-cov の子プロセス向け環境変数を引き継ぐと、子プロセスが
    # カバレッジデータを書き出して本体の combine と競合するため除去する
    env = {
        k: v
        for k, v in env.items()
        if not k.startswith("COVERAGE") and not k.startswith("COV_CORE")
    }
    result = subprocess.run(
        [sys.executable, "-c", "import gnn_talent_analyzer"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "logs").exists()
