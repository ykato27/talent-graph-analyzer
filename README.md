# Talent Graph Analyzer

SkillnoteのCSVを読み込み、指定した社員群との力量の共通点をローカルで調べるPythonツールです。Streamlit、Webサーバー、GPU、外部API、認証情報は不要です。

Python **3.15.1**、実行・依存管理は **uv** を使用します。依存の正本は `pyproject.toml` と `uv.lock` です。

## 最初に試す

リポジトリ内で実行します。uvがPythonを取得できない場合は、公式のPython 3.15.1を導入してから `uv sync --python /path/to/python3.15` を実行してください。

```bash
uv sync --locked
uv run --locked talent-graph demo --output /tmp/talent-demo
uv run --locked talent-graph validate --data-dir /tmp/talent-demo
uv run --locked talent-graph analyze \
  --data-dir /tmp/talent-demo \
  --reference-file /tmp/talent-demo/reference.csv \
  --output /tmp/talent-results
```

デモは50名（退職者2名を除く在籍48名）・16力量の**架空データ**です。生成済みフォルダは上書きしません。再実行時は新しい出力先を指定してください。依存の初回取得後、上記コマンドの実行自体は外部へ通信しません。

## Skillnoteのデータで実行する

実データをGit管理外のフォルダに置きます。社員と習得データの2ファイルが必須で、力量名が習得データに含まれる場合、マスタは省略できます。

| ファイル | 内容 | 必須 |
|---|---|---|
| `member_skillnote.csv` または `member.csv` | 社員マスタ | はい |
| `acquiredCompetenceLevel.csv` または `acquired_competence.csv` | 力量習得データ | はい |
| `skill_skillnote.csv` または `skill.csv` | スキル名を補完 | 任意 |
| `education_skillnote.csv` または `education.csv` | 教育名を補完 | 任意 |
| `license_skillnote.csv` または `license.csv` | 資格名を補完 | 任意 |

同じ種類の別名ファイルを両方置くとエラーになります。1フォルダには1テナント・同じ時点のエクスポートを置いてください。社員マスタと習得データのコードは一致する必要があります。

### 列の形式

`メンバーコード  ###[member_code]###` のようなSkillnote注釈付き列名、通常の日本語列名、以下の英語列名を受け付けます。コードは文字列で読み、`0001` の先頭ゼロを保持します。

| CSV | 必須列（英語 / 日本語） | 任意列 |
|---|---|---|
| 社員 | `member_code` / メンバーコード、`name` / メンバー名 | 退職年月日、入社年月日、職能・等級、役職 |
| 習得 | `member_code` / メンバーコード、`competence_type` / 力量タイプ、`competence_code` / 力量コード | 力量名、レベル（SKILLでは必須） |
| 任意マスタ | `competence_code` / 力量コード、`competence_name` / 力量名 | その他は無視 |

- 力量タイプ: `SKILL` / `EDUCATION` / `LICENSE`、またはスキル / 教育 / 資格。
- スキルのレベル: 0以上の有限数値。0は未習得、空欄・不正値はエラーです。
- 教育・資格: 習得データに行があれば保有=1。有効期限切れや失効は判定しません。必要に応じて入力を絞ってください。
- 同じ社員・タイプ・力量コードの重複は最大レベルに集約します。履歴CSVではなく、現在の習得エクスポートを指定してください。
- 力量名は習得CSV、任意マスタ、力量コードの順で補完します。同じタイプ・コードに複数の名称がある場合はエラーです。
- 基準日時点で退職済み・入社前の社員は除外し、在籍中で習得記録がない社員は残します。未記録と未習得は区別できません。
- 社員マスタの重複、未知の社員コード、日付・レベル・タイプの不正は件数または列名を示して停止します。

### 基準群を指定する

比較の基準にする社員を、業務上の根拠に沿って選びます。保有スキル数だけで自動的に「優秀」と決める機能はありません。

```csv
member_code
0001
0002
0003
```

```bash
uv run --locked talent-graph validate --data-dir /path/to/skillnote-csv
uv run --locked talent-graph analyze \
  --data-dir /path/to/skillnote-csv \
  --reference-file /path/to/reference.csv \
  --as-of 2026-10-23 \
  --output /path/to/new-results
```

直接指定する場合は `--reference 0001 0002 0003` に置き換えます。比較群も最低1名必要です。Shift-JISの場合は `--encoding cp932` を指定します。`--as-of` を省略すると実行日の在籍状態を使います。入力フォルダの内側や既存フォルダへの出力は拒否します。

## 出力

| ファイル | 内容 |
|---|---|
| `report.md` | 読み方、入力品質、検証結果、制約 |
| `member_scores.csv` | 基準群とのコサイン類似度、保有力量数。記録がない社員は類似度が空欄 |
| `skill_importance.csv` | 全力量の保有率差、Wilson信頼区間、Fisher検定、FDR補正p値 |
| `run.json` | 基準日、入力のSHA-256、基準群、乱数種、依存版、学習損失、交差検証結果 |
| `embeddings.npz` | 社員コードと対応するグラフ埋め込み（pickle不使用） |

結果には個人情報が含まれます。入力・結果・仮想環境はコミットしないでください。依存の導入以外にこのツールがデータを外部送信する処理はありません。

## 分析方法と限界

社員をノード、力量ベクトルのコサイン類似度による近傍をエッジとする疎グラフを作ります。NumPyの2層GraphSAGEがエッジと非エッジを区別する二値交差エントロピーを解析的な逆伝播とAdamで最小化します。PyTorchの導入は不要です。

指定した基準群の平均埋め込みとのコサイン類似度を算出します。**優秀さの確率、昇格・採用の推奨、スキル習得の因果効果ではありません。** 元実装の疑似勾配や、結果を因果効果と見せる疑似HTEは使用しません。

- 精度は層化交差検証でAUCとAverage Precisionを計算し、ロジスティック回帰の基準モデルと比較します。基準群・比較群に記録のある社員が各2名未満の場合は「評価不可」とします。
- グラフはラベルを使わず全社員の力量で学習するtransductive方式です。未知の社員や別企業への汎化性能を示すものではありません。
- 個別の類似度の計算では基準群本人も平均に含まれます。交差検証ではテスト社員のラベルを平均・基準モデルの学習へ使いません。
- 全力量を検定してからFDR補正します。大きな保有率差があっても、少人数では有意にならない場合があります。特定の精度は保証しません。
- 全員同じ力量など、エッジと非エッジの両方が得られない場合は学習を省略し、レポートに未学習の参考値と明記します。
- 力量行列と隣接行列は疎行列です。近傍探索は全社員の距離を計算するため、非常に大きいテナントは計算時間が増えます。

既存の `causal_inference/` は独立した研究用関数として保持しています。CLIは呼び出しません。任意ライブラリeconmlが必要なCausal Forestは標準環境の対象外です。

## 調整・開発

```bash
uv run --locked talent-graph analyze --help
uv run --locked pytest -q
uv run --locked ruff check talent_graph tests/test_data.py tests/test_graph_analysis.py tests/test_cli.py
```

学習回数は `--epochs 100`、近傍数は `--neighbors 10`、次元数は `--hidden-dim 32`、乱数種は `--seed 42` です。同じ入力・基準日・パラメータ・依存版で再現できます。並列数・OSによる浮動小数の微差はあり得ます。

```python
from pathlib import Path
from talent_graph.data import load_directory
from talent_graph.analysis import analyze

data = load_directory(Path('/path/to/skillnote-csv'))
result = analyze(data, ['0001', '0002', '0003'], seed=42)
print(result.skills.head())
```

## v2からの移行

- `streamlit run app.py` → `uv run talent-graph ...`。`uv run python app.py ...` も同じコマンドへ転送します。
- `requirements.txt` と分割requirementsは廃止しました。`uv sync --locked` を使用します。
- 旧 `TalentAnalyzer` / `SimpleGNN` のAPI、`config.yaml`、モデルpickle保存、疑似HTE・育成効果の自動推定は廃止しました。上記Python APIへ移行してください。
- `gnn_talent_analyzer.py` は新APIの移行用再エクスポートのみです。
- `docs/history/` の旧資料は過去の記録であり、現行の起動方法・精度・機能を示しません。

実際のSkillnote CSVはこの改修時点で未提供です。自動テストと架空のSkillnote形式CSVで検証しており、特定テナントの独自列や出力仕様は実データ提供後に確認が必要です。
