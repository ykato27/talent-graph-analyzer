# 開発ガイド (DEVELOPMENT GUIDE)

このドキュメントは、プロジェクトへの貢献者向けの開発ガイドです。

## 開発環境のセットアップ

### 1. 前提条件

- Python 3.8以上
- Git
- pip または conda

### 2. 開発環境の構築

```bash
# リポジトリのクローン
git clone https://github.com/ykato27/talent-graph-analyzer.git
cd talent-graph-analyzer

# 仮想環境の作成（推奨）
python -m venv venv

# 仮想環境の有効化
# Windows の場合
venv\Scripts\activate
# Mac/Linux の場合
source venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt

# 開発用パッケージのインストール（オプション）
pip install pytest pytest-cov black flake8
```

### 3. 設定ファイルの準備

```bash
# .env ファイルの作成（必要に応じて）
cp .env.example .env
```

## プロジェクト構成

### コアモジュール

#### `app.py`
Streamlit による Web UI メインアプリケーション

**主要コンポーネント:**
- `initialize_session_state()`: セッション状態の初期化
- `render_header()`: ヘッダー描画
- `render_data_upload_sidebar()`: データアップロード機能
- `render_skill_cards()`: Top スキルカード表示
- `render_analysis_metrics()`: 分析サマリー表示
- `render_dashboard_charts()`: グラフ統合表示
- `render_model_metrics()`: モデル性能表示
- `render_detailed_analysis()`: 詳細分析セクション

**主要処理フロー:**
1. データアップロード → analyzer 初期化
2. 優秀人材選択 → selected_members 設定
3. 分析実行 → train → analyze → evaluate → causal → interactions
4. 結果表示 → ダッシュボード描画

#### `gnn_talent_analyzer.py`
GNN モデルと分析エンジン

**主要クラス:**

##### `SimpleGNN`
Graph Neural Network モデルの実装

```python
class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        # GraphSAGE層の構成

    def forward(self, node_features, adj_matrix):
        # ノード埋め込みの計算
        return embeddings
```

**主要メソッド:**
- `forward()`: フォワードパス
- `fit_unsupervised()`: 半教師あり学習
- `get_embeddings()`: 埋め込みベクトルの取得

##### `TalentAnalyzer`
分析エンジンの実装

**主要メソッド:**
- `load_data()`: CSVデータの読み込みと前処理
- `train()`: GNN学習とスキル重要度計算
- `analyze()`: 詳細分析の実行
- `evaluate_model()`: モデル性能評価
- `estimate_causal_effects()`: 因果推論
- `analyze_skill_interactions()`: スキル相互作用分析
- `save_model()`: モデルの保存
- `load_model()`: モデルの読み込み

#### `config_loader.py`
設定ファイル読み込みユーティリティ

**主要関数:**
- `get_config(key, default)`: config.yaml から設定値を取得

#### `config.yaml`
設定ファイル

**主要セクション:**
```yaml
model:
  n_layers: 3
  hidden_dim: 128
  output_dim: 128
  dropout_rate: 0.3

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.01
  early_stopping_patience: 10

analysis:
  min_excellent_members: 3
  max_excellent_members_recommended: 20

ui:
  page_title: "GNN優秀人材分析システム"
  colors:
    excellent_group: "#FF6B6B"
    non_excellent_group: "#4ECDC4"

files:
  encoding: "utf-8-sig"
```

## 開発のワークフロー

### 1. ブランチ戦略

```
main (本番)
 └── develop (開発)
      ├── feature/xxx (機能開発)
      ├── fix/xxx (バグ修正)
      └── refactor/xxx (リファクタリング)
```

**ブランチ命名規則:**
- 機能: `feature/機能名`
- バグ修正: `fix/バグ名`
- リファクタリング: `refactor/対象`
- ドキュメント: `docs/ドキュメント名`

### 2. コミットメッセージフォーマット

Conventional Commits に従う：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type:**
- `feat`: 新機能
- `fix`: バグ修正
- `refactor`: リファクタリング
- `docs`: ドキュメント
- `test`: テスト追加
- `perf`: パフォーマンス改善

**例:**
```
feat(ui): Add dashboard layout for analysis results

- Implement render_skill_cards() for top 3 skills display
- Integrate render_dashboard_charts() for unified graph display
- Reduce app.py from 1197 to 624 lines (47% reduction)
```

### 3. プルリクエストプロセス

1. 機能ブランチを作成
2. コミットをプッシュ
3. Pull Request を作成
4. コードレビュー対応
5. マージ

## コード品質基準

### スタイルガイド

Python の PEP 8 に従う：

```bash
# コード整形（Black）
black app.py gnn_talent_analyzer.py

# リント検査（Flake8）
flake8 app.py gnn_talent_analyzer.py
```

### 命名規則

- **関数**: `snake_case` - `render_skill_cards()`
- **クラス**: `PascalCase` - `SimpleGNN`, `TalentAnalyzer`
- **定数**: `UPPER_SNAKE_CASE` - `MIN_EPOCHS`, `DEFAULT_EPOCHS`
- **プライベート**: アンダースコア接頭辞 - `_analyze_skill_importance()`

### ドキュメンテーション

すべての関数にはドキュメント文字列を記述：

```python
def render_skill_cards(skill_importance, top_n=3):
    """Top Nのスキルをカード型で表示

    Args:
        skill_importance (list): スキル重要度のリスト
        top_n (int): 表示するスキル数（デフォルト: 3）

    Returns:
        None (Streamlit に描画)

    Example:
        >>> render_skill_cards(results['skill_importance'], top_n=3)
    """
```

## テスト

### ユニットテスト

```bash
# テスト実行
pytest tests/

# カバレッジ測定
pytest --cov=. tests/
```

### 手動テスト

```bash
# Streamlit アプリの起動と動作確認
streamlit run app.py
```

## 常用コマンド

```bash
# ローカル開発サーバー起動
streamlit run app.py

# コード品質チェック
black --check app.py gnn_talent_analyzer.py
flake8 app.py gnn_talent_analyzer.py

# 依存関係の更新確認
pip list --outdated

# パッケージの再インストール
pip install -r requirements.txt --upgrade
```

## トラブルシューティング

### よくある問題

#### 1. モジュールインポートエラー

```
ModuleNotFoundError: No module named 'streamlit'
```

**解決方法:**
```bash
pip install streamlit>=1.28.0
```

#### 2. メモリ不足エラー

```
MemoryError: Unable to allocate memory
```

**解決方法:**
- `config.yaml` でエポック数を減らす
- 対象社員数を削減する
- ブラウザのキャッシュをクリア

#### 3. グラフ描画エラー

```
PlotlyError: Invalid value of type <class 'NoneType'>
```

**解決方法:**
- セッション状態をリセット（ブラウザ更新）
- `None` チェックを追加

## アーキテクチャの理解

### データフロー

```
CSV Upload
    ↓
Data Loading & Preprocessing
    ↓
Graph Construction
    ↓
GNN Training (with callback)
    ↓
Skill Importance Analysis
    ↓
Member Score Calculation
    ↓
Model Evaluation
    ↓
Causal Effect Estimation
    ↓
Skill Interaction Analysis
    ↓
Dashboard Rendering
```

### セッション状態管理

```python
st.session_state:
  - analyzer: TalentAnalyzer instance
  - data_loaded: bool
  - results: dict (skill_importance, member_scores, etc.)
  - evaluation_results: dict
  - causal_results: list
  - interaction_results: list
  - member_df: DataFrame
```

## パフォーマンス最適化

### メモリ使用量削減

- 大規模データセットは `pd.read_csv()` で chunk 処理
- 不要な中間変数は早期に削除
- NumPy 配列は計算後に `gc.collect()` を呼び出し

### 計算速度改善

- GraphSAGE はサンプリング時間最適化（デフォルト: nearest neighbors）
- 因果推論は並列化可能な部分で `multiprocessing` 使用
- Streamlit キャッシュ装飾子を活用

```python
@st.cache_data
def expensive_computation():
    return result
```

## リリースプロセス

### バージョンニング

[Semantic Versioning](https://semver.org/) に従う: `MAJOR.MINOR.PATCH`

### リリース手順

1. `CHANGELOG.md` を更新
2. バージョン番号を更新
3. リリースノートを作成
4. Git タグを作成
5. GitHub Releases に公開

## 関連リソース

- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [PEP 8 Style Guide](https://pep8.org/)

