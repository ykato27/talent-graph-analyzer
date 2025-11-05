# GNN優秀人材分析システム

Graph Neural Network (GNN) を用いた優秀人材の特徴抽出・分析システム

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

## 📖 ドキュメント

詳細なドキュメントは `docs/` ディレクトリを参照してください：

- **[docs/README.md](./docs/README.md)** - 使用方法とセットアップガイド
- **[docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)** - システムアーキテクチャ
- **[docs/DEVELOPMENT.md](./docs/DEVELOPMENT.md)** - 開発ガイド
- **[docs/CHANGELOG.md](./docs/CHANGELOG.md)** - 変更履歴
- **[docs/REFACTORING.md](./docs/REFACTORING.md)** - リファクタリング記録

## 🚀 クイックスタート

### 1. 環境構築

```bash
# リポジトリのクローン
git clone https://github.com/ykato27/talent-graph-analyzer.git
cd talent-graph-analyzer

# 仮想環境の作成（推奨）
python -m venv venv

# 仮想環境の有効化
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. アプリケーション起動

```bash
# Streamlit アプリの起動
streamlit run run.py
```

または

```bash
# Streamlit 直接実行
streamlit run src/talent_analyzer/ui/app.py
```

ブラウザで `http://localhost:8501` にアクセスします。

## 📁 プロジェクト構成

```
talent-graph-analyzer/
├── src/                              # ソースコード
│   └── talent_analyzer/
│       ├── __init__.py               # パッケージ初期化
│       ├── core/                     # コアロジック
│       │   ├── __init__.py
│       │   ├── analyzer.py           # 分析エンジン
│       │   └── gnn_models.py         # GNNモデル実装
│       ├── config/                   # 設定管理
│       │   ├── __init__.py
│       │   ├── constants.py          # 定数定義
│       │   └── loader.py             # 設定ローダー
│       ├── utils/                    # ユーティリティ関数
│       │   ├── __init__.py
│       │   └── helpers.py            # ヘルパー関数
│       └── ui/                       # ユーザーインターフェース
│           ├── __init__.py
│           └── app.py                # Streamlit アプリ
├── config/                           # 設定ファイル
│   ├── config.yaml                   # メイン設定ファイル
│   └── .env.example                  # 環境変数テンプレート
├── docs/                             # ドキュメント
│   ├── README.md                     # 詳細な使用ガイド
│   ├── ARCHITECTURE.md               # システム設計書
│   ├── DEVELOPMENT.md                # 開発ガイド
│   ├── CHANGELOG.md                  # 変更履歴
│   └── REFACTORING.md                # リファクタリング記録
├── tests/                            # テストコード
│   └── __init__.py
├── models/                           # 学習済みモデル（自動生成）
├── logs/                             # ログファイル（自動生成）
├── run.py                            # Streamlit エントリーポイント
├── requirements.txt                  # Python依存パッケージ
├── .gitignore                        # Git除外ファイル
└── README.md                         # このファイル
```

## 🎯 主な機能

- **Graph Neural Network**: 社員とスキルの関係性をグラフ構造で学習
- **半教師あり学習**: ラベルなしデータも活用して高精度化
- **Few-shot学習**: 優秀群5名程度の少数サンプルでも動作
- **統計的検定**: Fisher正確検定と多重検定補正による信頼性の高い分析
- **因果推論**: 傾向スコアマッチングによるスキルの真の効果推定
- **スキル相互作用**: 相乗効果のあるスキル組み合わせの自動発見
- **モデル評価**: HoldoutまたはLOOCVによる定量的な性能評価
- **リアルタイム進捗表示**: 学習中の進捗率・ロス・経過時間をリアルタイム表示

詳細は [docs/README.md](./docs/README.md) を参照してください。

## 🔧 必要環境

- Python 3.8以上
- CPU環境で動作（GPUは不要）

## 📊 技術スタック

| カテゴリ | 技術 |
|--------|------|
| UI Framework | Streamlit |
| データ処理 | Pandas, NumPy |
| 機械学習 | scikit-learn |
| グラフ処理 | 自作実装 |
| グラフ表示 | Plotly |
| 統計分析 | SciPy, StatsModels |
| 言語 | Python 3.8+ |

## 📝 使用例

### 基本的な使用フロー

1. **データアップロード**: 5つのCSVファイルをアップロード
2. **優秀人材選択**: 優秀と考える社員を選択
3. **分析実行**: 学習を実行して分析
4. **結果確認**: ダッシュボードで結果を確認
5. **結果ダウンロード**: 重要スキル一覧などをCSVでダウンロード

詳細な手順は [docs/README.md](./docs/README.md) を参照してください。

## 🤝 貢献

このプロジェクトへの貢献を歓迎します。

1. フォークする
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下でライセンスされています。詳細は[LICENSE](./LICENSE)ファイルを参照してください。

## 📧 サポート

質問や問題がある場合は、GitHubのIssueを作成してください。

## 🙏 謝辞

このプロジェクトは、Graph Neural Networkを用いた人材分析の最新技術を活用しています。

---

**最終更新**: 2025-11-05
**バージョン**: 1.2.0
