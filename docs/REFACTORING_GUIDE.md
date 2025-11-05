# ソフトウェアエンジニアリング リファクタリングガイド

## 概要

このドキュメントは、talent-graph-analyzer の運用保守性を向上させるためのリファクタリング計画を記載しています。

## 分析概要

**総問題数**: 50 issues
- Critical Issues: 15件
- Important Issues: 23件
- Enhancement Opportunities: 12件

**現在の状態**:
- Production Readiness: 3/10 (Low)
- Maintainability: 4/10 (Medium-Low)
- Testability: 2/10 (Very Low)
- Security: 3/10 (Low)

---

## Priority 1: Critical Issues (1-2 weeks)

### ✅ 実装済み

#### 1.1 例外処理の改善
**Status**: Partially Implemented
**Files**: gnn_talent_analyzer.py

**変更内容**:
- [x] カスタム例外クラスの定義 (lines 65-103)
- [x] 定数定義の追加 (lines 106-137)
- [x] Fisher検定の例外処理改善 (lines 704-711)
- [ ] 他のベアexcept句の修正（残り 7箇所）

**実装方法**:
```python
# Before (Bad)
try:
    result = operation()
except:
    return None

# After (Good)
try:
    result = operation()
except (ValueError, TypeError) as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    raise DataValidationError(f"Cannot process data: {e}") from e
```

**該当行**:
- Line 484-486 (process_skills での例外処理)
- Line 728-754 (多重検定補正での例外処理)
- Line 1425-1447 (causal effect 計算での例外処理)
- Line 949-953 (get_embeddings での例外処理)
- app.py 線 96-97 (Streamlit エラー処理)

---

### 未実装の Priority 1 Issues

#### 1.2 入力値の検証
**Effort**: 1 day
**Impact**: High
**Files**: gnn_talent_analyzer.py (全メソッド)

**実装方法**:
```python
def load_data(self, member_df, acquired_df, ...):
    """データ読み込み"""
    self._validate_input_data(member_df, 'member_df')
    self._validate_input_data(acquired_df, 'acquired_df')
    # ...

def _validate_input_data(self, df, df_name):
    """入力 DataFrame を検証"""
    if df is None:
        raise DataValidationError(f"{df_name} cannot be None")
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"{df_name} must be pd.DataFrame, got {type(df)}")
    if df.empty:
        raise DataValidationError(f"{df_name} cannot be empty")
```

**検証項目**:
- DataFrame 型の確認
- 必須カラムの存在確認
- 欠損値パターンの確認
- 異常値の検出

---

#### 1.3 セキュリティ修正
**Effort**: 2 hours
**Impact**: Critical
**Files**: gnn_talent_analyzer.py (lines 1127, 1408)

**Issue #14**: Path Traversal 脆弱性
```python
# Before (Bad)
model_path = model_dir / f"model_{version}.pkl"

# After (Good)
import re
safe_version = re.sub(r'[^a-zA-Z0-9_-]', '', version)
if safe_version != version:
    raise ValueError(f"Invalid version format: {version}")
model_path = model_dir / f"model_{safe_version}.pkl"
```

**Issue #10**: Pickle セキュリティリスク
```python
# Before (Bad)
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)  # リスク: 任意のコード実行

# After (Good)
import json
with open(model_path, 'r') as f:
    metadata = json.load(f)  # テキストベース、安全
# 行列は npz で保存
model_arrays = np.load(f"{model_path}.npz")
```

---

#### 1.4 エラーコンテキストの追加
**Effort**: 4 hours
**Impact**: High
**Files**: app.py (lines 96-97, 248-251), config_loader.py (lines 32-33)

**実装方法**:
```python
# app.py での改善
except FileNotFoundError as e:
    logger.error(f"File not found: {uploaded_file.name}", exc_info=True)
    st.error(f"ファイルが見つかりません: {uploaded_file.name}")
except pd.errors.ParserError as e:
    logger.error(f"CSV parsing failed: {uploaded_file.name}", exc_info=True)
    st.error(f"CSV形式が無効です。カラム名と型を確認してください")
except Exception as e:
    logger.error(f"Unexpected error in data upload: {e}", exc_info=True)
    st.error(f"予期しないエラーが発生しました: {str(e)}")
```

---

## Priority 2: Important Issues (2-4 weeks)

### 2.1 ロギング戦略の改善
**Effort**: 1 day
**Files**: 全ファイル
**Changes Required**:
- [x] logger.info() 呼び出しの追加 (線 252)
- [ ] 全 print() を logger 呼び出しに置き換え（残り 6箇所）
- [ ] ログレベルの適切な使い分け
- [ ] 構造化ログの導入

**ログレベルガイドライン**:
```
DEBUG: 詳細情報（開発時のみ）
  - グラフ構築詳細
  - 重み更新情報
  - 内部計算結果

INFO: 一般情報
  - データ読み込み開始/完了
  - 学習開始/完了
  - 分析完了

WARNING: 警告情報
  - サンプルサイズが小さい
  - 異常値検出
  - マッチング品質が低い

ERROR: エラー情報
  - 例外発生
  - データ検証失敗
  - 計算エラー
```

---

### 2.2 コード重複の除去
**Effort**: 4 hours
**Files**: gnn_talent_analyzer.py
**Example**:

Score calculation が 4箇所で重複:
```python
# Lines 664, 975-977, 1070-1072, 1082-1084
distance = np.linalg.norm(self.embeddings[idx] - self.prototype)
max_distance = np.max([np.linalg.norm(emb - self.prototype)
                       for emb in self.embeddings])
score = (1 - distance / max_distance) * 100

# 統合:
def _calculate_excellence_score(self, member_idx: int) -> float:
    """メンバーの優秀度スコアを計算"""
    if not hasattr(self, 'embeddings'):
        raise ModelEvaluationError("Model not trained yet")

    distance = np.linalg.norm(self.embeddings[member_idx] - self.prototype)
    max_distance = np.max([np.linalg.norm(emb - self.prototype)
                          for emb in self.embeddings])

    if max_distance < NUMERICAL_EPSILON:
        logger.warning("Max distance is near zero, returning 0 score")
        return 0.0

    return (1 - distance / max_distance) * 100
```

---

### 2.3 型ヒントの追加
**Effort**: 2 days
**Files**: 全 Python ファイル
**Status**: config_loader.py は既に完了

**実装例**:
```python
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

def load_data(
    self,
    member_df: pd.DataFrame,
    acquired_df: pd.DataFrame,
    skill_df: pd.DataFrame,
    education_df: pd.DataFrame,
    license_df: pd.DataFrame
) -> None:
    """Load and process CSV data"""

def _calculate_excellence_score(self, member_idx: int) -> float:
    """Calculate excellence score for a member"""

def _get_confounders(self) -> np.ndarray:
    """Get confounder variables"""
```

---

### 2.4 God Class の分割設計
**Effort**: 3 days
**Files**: gnn_talent_analyzer.py → 複数ファイルに分割
**Current**: 1 class (TalentAnalyzer), 1371 lines

**提案される設計**:

```
talent_analyzer/
  core/
    __init__.py
    analyzer.py         # Main orchestrator (200 lines)
    model.py           # Training & prediction (250 lines)
    gnn_models.py      # GNN implementation (unchanged)

  data/
    __init__.py
    loader.py          # CSV loading (100 lines)
    validator.py       # Data validation (150 lines)
    features.py        # Feature engineering (100 lines)

  analysis/
    __init__.py
    statistical.py     # Statistical testing (200 lines)
    causal.py          # Causal inference (150 lines)
    interactions.py    # Skill interactions (100 lines)

  persistence/
    __init__.py
    serializer.py      # Model save/load (100 lines)
```

**Benefits**:
- 各クラスが Single Responsibility を持つ
- テストが簡単（mock しやすい）
- 再利用が容易
- 依存関係が明確

**Before** (God Class):
```python
class TalentAnalyzer:
    def load_data(...): ...
    def process_skills(...): ...
    def create_member_features(...): ...
    def build_graph(...): ...
    def fit_unsupervised(...): ...
    def train(...): ...
    def analyze(...): ...
    def estimate_causal_effects(...): ...
    # ... 30+ more methods
```

**After** (Separation of Concerns):
```python
class DataLoader:
    def load_data(...): ...
    def validate_data(...): ...

class FeatureEngineer:
    def create_member_features(...): ...
    def process_skills(...): ...

class TalentModel:
    def __init__(self, gnn, feature_engineer, data_loader):
        self.gnn = gnn
        self.feature_engineer = feature_engineer
        self.data_loader = data_loader

    def train(self, data): ...
    def predict(self, member_idx): ...

class StatisticalAnalyzer:
    def estimate_causal_effects(...): ...
    def analyze_skill_interactions(...): ...
```

---

## Priority 3: Enhancements (2-3 months)

### 3.1 テストスイートの追加
**Effort**: 2 weeks
**Status**: ゼロから開始
**Target Coverage**: 80%+

```
tests/
  __init__.py
  conftest.py              # Fixtures & config

  unit/
    test_loader.py         # DataLoader tests
    test_validator.py      # Validation tests
    test_features.py       # Feature engineering
    test_gnn.py            # GNN model

  integration/
    test_pipeline.py       # Full pipeline
    test_serialization.py  # Save/load

  performance/
    test_training_speed.py # Benchmark
    test_memory.py         # Memory profiling
```

### 3.2 CI/CD パイプライン
**Effort**: 2 days
**Status**: 未実装

**.github/workflows/tests.yml**:
```yaml
- Linting (flake8, black, mypy)
- Testing (pytest with coverage)
- Security scanning (bandit, safety)
- Build check
```

### 3.3 設定検証スキーマ
**Effort**: 1 day
**Tool**: pydantic

```python
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    n_layers: int = Field(3, ge=1, le=10)
    hidden_dim: int = Field(128, ge=32, le=512)
    dropout: float = Field(0.3, ge=0.0, le=0.5)

    class Config:
        env_prefix = "MODEL_"

# Validate
try:
    config = ModelConfig(**config_dict)
except ValidationError as e:
    raise ConfigurationError(f"Invalid config: {e}")
```

---

## ハウツーガイド

### パターン 1: 例外処理
```python
# Good
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    raise DomainError(f"Cannot process: {e}") from e
except AnotherError as e:
    logger.warning(f"Fallback: {e}")
    return default_value
```

### パターン 2: 入力検証
```python
def process_data(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(data)}")
    if data.empty:
        raise ValueError("DataFrame cannot be empty")
    if not all(col in data.columns for col in REQUIRED_COLS):
        raise KeyError(f"Missing required columns: {REQUIRED_COLS}")
    logger.info(f"Data validated: {len(data)} rows")
```

### パターン 3: ロギング
```python
logger.debug(f"Processing {n_items} items")
logger.info(f"Model training started with {n_epochs} epochs")
logger.warning(f"Sample size small ({n} < 30), results may be unstable")
logger.error(f"Training failed: {e}", exc_info=True)
```

### パターン 4: 依存性注入
```python
# Before (Bad)
class TalentModel:
    def __init__(self):
        self.gnn = SimpleGNN()  # Tight coupling

# After (Good)
class TalentModel:
    def __init__(self, gnn=None, scaler=None):
        self.gnn = gnn or SimpleGNN()
        self.scaler = scaler or StandardScaler()

# Test:
test_gnn = MockGNN()
model = TalentModel(gnn=test_gnn)
```

---

## チェックリスト

### 実装前
- [ ] 関連する Issue を理解している
- [ ] リファクタリングの目的が明確
- [ ] テストが書ける状態

### 実装中
- [ ] 例外メッセージが明確
- [ ] ログレベルが適切
- [ ] 型ヒントが完全
- [ ] docstring が完全

### 実装後
- [ ] テストが通る
- [ ] Linter が通る
- [ ] Mypy チェック OK
- [ ] カバレッジ 80% 以上

---

## 推奨実装順序

1. **Week 1**: 例外処理、入力検証、セキュリティ修正
2. **Week 2**: ロギング改善、コード重複除去
3. **Week 3-4**: 型ヒント、God class 分割計画
4. **Week 5-6**: テストスイート、CI/CD
5. **Ongoing**: リファクタリング

---

## 参考資料

- Clean Code (Robert C. Martin)
- The Pragmatic Programmer
- Python 最高の実践
- PEP 8 (Style Guide for Python Code)
- Google Python Style Guide

---

## 連絡先

リファクタリングに関する質問: Issue を作成してください。
