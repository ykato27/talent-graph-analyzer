# リファクタリングガイド v2.0

## 概要

このドキュメントでは、talent-graph-analyzerのリファクタリングで実施した改善内容と、
今後のベストプラクティスについて説明します。

---

## 実施したリファクタリング

### Phase 1: Critical Issues（完了）

#### 1. ✅ テストスイートの構築

**Before**: テストファイル1つのみ
**After**: 包括的なテストスイート

```
tests/
├── __init__.py
├── conftest.py                 # 共通フィクスチャ
├── test_validators.py          # 入力検証のテスト
├── test_sensitivity_analysis.py # 感度分析のテスト
├── test_cluster_robust.py      # クラスター頑健SEのテスト（追加予定）
├── test_psm_diagnostics.py     # PSM診断のテスト（追加予定）
├── test_did_analysis.py        # DID分析のテスト（追加予定）
└── benchmarks/                 # パフォーマンステスト
```

**カバレッジ目標**: 80%以上

**実行方法**:
```bash
# 全テスト実行
pytest

# カバレッジ付き
pytest --cov

# 特定のモジュールのみ
pytest tests/test_validators.py -v
```

---

#### 2. ✅ 型ヒントの完全化

**Before**: 不完全な型ヒント
```python
def cluster_robust_se(...) -> Dict:  # ❌ 具体的な型が不明
    ...
```

**After**: 完全な型ヒント
```python
from causal_inference.types import ClusterRobustResult

def cluster_robust_se(...) -> ClusterRobustResult:  # ✅ 明確
    ...
```

**新規追加**:
- `causal_inference/types.py`: TypedDict、Protocol、Literal型の定義
- すべての関数に完全な型ヒント
- mypy strict modeで検証

**型チェック実行**:
```bash
mypy causal_inference model_evaluation visualization --strict
```

---

#### 3. ✅ エラーハンドリングの強化

**Before**: 一般的なValueErrorのみ
```python
if len(y) != len(X):
    raise ValueError("Length mismatch")  # ❌ 不親切
```

**After**: カスタム例外と詳細なメッセージ
```python
from causal_inference.exceptions import InvalidInputError

if len(y) != len(X):
    raise InvalidInputError(
        f"Length mismatch: y ({len(y)}) != X ({len(X)}). "
        f"Please ensure all inputs have the same number of samples."
    )  # ✅ 親切
```

**新規追加**:
- `causal_inference/exceptions.py`: カスタム例外階層
- `causal_inference/validators.py`: 入力検証関数

**例外階層**:
```
CausalInferenceError
├── InvalidInputError
├── InsufficientDataError
├── ConvergenceError
├── ConfigurationError
├── MatchingError
└── EstimationError
```

---

#### 4. ✅ 依存関係管理の改善

**Before**: バージョン固定なし
```
econml>=0.14.0  # ❌ 上限なし
matplotlib>=3.7.0  # ❌ オプショナルなのに必須
```

**After**: バージョン範囲固定 + モジュール分割
```
requirements/
├── base.txt        # 必須
├── causal.txt      # 因果推論用
├── viz.txt         # 可視化用（オプショナル）
├── ml.txt          # ML用（オプショナル）
├── dev.txt         # 開発用
└── streamlit.txt   # Streamlit UI用
```

**インストール方法**:
```bash
# 最小構成
pip install -r requirements/base.txt

# 因果推論機能を使う場合
pip install -r requirements/causal.txt

# 開発環境
pip install -r requirements/dev.txt
```

---


#### 5. ✅ ロギング戦略の統一

**Before**: 不統一なロギング
```python
logger.info(f"Calculating...")  # ❌ 開始/終了が不明
```

**After**: 構造化ロギング
```python
from causal_inference.utils import log_execution_time

with log_execution_time(logger, "cluster-robust SE") as metadata:
    # 処理
    metadata['n_clusters'] = n_clusters
    metadata['n_samples'] = n_samples

# ログ出力:
# INFO: Starting: cluster-robust SE
# INFO: Completed: cluster-robust SE in 0.12s (n_clusters=10, n_samples=100)
```

**新規追加**:
- `causal_inference/utils.py`: ロギングユーティリティ
- `log_execution_time`: コンテキストマネージャー
- `timing_decorator`: デコレーター

---

### Phase 2: High Priority（実施中）

#### 6. 🔄 ドキュメント生成（Sphinx）

**計画**:
```
docs/
├── source/
│   ├── conf.py
│   ├── index.rst
│   ├── api/
│   │   ├── causal_inference.rst
│   │   ├── model_evaluation.rst
│   │   └── visualization.rst
│   ├── tutorials/
│   │   ├── quickstart.rst
│   │   ├── sensitivity_analysis.rst
│   │   └── did_analysis.rst
│   └── examples/
└── build/
```

**生成**:
```bash
cd docs
make html
```

---

#### 7. 🔄 パフォーマンス最適化

**計画**:
- プロファイリングの追加（cProfile, line_profiler）
- メモリ最適化（numpy配列の再利用）
- 並列処理（joblib, multiprocessing）
- キャッシュ戦略（functools.lru_cache）

---

## ベストプラクティス

### コーディング規約

#### 1. 型ヒントを必ず使用

```python
# ✅ GOOD
def calculate_smd(
    treated: np.ndarray,
    control: np.ndarray,
    continuous: bool = True
) -> float:
    ...

# ❌ BAD
def calculate_smd(treated, control, continuous=True):
    ...
```

#### 2. 入力検証を徹底

```python
# ✅ GOOD
from causal_inference.validators import validate_array_lengths

def my_function(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    validate_array_lengths(y, X, names=["y", "X"])
    # 処理
    ...

# ❌ BAD
def my_function(y, X):
    # 検証なし
    ...
```

#### 3. エラーメッセージを親切に

```python
# ✅ GOOD
raise InvalidInputError(
    f"gamma_values must be >= 1.0, got {gamma}. "
    f"Gamma represents the strength of hidden confounding."
)

# ❌ BAD
raise ValueError("Invalid gamma")
```

#### 4. ロギングを適切に

```python
# ✅ GOOD
with log_execution_time(logger, "DID estimation") as meta:
    result = did_estimation(...)
    meta['n_treated'] = result['n_treated']

# ❌ BAD
logger.info("Starting DID")
result = did_estimation(...)
logger.info("Done")
```

#### 5. テストを先に書く（TDD）

```python
# test_new_feature.py
def test_new_feature():
    result = new_feature(input_data)
    assert result['status'] == 'success'
    assert result['value'] > 0

# ↑ これを先に書いてから実装
```

---

### テスト戦略

#### 1. テストの種類

```python
import pytest

@pytest.mark.unit
def test_unit():
    """単体テスト: 1つの関数をテスト"""
    ...

@pytest.mark.integration
def test_integration():
    """統合テスト: 複数のモジュールを組み合わせ"""
    ...

@pytest.mark.slow
def test_slow():
    """遅いテスト: 実行時間がかかる"""
    ...
```

**実行**:
```bash
# 単体テストのみ
pytest -m unit

# 遅いテストを除外
pytest -m "not slow"
```

#### 2. フィクスチャの活用

```python
# conftest.py
@pytest.fixture
def sample_data():
    return np.random.randn(100, 5)

# test_module.py
def test_function(sample_data):
    result = my_function(sample_data)
    assert result.shape == (100,)
```

#### 3. パラメトライズテスト

```python
@pytest.mark.parametrize("input,expected", [
    (1.0, 2.0),
    (2.0, 4.0),
    (3.0, 6.0),
])
def test_double(input, expected):
    assert double(input) == expected
```

---

### ドキュメント規約

#### 1. docstringはNumPy形式

```python
def my_function(x: np.ndarray, y: int = 5) -> float:
    """
    関数の短い説明（1行）

    より詳細な説明（複数行可）

    Parameters
    ----------
    x : np.ndarray
        入力配列の説明
    y : int, default=5
        パラメータの説明

    Returns
    -------
    float
        返り値の説明

    Raises
    ------
    InvalidInputError
        エラーの説明

    Examples
    --------
    >>> my_function(np.array([1, 2, 3]))
    42.0

    Notes
    -----
    追加の注意事項

    References
    ----------
    .. [1] Smith et al. (2023). "Title". Journal.
    """
    ...
```

#### 2. 型ヒントとdocstringを両方

```python
# ✅ GOOD: 型ヒント + docstring
def func(x: int) -> str:
    """整数を文字列に変換

    Parameters
    ----------
    x : int
        変換する整数
    """
    return str(x)
```

---

### セキュリティ

#### 1. Pickleの使用を避ける

```python
# ❌ BAD
import pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)  # セキュリティリスク

# ✅ GOOD
import json
with open('model.json', 'r') as f:
    model = json.load(f)
```

#### 2. パストラバーサルを防ぐ

```python
from pathlib import Path

def safe_read_file(filename: str) -> str:
    # ✅ GOOD: Path.resolve()で正規化
    base_dir = Path('/home/user/data')
    file_path = (base_dir / filename).resolve()

    if not file_path.is_relative_to(base_dir):
        raise SecurityError("Path traversal detected")

    return file_path.read_text()
```

---

## まとめ

### 実装済み ✅
1. テストスイート
2. 型ヒント完全化
3. エラーハンドリング強化
4. 依存関係管理
5. ロギング戦略
6. 入力検証の統一

### 実装中 🔄
7. ドキュメント生成
8. パフォーマンス最適化

### 今後の課題 📋
9. セキュリティ監査
10. アーキテクチャドキュメント
11. ベンチマークスイート

---

## 参考資料

- [PEP 8](https://pep8.org/): Pythonコーディング規約
- [PEP 257](https://peps.python.org/pep-0257/): Docstring規約
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)
- [mypy Documentation](https://mypy.readthedocs.io/)
