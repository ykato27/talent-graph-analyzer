# システムアーキテクチャ (ARCHITECTURE)

GNN優秀人材分析システムの技術アーキテクチャを説明するドキュメントです。

## システム概要

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit UI Layer                     │
│  (app.py - ダッシュボード、ユーザー対話)               │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼─────────────┐    ┌─────────▼────────────────┐
│  Data Upload        │    │  Analysis Engine         │
│  & Preprocessing    │    │  (gnn_talent_analyzer)   │
│                     │    │                          │
│  - CSV Loading      │    │  - GNN Model (PyTorch)   │
│  - Validation       │    │  - Statistical Analysis  │
│  - Graph Building   │    │  - Causal Inference      │
└─────────┬───────────┘    │  - Skill Interaction     │
          │                └─────────┬────────────────┘
          │                          │
          └──────────────┬───────────┘
                         │
        ┌────────────────▼─────────────┐
        │   Results & Evaluation        │
        │                              │
        │  - Model Metrics             │
        │  - Skill Rankings            │
        │  - Member Scores             │
        │  - Causal Effects            │
        │  - Interactions              │
        └──────────────┬────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  Visualization & Export      │
        │                             │
        │  - Dashboard Charts         │
        │  - CSV Export               │
        │  - Model Versioning         │
        └─────────────────────────────┘
```

## コンポーネント詳細

### 1. UI Layer (app.py)

#### 責務
- ユーザーインターフェース提供
- データのインタラクティブ入力
- リアルタイム学習進捗表示
- 結果の可視化

#### 主要コンポーネント関数

```python
# ヘッダー & 初期化
- initialize_session_state()
- render_header()

# データ入力
- render_data_upload_sidebar()

# ダッシュボード表示
- render_skill_cards()          # Top 3 skills
- render_analysis_metrics()     # Summary metrics
- render_dashboard_charts()     # Integrated graphs
- render_model_metrics()        # Model performance
- render_detailed_analysis()    # Collapsible sections

# リアルタイム進捗
- on_epoch_callback()           # Learning progress
```

#### Streamlit セッション管理

```python
st.session_state:
  analyzer          # TalentAnalyzer instance
  data_loaded       # bool - データ読み込み完了フラグ
  results           # dict - 分析結果
  evaluation_results # dict - モデル評価結果
  causal_results    # list - 因果推論結果
  interaction_results # list - スキル相互作用結果
  member_df         # DataFrame - 社員データ
```

---

### 2. Data Processing Layer

#### CSV データ読み込み

```python
analyzer.load_data(
    member_df,      # 社員マスタ
    acquired_df,    # スキル習得データ
    skill_df,       # スキルマスタ
    education_df,   # 教育マスタ
    license_df      # 資格マスタ
)
```

**処理フロー:**
1. CSV ファイル読み込み
2. カラム名マッピング（config.yaml から）
3. データ型変換
4. 欠損値処理
5. スキルマトリクス構築

#### スキルマトリクス

```
          Skill1 Skill2 Skill3 ... SkillN
Member1    1.0    0.0    0.5   ...  1.0
Member2    0.5    1.0    0.0   ...  0.0
Member3    1.0    0.5    1.0   ...  0.5
...
MemberM    0.0    1.0    1.0   ...  1.0

shape = (M, N)  where M=members, N=skills
dtype = float32
```

**値の意味:**
- 0.0: スキルなし
- 0.5: スキルレベル低
- 1.0: スキルレベル高

---

### 3. Graph Construction

#### グラフ構築処理

```python
# ノード（社員）: 全M名
# エッジ: スキル類似度により自動接続
# 重み: スキルレベルと保有数

def build_graph():
    # コサイン類似度でエッジ構築
    similarity = cosine_similarity(skill_matrix)

    # 閾値以上の類似度でエッジを生成
    edges = similarity > threshold

    # 隣接行列（重み付き）を作成
    adj_matrix = similarity * edges

    return adj_matrix  # shape: (M, M)
```

#### グラフ特性

```
ノード特徴:
  - ノード次数: 社員のスキル保有数
  - ノード埋め込み初期値: スキルプロファイルの平均

エッジ重み:
  - コサイン類似度 [0, 1]
  - 対象社員間のスキル類似性を表現

グラフ統計:
  - ノード数: 全社員数 (M)
  - エッジ密度: 接続比率
  - クラスタリング係数: 三角形の割合
```

---

### 4. GNN Model Layer (gnn_talent_analyzer.py)

#### SimpleGNN モデルアーキテクチャ

```
Input: Node Features (M × input_dim)
       + Adjacency Matrix (M × M)
       │
       ├─ Dense Layer: input_dim → hidden_dim
       │
       ├─ GraphSAGE Layer 1
       │  ├─ Neighbor Aggregation (Mean)
       │  ├─ Linear Transformation
       │  ├─ ReLU Activation
       │  └─ Dropout
       │
       ├─ GraphSAGE Layer 2
       │  ├─ Neighbor Aggregation (Mean)
       │  ├─ Linear Transformation
       │  ├─ ReLU Activation
       │  └─ Dropout
       │
       ├─ GraphSAGE Layer 3
       │  ├─ Neighbor Aggregation (Mean)
       │  ├─ Linear Transformation
       │  ├─ ReLU Activation
       │  └─ Dropout
       │
       └─ Output: Node Embeddings (M × output_dim=128)
```

#### 損失関数

```python
L_total = L_edge + L_contrastive

# Edge Prediction Loss
L_edge = BCE(predicted_edges, true_edges)

# Contrastive Loss (DGI-style)
L_contrastive = -log(sigma(z_u · z_v+))
                + log(sigma(z_u · z_v-))

where:
  z_u, z_v = node embeddings
  v+ = positive neighbor
  v- = negative sample
```

#### 学習パラメータ

```yaml
model:
  input_dim: 100          # スキル特徴次元
  hidden_dim: 128         # 隠れ層次元
  output_dim: 128         # 出力埋め込み次元
  n_layers: 3             # グラフ畳み込み層数
  dropout_rate: 0.3       # ドロップアウト率

training:
  learning_rate: 0.01
  epochs: 100
  batch_size: 32
  early_stopping_patience: 10
  optimizer: Adam
```

---

### 5. Analysis Engine

#### 5.1 スキル重要度分析

```python
def analyze_skill_importance(skill_matrix, excellent_indices):
    """Fisher正確検定によるスキル重要度計算"""

    for each skill:
        # 2×2分割表の作成
        excellent_has = sum(excellent & skill)
        excellent_not = sum(excellent & ~skill)
        non_excellent_has = sum(~excellent & skill)
        non_excellent_not = sum(~excellent & ~skill)

        # Fisher正確検定
        p_value = fisher_exact_test(contingency_table)

        # FDR補正
        p_adjusted = fdr_correction(p_values)

        # 重要度スコア計算
        importance = rate_diff * (1 + statistical_weight)
```

**出力:**
```python
{
  'skill_name': str,
  'excellent_rate': float,          # 優秀群保有率
  'non_excellent_rate': float,      # 非優秀群保有率
  'rate_diff': float,               # 保有率差分
  'importance_score': float,        # 重要度スコア
  'p_value': float,                 # Fisher正確検定p値
  'p_adjusted': float,              # FDR補正p値
  'significance_level': str,        # '***', '**', '*', 'n.s.'
  'ci_lower': float, 'ci_upper': float  # 95%信頼区間
}
```

#### 5.2 社員スコア計算

```python
def calculate_member_scores(embeddings, excellent_embeddings):
    """Few-shot学習による社員スコア計算"""

    # 優秀群の平均埋め込み（プロトタイプ）を計算
    prototype = mean(excellent_embeddings)

    # 全社員との距離を計算
    for each member:
        similarity = cosine_similarity(embedding, prototype)
        score = sigmoid(similarity)  # [0, 1]にスケール
```

**出力:**
```python
{
  'member_name': str,
  'member_code': str,
  'score': float,        # 優秀度スコア [0, 1]
  'is_excellent': bool   # 優秀群フラグ
}
```

#### 5.3 モデル評価

**Holdout法:**
```python
# 優秀群をランダムに80:20に分割
train_excellent, test_excellent = split(excellent, 0.2)

# モデル訓練
train()

# テストセット予測
pred_scores = predict(test_excellent)

# 評価指標計算
auc = roc_auc_score(true_labels, pred_scores)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

# 過学習検出
auc_diff = train_auc - test_auc
is_overfitting = auc_diff > threshold
```

**LOOCV法:**
```python
# Leave-One-Out交差検証
for each excellent_member:
    train_set = all_excellent - member
    test_set = member

    train_model()
    evaluate_on_test()

# 全 fold の結果を平均
avg_auc = mean(fold_aucs)
```

#### 5.4 因果推論

```python
def estimate_causal_effects(skill_matrix, excellent_flags, confounders):
    """傾向スコアマッチングによる因果推論"""

    for each skill:
        # 1. 処置群（スキルあり）・対照群（スキルなし）を分割
        treated = (skill_matrix[:, skill] > 0)
        control = (skill_matrix[:, skill] == 0)

        # 2. 傾向スコア計算（交絡因子の逆確率加重）
        propensity_score = logistic_regression(
            features=confounders,
            target=treated
        )

        # 3. マッチング（1:1最近傍マッチング）
        matched_pairs = nearest_neighbor_matching(
            treated_ps=propensity_score[treated],
            control_ps=propensity_score[control],
            threshold=0.1  # caliper
        )

        # 4. 平均処置効果（ATE）計算
        ate = mean(excellent[treated_matched] -
                   excellent[control_matched])

        # 5. 統計検定
        t_stat, p_value = ttest_ind(
            excellent[treated_matched],
            excellent[control_matched]
        )

        # 6. 信頼区間
        ci_lower, ci_upper = bootstrap_ci(matched_pairs, 0.95)
```

**出力:**
```python
{
  'skill_name': str,
  'status': 'success' | 'insufficient_samples' | 'no_variation',
  'causal_effect': float,              # ATE
  'p_value': float,                    # t検定p値
  'significant': bool,                 # p < 0.05
  'ci_lower': float, 'ci_upper': float,  # 95%信頼区間
  'n_matched_pairs': int,
  'interpretation': str                # 日本語説明
}
```

#### 5.5 スキル相互作用分析

```python
def analyze_skill_interactions(skill_matrix, excellent_flags):
    """スキル相互作用（相乗効果）分析"""

    for each pair of skills (skill_a, skill_b):
        # 4グループに分類
        neither = ~skill_a & ~skill_b
        a_only = skill_a & ~skill_b
        b_only = ~skill_a & skill_b
        both = skill_a & skill_b

        # 各グループの優秀率
        rate_neither = mean(excellent[neither])
        rate_a = mean(excellent[a_only])
        rate_b = mean(excellent[b_only])
        rate_both = mean(excellent[both])

        # 効果の計算
        effect_a = rate_a - rate_neither
        effect_b = rate_b - rate_neither
        actual_effect = rate_both - rate_neither
        additive_effect = effect_a + effect_b

        # 相乗効果（シナジー）
        synergy = actual_effect - additive_effect

        # サンプル数
        n_neither = sum(neither)
        n_a = sum(a_only)
        n_b = sum(b_only)
        n_both = sum(both)
```

**出力:**
```python
{
  'skill_a_name': str,
  'skill_b_name': str,
  'synergy': float,                   # 相乗効果
  'rate_neither': float, 'n_neither': int,
  'rate_a': float, 'n_a': int,
  'rate_b': float, 'n_b': int,
  'rate_both': float, 'n_both': int,
  'effect_a': float,                  # A単独の効果
  'effect_b': float,                  # B単独の効果
  'additive_effect': float,           # 相加効果
  'actual_effect': float              # 実際の効果
}
```

---

### 6. 結果表示層 (Visualization)

#### ダッシュボード構成

```
┌─────────────────────────────────────────────────┐
│  🎯 Top スキル                                  │
│  ┌──────────┬──────────┬──────────┐            │
│  │ Skill1   │ Skill2   │ Skill3   │            │
│  │ Score99  │ Score85  │ Score72  │            │
│  └──────────┴──────────┴──────────┘            │
├─────────────────────────────────────────────────┤
│  📊 分析サマリー                                │
│  優秀群: 10名 | 分析対象: 94名 | 比率: 10.6%  │
│  学習時間: 2.5分                               │
├─────────────────────────────────────────────────┤
│  📈 分析グラフ                                  │
│  ┌────────────────────┬────────────────────┐   │
│  │ スキル保有率比較   │ 社員スコア分布    │   │
│  │ (Bar Chart)        │ (Histogram)        │   │
│  └────────────────────┴────────────────────┘   │
│  ┌────────────────────┐                        │
│  │ スキル相互作用     │                        │
│  │ (Synergy Bar)      │                        │
│  └────────────────────┘                        │
├─────────────────────────────────────────────────┤
│  🎯 モデル性能                                  │
│  Train AUC: 0.85 | Test AUC: 0.82             │
│  Precision: 0.88 | Recall: 0.80               │
├─────────────────────────────────────────────────┤
│  📋 詳細分析 (Collapsible Sections)            │
│  ▼ 詳細スキル一覧  ▶ 社員スコアランキング    │
│  ▶ 詳細モデル性能   ▶ 因果効果分析            │
│  ▶ スキル相互作用詳細                         │
└─────────────────────────────────────────────────┘
```

#### グラフの種類

| グラフ | ライブラリ | 用途 |
|--------|----------|------|
| Bar Chart | Plotly | スキル比較、因果効果 |
| Histogram | Plotly | スコア分布 |
| Scatter | Plotly/Seaborn | スキル散布図、PCA可視化 |
| Heatmap | Plotly | 相関マトリクス |

---

## データフロー図

### 全体フロー

```
1. CSV Upload
   ├─ member_skillnote.csv
   ├─ acquiredCompetenceLevel.csv
   ├─ skill_skillnote.csv
   ├─ education_skillnote.csv
   └─ license_skillnote.csv
                │
                ▼
2. Data Preprocessing & Validation
   ├─ Column Mapping
   ├─ Type Conversion
   ├─ Missing Value Handling
   └─ Skill Matrix Creation
                │
                ▼
3. Graph Construction
   ├─ Similarity Computation (Cosine)
   ├─ Edge Creation
   └─ Adjacency Matrix
                │
                ▼
4. GNN Training (with Callback)
   ├─ Forward Pass
   ├─ Loss Computation
   ├─ Backward Pass
   ├─ Optimization
   └─ Progress Update
                │
                ▼
5. Skill Importance Analysis
   ├─ Fisher Exact Test
   ├─ FDR Correction
   ├─ Confidence Intervals
   └─ Ranking
                │
                ▼
6. Member Score Calculation
   ├─ Prototype Embedding
   ├─ Similarity Computation
   └─ Score Ranking
                │
                ▼
7. Analysis Execution (Parallel)
   ├─ Model Evaluation
   │  ├─ Holdout or LOOCV
   │  ├─ Metrics Calculation
   │  └─ Overfitting Detection
   │
   ├─ Causal Effect Estimation
   │  ├─ Propensity Score
   │  ├─ Matching
   │  └─ ATE Calculation
   │
   └─ Skill Interaction Analysis
      ├─ 4-Group Comparison
      └─ Synergy Calculation
                │
                ▼
8. Results Aggregation
   ├─ Combine Results
   ├─ Save Models
   └─ Log Execution
                │
                ▼
9. Dashboard Rendering
   ├─ Skill Cards
   ├─ Metrics Display
   ├─ Chart Visualization
   ├─ Model Performance
   └─ Detailed Analysis Sections
                │
                ▼
10. Result Export
    ├─ CSV Download
    └─ Model Versioning
```

---

## パフォーマンス特性

### 時間計算量

| 処理 | 計算量 | 備考 |
|------|--------|------|
| グラフ構築 | O(N²) | N=スキル数 |
| GNN 学習 | O(E × L × H²) | E=エポック、L=層数、H=隠れ次元 |
| スキル重要度 | O(S × M) | S=スキル数、M=社員数 |
| 因果推論 | O(S × M) | マッチングコスト含む |
| 相互作用 | O(S²) | S=スキル数 |

### 空間計算量

| 構造 | メモリ | 備考 |
|------|--------|------|
| スキルマトリクス | O(M × N) | M=社員、N=スキル |
| 隣接行列 | O(M²) | M=社員 |
| GNN 埋め込み | O(M × H) | H=埋め込み次元 |
| モデルパラメータ | O(H² × L) | H=隠れ次元、L=層数 |

---

## 拡張性ポイント

### 追加可能な分析

1. **集団分析**: k-means による優秀群の細分化
2. **時系列分析**: スキル習得の時間軸分析
3. **推奨システム**: スキル習得順序の最適化提案
4. **異常検出**: 異常な行動パターンの検出

### スケーラビリティ対応

1. **分散学習**: DDP による複数GPU対応
2. **バッチ処理**: ストリーミング処理に対応
3. **キャッシング**: 計算結果の段階的キャッシング
4. **API化**: REST API エンドポイント提供

