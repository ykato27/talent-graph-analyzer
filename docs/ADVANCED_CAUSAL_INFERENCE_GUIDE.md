# 拡張因果推論機能ガイド

## 概要

このガイドでは、talent-graph-analyzerに新しく追加された高度な因果推論機能について説明します。

### 新規追加された機能

1. **感度分析（Sensitivity Analysis）**
   - Rosenbaum Bounds
   - E-value計算
   - 隠れた交絡因子の影響評価

2. **クラスター頑健標準誤差（Cluster-Robust SE）**
   - 部門レベルの相関を考慮
   - クラスター内相関係数（ICC）の計算
   - 頑健な統計的推論

3. **PSM診断指標（PSM Diagnostics）**
   - 共変量バランステーブル
   - 標準化平均差（SMD）の計算
   - 傾向スコアの重なり度合い評価
   - Love Plot可視化

4. **差分の差分法（DID Analysis）**
   - 標準的なDID推定
   - 平行トレンド仮定の検定
   - 共変量調整付きDID
   - スタガード採用デザイン対応

5. **Causal Forest**
   - 機械学習ベースのHTE推定
   - サブグループ分析
   - 政策学習

6. **ベースラインモデル比較**
   - Logistic Regression
   - Random Forest
   - XGBoost
   - GNNとの性能比較

7. **クロスバリデーション**
   - Stratified K-Fold CV
   - Leave-One-Out CV
   - Nested CV

8. **可視化ツール**
   - Love Plot
   - Calibration Plot
   - DID Plot

---

## 使用方法

### 1. 感度分析

隠れた交絡因子の影響を評価します。

```python
from causal_inference import sensitivity_analysis

# Rosenbaum Bounds
rosenbaum_results = sensitivity_analysis.rosenbaum_bounds(
    treated_outcomes=treated_outcomes,
    control_outcomes=control_outcomes,
    gamma_values=[1.0, 1.5, 2.0, 2.5, 3.0]
)
print(rosenbaum_results)

# E-value
e_value_results = sensitivity_analysis.calculate_e_value(
    effect_estimate=1.8,  # オッズ比
    effect_se=0.3,
    effect_type="odds_ratio"
)
print(f"E-value: {e_value_results['point_estimate']:.2f}")
print(f"解釈: {e_value_results['interpretation']}")

# 包括的レポート
report = sensitivity_analysis.sensitivity_analysis_report(
    treated_outcomes=treated_outcomes,
    control_outcomes=control_outcomes,
    effect_estimate=1.8,
    effect_se=0.3
)
print(report['summary'])
print(report['recommendation'])
```

**結果の解釈:**
- **Gamma=2まで有意** → 傾向スコアが2倍違っても結果は頑健
- **E-value=3.0** → RR=3.0の隠れた交絡因子が必要（頑健）
- **E-value=1.5** → RR=1.5の隠れた交絡因子で無効化（脆弱）

---

### 2. クラスター頑健標準誤差

部門内の相関を考慮した統計的推論を行います。

```python
from causal_inference import cluster_robust

# クラスター頑健推定
results = cluster_robust.cluster_robust_inference(
    y=outcome,
    treatment=skill_indicator,
    covariates=X_covariates,
    clusters=department_codes,
    treatment_name="Python Skill"
)

print(results)
# 出力例:
# Variable       Coefficient  SE_Regular  SE_Cluster  P_Value  Significance
# Python Skill   0.25         0.05        0.12        0.041    *

# ICC計算
icc = cluster_robust.calculate_icc(outcome, department_codes)
print(f"クラスター内相関係数: {icc:.4f}")

# 推奨事項
recommendation = cluster_robust.recommend_clustering_approach(
    n_clusters=20,
    n_observations=100,
    icc=icc
)
print(recommendation)
```

**結果の解釈:**
- **SE_Cluster > SE_Regular** → クラスタリング効果が存在
- **ICC < 0.05** → クラスタリング効果は小さい
- **ICC > 0.15** → クラスタリング効果が大きい（クラスター頑健SE必須）

---

### 3. PSM診断指標

傾向スコアマッチングの品質を評価します。

```python
from causal_inference import psm_diagnostics

# 共変量バランステーブル
balance_table = psm_diagnostics.covariate_balance_table(
    X_treated_before=X_treated_before,
    X_control_before=X_control_before,
    X_treated_after=X_treated_after,
    X_control_after=X_control_after,
    covariate_names=['勤続年数', '等級', '役職', 'スキル数']
)
print(balance_table)

# 重なり度合いの評価
overlap_results = psm_diagnostics.check_overlap(
    ps_treated=propensity_scores_treated,
    ps_control=propensity_scores_control,
    method="minmax"
)
print(overlap_results['recommendation'])

# 総合品質レポート
quality_report = psm_diagnostics.psm_quality_report(
    balance_table=balance_table,
    overlap_results=overlap_results,
    n_matched_pairs=80,
    n_treated_total=100,
    n_control_total=400
)
print(quality_report['summary'])
print("推奨事項:")
for rec in quality_report['recommendations']:
    print(f"  {rec}")
```

**結果の解釈:**
- **|SMD| < 0.1** → 優れたバランス
- **|SMD| >= 0.2** → 不十分なバランス（カリパー調整が必要）
- **重なり > 90%** → 優れた共通サポート
- **重なり < 50%** → 不十分（トリミング推奨）

---

### 4. DID分析

時系列データを活用した因果推論を行います。

```python
from causal_inference import did_analysis

# 標準的なDID推定
did_results = did_analysis.did_estimation(
    df=panel_data,
    outcome_col='is_excellent',
    treatment_col='has_skill',
    time_col='year',
    unit_col='member_id',
    pre_period=(2020, 2021),
    post_period=(2022, 2023)
)

print(f"DID推定値: {did_results['did_estimate']:.4f}")
print(f"p値: {did_results['p_value']:.4f}")
print(f"95%CI: [{did_results['ci_lower']:.4f}, {did_results['ci_upper']:.4f}]")

# 平行トレンド検定
pt_test = did_results['parallel_trends_test']
print(f"平行トレンド検定: {pt_test['result']}")
print(pt_test['interpretation'])

# 共変量調整付きDID
did_cov_results = did_analysis.did_with_covariates(
    df=panel_data,
    outcome_col='is_excellent',
    treatment_col='has_skill',
    time_col='year',
    unit_col='member_id',
    covariate_cols=['years_of_service', 'grade', 'position']
)
print(f"調整済みDID推定値: {did_cov_results['did_estimate']:.4f}")

# スタガードDID（異なる時点での処置）
staggered_results = did_analysis.staggered_did(
    df=panel_data,
    outcome_col='is_excellent',
    unit_col='member_id',
    time_col='year',
    treatment_date_col='skill_acquisition_date'
)
print(f"全体のATT: {staggered_results['overall_att']:.4f}")
```

**結果の解釈:**
- **DID推定値 > 0かつp < 0.05** → 正の因果効果（統計的に有意）
- **平行トレンド検定 Pass** → DID推定は適切
- **平行トレンド検定 Fail** → 共変量調整DIDを使用

---

### 5. Causal Forest

機械学習ベースの異質的処置効果推定を行います。

```python
from causal_inference import causal_forest

# Causal Forestモデルを学習
model = causal_forest.fit_causal_forest(
    X=X_features,
    T=treatment,
    y=outcome,
    feature_names=['勤続年数', '等級', '役職', 'スキル数'],
    n_estimators=100,
    min_samples_leaf=5
)

# 個人ごとのHTE推定
hte_df = causal_forest.get_heterogeneous_effects(
    model=model,
    X=X_features,
    member_ids=member_ids,
    feature_names=['勤続年数', '等級', '役職', 'スキル数']
)
print(hte_df.head(10))

# サブグループ分析
hte_df, subgroup_stats = causal_forest.identify_subgroups(
    hte_df=hte_df,
    feature_cols=['勤続年数', '等級', '役職'],
    n_clusters=3,
    method='kmeans'
)
print(subgroup_stats)

# 政策学習：誰にスキル開発を推奨すべきか
recommendations = causal_forest.policy_learning(
    hte_df=hte_df,
    budget_constraint=50,  # 最大50人まで
    min_effect_threshold=0.1  # CATE > 0.1
)
print(f"推奨人数: {len(recommendations)}")
print(f"期待効果合計: {recommendations['CATE'].sum():.2f}")
```

**結果の解釈:**
- **CATE > 0** → この人にとって正の効果
- **CATE < 0** → この人にとって負の効果
- **CI が0を跨がない** → 統計的に有意

---

### 6. ベースラインモデル比較

GNNと従来手法を比較します。

```python
from model_evaluation import baseline_models

# ベースラインモデルを準備
models = {
    'Logistic Regression': baseline_models.LogisticRegressionBaseline(),
    'Random Forest': baseline_models.RandomForestBaseline(),
    'XGBoost': baseline_models.XGBoostBaseline()
}

# モデル比較
comparison_df = baseline_models.compare_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_names
)
print(comparison_df)

# GNNとのベンチマーク
benchmark_df = baseline_models.benchmark_against_gnn(
    gnn_predictions=gnn_pred,
    gnn_probabilities=gnn_proba,
    baseline_models={'LR': lr_model, 'RF': rf_model},
    X_test=X_test,
    y_test=y_test
)
print(benchmark_df)
```

**結果の解釈:**
- **Test AUC > Train AUC** → 過学習の可能性
- **GNN AUC > ベースライン** → GNNの優位性を確認
- **Overfit_AUC > 0.2** → 過学習に注意

---

### 7. クロスバリデーション

モデルの汎化性能を評価します。

```python
from model_evaluation import cross_validation

# Stratified K-Fold CV
cv_results = cross_validation.stratified_cv(
    model=model,
    X=X,
    y=y,
    n_splits=5,
    metrics=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
)
print(f"Mean AUC: {cv_results['mean_scores']['roc_auc']:.4f} ± {cv_results['std_scores']['roc_auc']:.4f}")

# Leave-One-Out CV（少数サンプル用）
loocv_results = cross_validation.leave_one_out_cv(
    model=model,
    X=X,
    y=y
)
print(f"LOOCV Accuracy: {loocv_results['accuracy']:.4f}")

# Nested CV（ハイパーパラメータ調整付き）
param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
nested_results = cross_validation.nested_cv(
    model=model,
    X=X,
    y=y,
    param_grid=param_grid,
    n_outer_splits=5,
    n_inner_splits=3
)
print(f"Nested CV Score: {nested_results['mean_score']:.4f}")
```

---

### 8. 可視化

結果を可視化します。

```python
from visualization import love_plot, calibration_plot, did_plots

# Love Plot（共変量バランス）
love_plot.plot_love(
    balance_table=balance_table,
    save_path='./plots/love_plot.png'
)

# Calibration Plot（予測確率の信頼性）
calibration_plot.plot_calibration(
    y_true=y_test,
    y_proba=y_proba,
    n_bins=10,
    save_path='./plots/calibration.png'
)

# DID Plot（平行トレンド）
did_plots.plot_parallel_trends(
    df=panel_data,
    outcome_col='is_excellent',
    time_col='year',
    group_col='treated_group',
    save_path='./plots/parallel_trends.png'
)

# DID効果プロット
did_plots.plot_treatment_effect_over_time(
    did_results=did_results,
    save_path='./plots/did_effect.png'
)
```

---

## 推奨ワークフロー

### Step 1: データ準備
```python
# データ読み込み
df = pd.read_csv('member_data.csv')

# 特徴量エンジニアリング
X = df[['years_of_service', 'grade', 'position', 'skill_count']].values
y = df['is_excellent'].values
treatment = df['has_skill'].values
clusters = df['department_code'].values
```

### Step 2: ベースライン比較
```python
# GNNと従来手法を比較
models = {...}
comparison = compare_models(models, X_train, y_train, X_test, y_test)
```

### Step 3: 因果推論
```python
# PSM + 診断
# -> 感度分析
# -> クラスター頑健推定
```

### Step 4: DID分析（時系列データがある場合）
```python
# DID推定
# -> 平行トレンド検定
```

### Step 5: Causal Forest（より詳細なHTE）
```python
# Causal Forest学習
# -> サブグループ分析
# -> 政策学習
```

### Step 6: レポート生成
```python
# 可視化
# -> 推奨事項まとめ
```

---

## 設定ファイル（config.yaml）

新機能は `config.yaml` で有効化/無効化できます。

```yaml
advanced_causal_inference:
  sensitivity_analysis:
    enabled: true
  cluster_robust:
    enabled: true
  psm_diagnostics:
    enabled: true
  did_analysis:
    enabled: true
  causal_forest:
    enabled: true

baseline_comparison:
  enabled: true

cross_validation:
  enabled: true
  method: "stratified"
  n_splits: 5

visualization:
  enabled: true
```

---

## トラブルシューティング

### Q1: "econml not found"エラー
```bash
pip install econml
```

### Q2: 少数サンプル（n<30）で何を使うべきか？
- LOOCV
- 感度分析（必須）
- クラスター頑健SE

### Q3: 平行トレンド検定がFailする
- 共変量調整DIDを使用
- より短い比較期間を選択
- 別の手法（IV法など）を検討

### Q4: ICCが高い（>0.15）
- クラスター頑健SEが必須
- 固定効果モデルを検討
- 階層モデルを検討

---

## 参考文献

1. Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.). Springer.
2. VanderWeele, T. J., & Ding, P. (2017). "Sensitivity Analysis in Observational Research: Introducing the E-Value." *Annals of Internal Medicine*, 167(4), 268-274.
3. Cameron, A. C., & Miller, D. L. (2015). "A practitioner's guide to cluster-robust inference." *Journal of Human Resources*, 50(2), 317-372.
4. Austin, P. C. (2011). "An introduction to propensity score methods for reducing the effects of confounding in observational studies." *Multivariate Behavioral Research*, 46(3), 399-424.
5. Wager, S., & Athey, S. (2018). "Estimation and inference of heterogeneous treatment effects using random forests." *Journal of the American Statistical Association*, 113(523), 1228-1242.
6. Callaway, B., & Sant'Anna, P. H. (2021). "Difference-in-differences with multiple time periods." *Journal of Econometrics*, 225(2), 200-230.

---

## サポート

質問や問題があれば、GitHubのIssueで報告してください。
