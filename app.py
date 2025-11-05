"""
GNN優秀人材分析システム - Streamlitアプリケーション
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from gnn_talent_analyzer import TalentAnalyzer
from config_loader import get_config

# ページ設定
st.set_page_config(
    page_title=get_config('ui.page_title', 'GNN優秀人材分析システム'),
    page_icon=get_config('ui.page_icon', '🎯'),
    layout="wide"
)

# 設定値の取得
MIN_EXCELLENT = get_config('analysis.min_excellent_members', 3)
MAX_EXCELLENT_RECOMMENDED = get_config('analysis.max_excellent_members_recommended', 20)
ESSENTIAL_THRESHOLD = get_config('analysis.essential_skill_threshold', 0.8)
IMPORTANT_DIFF_THRESHOLD = get_config('analysis.important_skill_diff_threshold', 0.3)
SIGNIFICANT_DIFF_THRESHOLD = get_config('analysis.significant_skill_diff_threshold', 0.2)

COLOR_EXCELLENT = get_config('ui.colors.excellent_group', '#FF6B6B')
COLOR_NON_EXCELLENT = get_config('ui.colors.non_excellent_group', '#4ECDC4')

TOP_SKILLS_CHART = get_config('ui.display.top_skills_chart', 15)
MEMBER_SCORES_HEIGHT = get_config('ui.display.member_scores_height', 400)
CHART_HEIGHT = get_config('ui.display.chart_height', 600)
HISTOGRAM_BINS = get_config('ui.display.histogram_bins', 20)

MIN_EPOCHS = get_config('training.min_epochs', 50)
MAX_EPOCHS = get_config('training.max_epochs', 500)
DEFAULT_EPOCHS = get_config('training.default_epochs', 100)

EXPORT_SKILL_FILE = get_config('files.export.skill_importance', 'skill_importance.csv')
EXPORT_MEMBER_FILE = get_config('files.export.member_scores', 'member_scores.csv')
FILE_ENCODING = get_config('files.encoding', 'utf-8-sig')

# セッション状態の初期化
def initialize_session_state():
    """セッション状態を初期化"""
    session_defaults = {
        'analyzer': None,
        'data_loaded': False,
        'results': None,
        'evaluation_results': None,
        'causal_results': None,
        'interaction_results': None,
        'member_df': None,
    }
    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# ==================== UIコンポーネント関数 ====================
def render_header():
    """ヘッダーを描画"""
    st.title(f"{get_config('ui.page_icon', '🎯')} {get_config('ui.page_title', 'GNN優秀人材分析システム')}")
    st.markdown("---")

def render_data_upload_sidebar():
    """データアップロードサイドバーを描画"""
    st.sidebar.header("📁 データアップロード")

    uploaded_files = {
        'member': st.sidebar.file_uploader("社員マスタ (member_skillnote.csv)", type=['csv']),
        'acquired': st.sidebar.file_uploader("スキル習得データ (acquiredCompetenceLevel.csv)", type=['csv']),
        'skill': st.sidebar.file_uploader("スキルマスタ (skill_skillnote.csv)", type=['csv']),
        'education': st.sidebar.file_uploader("教育マスタ (education_skillnote.csv)", type=['csv']),
        'license': st.sidebar.file_uploader("資格マスタ (license_skillnote.csv)", type=['csv'])
    }

    if st.sidebar.button("📊 データ読み込み"):
        if all(uploaded_files.values()):
            try:
                with st.spinner("データ読み込み中..."):
                    # CSVファイルを読み込み
                    member_df = pd.read_csv(uploaded_files['member'], encoding=FILE_ENCODING)
                    acquired_df = pd.read_csv(uploaded_files['acquired'], encoding=FILE_ENCODING)
                    skill_df = pd.read_csv(uploaded_files['skill'], encoding=FILE_ENCODING)
                    education_df = pd.read_csv(uploaded_files['education'], encoding=FILE_ENCODING)
                    license_df = pd.read_csv(uploaded_files['license'], encoding=FILE_ENCODING)

                    # アナライザーの初期化
                    analyzer = TalentAnalyzer()
                    analyzer.load_data(member_df, acquired_df, skill_df, education_df, license_df)

                    st.session_state.analyzer = analyzer
                    st.session_state.member_df = member_df
                    st.session_state.data_loaded = True

                    st.sidebar.success("✅ データ読み込み完了")
            except Exception as e:
                st.sidebar.error(f"❌ エラー: {str(e)}")
        else:
            st.sidebar.warning("⚠️ すべてのCSVファイルをアップロードしてください")

    st.sidebar.markdown("---")

# ==================== メイン処理 ====================
render_header()
render_data_upload_sidebar()

# メインコンテンツ
if st.session_state.data_loaded:
    analyzer = st.session_state.analyzer

    # データ概要
    with st.expander("📊 データ概要", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("総社員数", len(analyzer.members))
        with col2:
            st.metric("スキル種類数", len(analyzer.skill_codes))
        with col3:
            avg_skills = np.mean(np.sum(analyzer.skill_matrix > 0, axis=1))
            st.metric("平均スキル保有数", f"{avg_skills:.1f}")
        with col4:
            sparsity = 1 - np.count_nonzero(analyzer.skill_matrix) / analyzer.skill_matrix.size
            st.metric("データスパース性", f"{sparsity*100:.1f}%")

    st.markdown("---")

    # 優秀人材選択
    st.header("1️⃣ 優秀人材の選択")

    # 社員リストの表示
    member_list = []
    for member_code in analyzer.members:
        member_name = analyzer.member_names.get(member_code, '不明')
        n_skills = int(np.sum(analyzer.skill_matrix[analyzer.member_to_idx[member_code]] > 0))
        member_list.append({
            'コード': member_code,
            '名前': member_name,
            'スキル保有数': n_skills
        })

    member_df_display = pd.DataFrame(member_list)

    # 選択方法
    selection_method = st.radio(
        "選択方法",
        ["手動選択", "スキル保有数上位を自動選択"],
        horizontal=True
    )

    if selection_method == "手動選択":
        # マルチセレクト
        selected_members = st.multiselect(
            f"優秀な社員を選択してください（5-{MAX_EXCELLENT_RECOMMENDED}名推奨）",
            options=member_df_display['コード'].tolist(),
            format_func=lambda x: f"{member_df_display[member_df_display['コード']==x]['名前'].values[0]} ({x})"
        )
    else:
        # 上位N名を自動選択
        n_top = st.slider(
            "上位何名を選択しますか？",
            min_value=MIN_EXCELLENT,
            max_value=MAX_EXCELLENT_RECOMMENDED,
            value=10
        )
        top_members = member_df_display.nlargest(n_top, 'スキル保有数')
        selected_members = top_members['コード'].tolist()

        st.info(f"スキル保有数上位{n_top}名を自動選択しました")
        st.dataframe(top_members, use_container_width=True)

    st.markdown(f"**選択された社員数: {len(selected_members)}名**")

    if len(selected_members) < MIN_EXCELLENT:
        st.warning(f"⚠️ 最低{MIN_EXCELLENT}名以上の優秀人材を選択してください")
    elif len(selected_members) > MAX_EXCELLENT_RECOMMENDED:
        st.warning(f"⚠️ {MAX_EXCELLENT_RECOMMENDED}名以下での選択を推奨します")

    st.markdown("---")

    # 分析実行
    st.header("2️⃣ 分析実行")

    col1, col2 = st.columns([1, 3])

    with col1:
        epochs = st.number_input(
            "学習エポック数",
            min_value=MIN_EPOCHS,
            max_value=MAX_EPOCHS,
            value=DEFAULT_EPOCHS,
            step=50,
            help="学習の反復回数。多いほど精度が上がりますが時間がかかります"
        )

    with col2:
        epoch_recs = get_config('ui.epoch_recommendations', {})
        small = epoch_recs.get('small_group', [50, 100])
        medium = epoch_recs.get('medium_group', [100, 200])
        large = epoch_recs.get('large_group', [200, 300])

        st.info(f"""
        **推奨設定**
        - 優秀群5名以下: {small[0]}-{small[1]}エポック
        - 優秀群10名程度: {medium[0]}-{medium[1]}エポック
        - 優秀群20名以上: {large[0]}-{large[1]}エポック
        """)

    if st.button("🚀 分析開始", type="primary", disabled=(len(selected_members) < MIN_EXCELLENT)):
        try:
            # 学習実行用のプログレス表示UI
            progress_placeholder = st.empty()
            metrics_cols = st.columns(4)
            epoch_metric = metrics_cols[0].empty()
            loss_metric = metrics_cols[1].empty()
            elapsed_metric = metrics_cols[2].empty()
            remaining_metric = metrics_cols[3].empty()

            def on_epoch_callback(epoch_info):
                """各エポック終了時に呼び出されるコールバック関数"""
                with progress_placeholder.container():
                    # 進捗バー
                    st.progress(epoch_info['progress'])

                # メトリクスを更新
                epoch_metric.metric(
                    "エポック",
                    f"{epoch_info['epoch']}/{epoch_info['epochs']}"
                )
                loss_metric.metric(
                    "ロス",
                    f"{epoch_info['loss']:.4f}"
                )

                # 時間をフォーマット
                def format_time(seconds):
                    if seconds < 60:
                        return f"{seconds:.1f}秒"
                    elif seconds < 3600:
                        return f"{seconds/60:.1f}分"
                    else:
                        return f"{seconds/3600:.1f}時"

                elapsed_metric.metric(
                    "経過時間",
                    format_time(epoch_info['elapsed_time'])
                )
                remaining_metric.metric(
                    "推定残り時間",
                    format_time(epoch_info['estimated_remaining_time'])
                )

            # 学習実行（コールバック関数付き）
            analyzer.train(selected_members, epochs_unsupervised=epochs, on_epoch_callback=on_epoch_callback)

            # 学習完了メッセージ
            progress_placeholder.empty()
            if hasattr(analyzer.gnn, 'last_training_time') and analyzer.gnn.last_training_time is not None:
                training_time_seconds = analyzer.gnn.last_training_time
                if training_time_seconds < 60:
                    time_str = f"{training_time_seconds:.1f}秒"
                else:
                    time_str = f"{training_time_seconds/60:.1f}分"
                st.success(f"✅ GNN学習完了 - 総学習時間: {time_str}")
            else:
                st.success("✅ GNN学習完了")

            # 基本分析
            with st.spinner("分析実行中..."):
                results = analyzer.analyze(selected_members)
                st.session_state.results = results

            # モデル評価
            eval_config = get_config('evaluation', {})
            if eval_config.get('enabled', True):
                with st.spinner("モデル評価を実行中..."):
                    evaluation_results = analyzer.evaluate_model(selected_members, epochs_unsupervised=epochs)
                    st.session_state.evaluation_results = evaluation_results
            else:
                st.session_state.evaluation_results = None

            # 因果推論
            causal_config = get_config('causal_inference', {})
            if causal_config.get('enabled', True):
                with st.spinner("因果推論を実行中..."):
                    causal_results = analyzer.estimate_causal_effects(selected_members)
                    st.session_state.causal_results = causal_results
            else:
                st.session_state.causal_results = None

            # スキル相互作用分析
            interaction_config = get_config('skill_interaction', {})
            if interaction_config.get('enabled', True):
                with st.spinner("スキル相互作用を分析中..."):
                    interaction_results = analyzer.analyze_skill_interactions(selected_members)
                    st.session_state.interaction_results = interaction_results
            else:
                st.session_state.interaction_results = None

            # モデル保存
            versioning_config = get_config('versioning', {})
            if versioning_config.get('enabled', True) and versioning_config.get('save_models', True):
                analyzer.save_model(selected_members)

            st.success("✅ 分析完了！")
        except Exception as e:
            st.error(f"❌ エラーが発生しました: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

    st.markdown("---")

    # 結果表示
    if st.session_state.results is not None:
        results = st.session_state.results

        st.header("3️⃣ 分析結果")

        # サマリー
        with st.expander("📋 分析サマリー", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("優秀群", f"{results['n_excellent']}名")
            with col2:
                st.metric("分析対象", f"{results['n_total']}名")
            with col3:
                coverage = results['n_excellent'] / results['n_total'] * 100
                st.metric("優秀群比率", f"{coverage:.1f}%")

        # タブで結果を分割表示
        tabs = [
            "🎯 重要スキルランキング",
            "👥 社員スコアランキング",
            "📊 スキル比較分析",
            "🗺️ 埋め込み可視化"
        ]

        # 条件付きでタブを追加
        if st.session_state.evaluation_results is not None:
            tabs.append("📈 モデル性能")
        if st.session_state.causal_results is not None:
            tabs.append("🔬 因果効果")
        if st.session_state.interaction_results is not None:
            tabs.append("🔗 スキル相互作用")

        tab_objects = st.tabs(tabs)
        tab_idx = 0
        tab1 = tab_objects[tab_idx]; tab_idx += 1
        tab2 = tab_objects[tab_idx]; tab_idx += 1
        tab3 = tab_objects[tab_idx]; tab_idx += 1
        tab4 = tab_objects[tab_idx]; tab_idx += 1
        tab5 = tab_objects[tab_idx] if tab_idx < len(tab_objects) else None; tab_idx += 1 if tab5 is not None else 0
        tab6 = tab_objects[tab_idx] if tab_idx < len(tab_objects) else None; tab_idx += 1 if tab6 is not None else 0
        tab7 = tab_objects[tab_idx] if tab_idx < len(tab_objects) else None

        with tab1:
            st.subheader(f"優秀群に特徴的なスキル Top{MAX_EXCELLENT_RECOMMENDED}")

            skill_df = pd.DataFrame(results['skill_importance'][:MAX_EXCELLENT_RECOMMENDED])

            # 表示用にフォーマット
            skill_df_display = skill_df.copy()
            skill_df_display['優秀群保有率'] = skill_df_display['excellent_rate'].apply(lambda x: f"{x*100:.1f}%")
            skill_df_display['非優秀群保有率'] = skill_df_display['non_excellent_rate'].apply(lambda x: f"{x*100:.1f}%")
            skill_df_display['差分'] = skill_df_display['rate_diff'].apply(lambda x: f"+{x*100:.1f}%" if x > 0 else f"{x*100:.1f}%")
            skill_df_display['重要度'] = skill_df_display['importance_score'].apply(lambda x: f"{x:.3f}")

            # 統計的有意性を追加
            if 'p_adjusted' in skill_df.columns and 'significance_level' in skill_df.columns:
                skill_df_display['有意性'] = skill_df_display['significance_level']
                skill_df_display['p値'] = skill_df_display['p_adjusted'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                columns_to_show = ['skill_name', '優秀群保有率', '非優秀群保有率', '差分', '重要度', '有意性', 'p値']
            else:
                columns_to_show = ['skill_name', '優秀群保有率', '非優秀群保有率', '差分', '重要度']

            st.dataframe(
                skill_df_display[columns_to_show],
                use_container_width=True
            )

            # 有意性に関する説明を表示
            if 'significance_level' in skill_df.columns:
                st.info("""
                **有意性マーク**:
                - *** : p < 0.001（非常に高い有意性）
                - ** : p < 0.01（高い有意性）
                - * : p < 0.05（有意）
                - n.s. : 有意差なし
                """)

            # 棒グラフ
            fig = go.Figure()

            top_skills = results['skill_importance'][:TOP_SKILLS_CHART]

            fig.add_trace(go.Bar(
                x=[s['excellent_rate']*100 for s in top_skills],
                y=[s['skill_name'] for s in top_skills],
                orientation='h',
                name='優秀群',
                marker_color=COLOR_EXCELLENT
            ))

            fig.add_trace(go.Bar(
                x=[s['non_excellent_rate']*100 for s in top_skills],
                y=[s['skill_name'] for s in top_skills],
                orientation='h',
                name='非優秀群',
                marker_color=COLOR_NON_EXCELLENT
            ))

            fig.update_layout(
                title=f"スキル保有率比較 Top{TOP_SKILLS_CHART}",
                xaxis_title="保有率 (%)",
                yaxis_title="スキル名",
                barmode='group',
                height=CHART_HEIGHT,
                yaxis={'categoryorder': 'total ascending'}
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("社員の優秀度スコアランキング")

            # 全社員のスコア
            member_scores_df = pd.DataFrame(results['member_scores'])
            member_scores_df['is_excellent_label'] = member_scores_df['is_excellent'].apply(
                lambda x: '✅ 優秀群' if x else ''
            )

            # 表示
            st.dataframe(
                member_scores_df[['member_name', 'score', 'is_excellent_label']].rename(columns={
                    'member_name': '社員名',
                    'score': '優秀度スコア',
                    'is_excellent_label': ''
                }),
                use_container_width=True,
                height=MEMBER_SCORES_HEIGHT
            )

            # 分布のヒストグラム
            fig = go.Figure()

            excellent_scores = member_scores_df[member_scores_df['is_excellent']]['score']
            non_excellent_scores = member_scores_df[~member_scores_df['is_excellent']]['score']

            fig.add_trace(go.Histogram(
                x=excellent_scores,
                name='優秀群',
                opacity=0.7,
                marker_color=COLOR_EXCELLENT,
                nbinsx=HISTOGRAM_BINS
            ))

            fig.add_trace(go.Histogram(
                x=non_excellent_scores,
                name='非優秀群',
                opacity=0.7,
                marker_color=COLOR_NON_EXCELLENT,
                nbinsx=HISTOGRAM_BINS
            ))

            fig.update_layout(
                title="優秀度スコアの分布",
                xaxis_title="優秀度スコア",
                yaxis_title="人数",
                barmode='overlay',
                height=MEMBER_SCORES_HEIGHT
            )

            st.plotly_chart(fig, use_container_width=True)

            # 統計情報
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**優秀群の統計**")
                st.write(f"平均スコア: {excellent_scores.mean():.2f}")
                st.write(f"標準偏差: {excellent_scores.std():.2f}")
                st.write(f"最小値: {excellent_scores.min():.2f}")
                st.write(f"最大値: {excellent_scores.max():.2f}")

            with col2:
                st.markdown("**非優秀群の統計**")
                st.write(f"平均スコア: {non_excellent_scores.mean():.2f}")
                st.write(f"標準偏差: {non_excellent_scores.std():.2f}")
                st.write(f"最小値: {non_excellent_scores.min():.2f}")
                st.write(f"最大値: {non_excellent_scores.max():.2f}")

        with tab3:
            st.subheader("優秀群と非優秀群のスキル比較")

            # 保有率の差が大きいスキルを抽出
            significant_skills = [
                s for s in results['skill_importance']
                if abs(s['rate_diff']) > SIGNIFICANT_DIFF_THRESHOLD
            ][:30]

            if len(significant_skills) > 0:
                # スキルカテゴリごとの分析（簡易版）
                st.markdown(f"### 保有率差が大きいスキル（差分{SIGNIFICANT_DIFF_THRESHOLD*100:.0f}%以上）")

                diff_df = pd.DataFrame(significant_skills)
                diff_df_display = diff_df.copy()
                diff_df_display['差分'] = diff_df_display['rate_diff'].apply(lambda x: f"{x*100:.1f}%")
                diff_df_display['優秀群保有率'] = diff_df_display['excellent_rate'].apply(lambda x: f"{x*100:.1f}%")

                st.dataframe(
                    diff_df_display[['skill_name', '差分', '優秀群保有率']].rename(columns={
                        'skill_name': 'スキル名',
                        '差分': '保有率差分',
                        '優秀群保有率': '優秀群保有率'
                    }),
                    use_container_width=True
                )

                # 散布図
                all_skills_df = pd.DataFrame(results['skill_importance'])

                fig = px.scatter(
                    all_skills_df,
                    x='non_excellent_rate',
                    y='excellent_rate',
                    hover_data=['skill_name'],
                    labels={
                        'non_excellent_rate': '非優秀群保有率',
                        'excellent_rate': '優秀群保有率'
                    },
                    title='スキル保有率の散布図'
                )

                # 対角線を追加
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    name='同一保有率',
                    showlegend=True
                ))

                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                st.info("対角線より上にあるスキルは優秀群で保有率が高いスキルです")
            else:
                st.info("保有率差が大きいスキルが見つかりませんでした")

        with tab4:
            st.subheader("GNN埋め込み空間の可視化")

            st.info("GNNによって学習された社員の潜在表現を2次元に圧縮して可視化しています")

            # PCAで2次元に削減
            from sklearn.decomposition import PCA

            embeddings = results['embeddings']
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)

            # データフレーム作成
            viz_df = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'member_name': [analyzer.member_names.get(m, '不明') for m in analyzer.members],
                'is_excellent': [i in results['excellent_indices'] for i in range(len(analyzer.members))]
            })

            # 散布図
            fig = px.scatter(
                viz_df,
                x='x',
                y='y',
                color='is_excellent',
                hover_data=['member_name'],
                labels={
                    'x': f'第1主成分 (寄与率: {pca.explained_variance_ratio_[0]*100:.1f}%)',
                    'y': f'第2主成分 (寄与率: {pca.explained_variance_ratio_[1]*100:.1f}%)',
                    'is_excellent': '優秀群'
                },
                title='社員の埋め込み表現（2次元PCA）',
                color_discrete_map={True: COLOR_EXCELLENT, False: COLOR_NON_EXCELLENT}
            )

            fig.update_traces(marker=dict(size=10))
            fig.update_layout(height=CHART_HEIGHT)

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **解釈のポイント**
            - 赤い点が優秀群、青い点が非優秀群
            - 近い位置にある社員は似たスキルプロファイルを持つ
            - 優秀群が集まっている領域が「優秀な人材の特徴空間」
            """)

        # モデル性能タブ
        if tab5 is not None:
            with tab5:
                st.subheader("モデル性能評価")

                evaluation_results = st.session_state.evaluation_results

                if evaluation_results is None:
                    st.info("モデル評価が実行されていません")
                else:
                    method = evaluation_results.get('method', 'unknown')

                    if method == 'holdout':
                        st.markdown("### Holdout法による評価")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### 訓練データ")
                            train_metrics = evaluation_results.get('train_metrics', {})
                            st.metric("AUC", f"{train_metrics.get('auc', 0):.3f}")
                            st.metric("Precision", f"{train_metrics.get('precision', 0):.3f}")
                            st.metric("Recall", f"{train_metrics.get('recall', 0):.3f}")
                            st.metric("F1スコア", f"{train_metrics.get('f1', 0):.3f}")
                            st.metric("サンプル数", f"{evaluation_results.get('n_train', 0)}名")

                        with col2:
                            st.markdown("#### テストデータ")
                            test_metrics = evaluation_results.get('test_metrics', {})
                            st.metric("AUC", f"{test_metrics.get('auc', 0):.3f}")
                            st.metric("Precision", f"{test_metrics.get('precision', 0):.3f}")
                            st.metric("Recall", f"{test_metrics.get('recall', 0):.3f}")
                            st.metric("F1スコア", f"{test_metrics.get('f1', 0):.3f}")
                            st.metric("サンプル数", f"{evaluation_results.get('n_test', 0)}名")

                        # 過学習の警告
                        if evaluation_results.get('is_overfitting', False):
                            st.warning(f"""
                            ⚠️ **過学習の可能性があります**

                            訓練データとテストデータのAUC差分が{evaluation_results.get('auc_diff', 0):.3f}と大きいため、
                            モデルが訓練データに過適合している可能性があります。

                            **改善案:**
                            - 優秀群の人数を増やす
                            - 学習エポック数を減らす
                            - ドロップアウト率を上げる（config.yamlで設定）
                            """)
                        else:
                            st.success("✅ 過学習の兆候は見られません")

                    elif method == 'loocv':
                        st.markdown("### LOOCV（Leave-One-Out Cross-Validation）による評価")

                        metrics = evaluation_results.get('metrics', {})

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("AUC", f"{metrics.get('auc', 0):.3f}")
                        with col2:
                            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
                        with col3:
                            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                        with col4:
                            st.metric("F1スコア", f"{metrics.get('f1', 0):.3f}")

                        st.info(f"交差検証数: {evaluation_results.get('n_folds', 0)}回（Leave-One-Out）")

                    # メトリクスの解釈ガイド
                    with st.expander("📘 評価指標の解釈ガイド"):
                        st.markdown("""
                        **AUC (Area Under the ROC Curve)**
                        - 0.5: ランダム（性能なし）
                        - 0.7-0.8: まあまあ
                        - 0.8-0.9: 良好
                        - 0.9以上: 優秀

                        **Precision（精度）**
                        - 優秀と予測した中で、実際に優秀だった割合
                        - 高いほど誤検出が少ない

                        **Recall（再現率）**
                        - 実際の優秀群のうち、正しく検出できた割合
                        - 高いほど見逃しが少ない

                        **F1スコア**
                        - PrecisionとRecallの調和平均
                        - バランスの取れた指標
                        """)

        # 因果効果タブ
        if tab6 is not None:
            with tab6:
                st.subheader("スキルの因果効果推定")

                causal_results = st.session_state.causal_results

                if causal_results is None or len(causal_results) == 0:
                    st.info("因果推論の結果がありません")
                else:
                    st.markdown("""
                    **因果推論とは**: 傾向スコアマッチングにより、勤続年数・等級・役職などの交絡因子を調整し、
                    スキルの**純粋な効果**を推定します。「このスキルを習得すると優秀になる確率がX%変化する」
                    という因果関係を定量化します。
                    """)

                    # 有意な因果効果のみをフィルタ
                    significant_causal = [r for r in causal_results if r.get('status') == 'success' and r.get('significant', False)]

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("分析スキル数", len(causal_results))
                    with col2:
                        st.metric("有意な因果効果", len(significant_causal))
                    with col3:
                        successful = len([r for r in causal_results if r.get('status') == 'success'])
                        st.metric("マッチング成功", f"{successful}個")

                    # Top20の因果効果を表示
                    st.markdown("### 因果効果が大きいスキル Top20")

                    causal_df = pd.DataFrame([r for r in causal_results if r.get('causal_effect') is not None][:20])

                    if len(causal_df) > 0:
                        causal_df_display = causal_df.copy()
                        causal_df_display['因果効果'] = causal_df_display['causal_effect'].apply(lambda x: f"{x*100:+.1f}%")
                        causal_df_display['p値'] = causal_df_display['p_value'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                        causal_df_display['マッチングペア数'] = causal_df_display['n_matched_pairs']
                        causal_df_display['有意'] = causal_df_display['significant'].apply(lambda x: '✓' if x else '')

                        if 'ci_lower' in causal_df_display.columns and 'ci_upper' in causal_df_display.columns:
                            causal_df_display['95%CI'] = causal_df_display.apply(
                                lambda row: f"[{row['ci_lower']*100:.1f}%, {row['ci_upper']*100:.1f}%]"
                                if pd.notna(row['ci_lower']) and pd.notna(row['ci_upper']) else "N/A",
                                axis=1
                            )
                            columns_to_show = ['skill_name', '因果効果', '95%CI', 'p値', 'マッチングペア数', '有意']
                        else:
                            columns_to_show = ['skill_name', '因果効果', 'p値', 'マッチングペア数', '有意']

                        st.dataframe(
                            causal_df_display[columns_to_show].rename(columns={'skill_name': 'スキル名'}),
                            use_container_width=True
                        )

                        # 因果効果の可視化
                        st.markdown("### 因果効果の可視化")

                        fig = go.Figure()

                        top_effects = causal_df.head(15)

                        colors = ['green' if x > 0 else 'red' for x in top_effects['causal_effect']]

                        fig.add_trace(go.Bar(
                            x=top_effects['causal_effect'] * 100,
                            y=top_effects['skill_name'],
                            orientation='h',
                            marker_color=colors,
                            text=[f"{x*100:+.1f}%" for x in top_effects['causal_effect']],
                            textposition='outside'
                        ))

                        fig.update_layout(
                            title="スキルの因果効果（優秀になる確率の変化）",
                            xaxis_title="因果効果 (%)",
                            yaxis_title="スキル名",
                            height=500,
                            yaxis={'categoryorder': 'total ascending'}
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        st.info("""
                        **解釈のポイント**:
                        - プラスの値: スキルを習得すると優秀になる確率が上昇
                        - マイナスの値: スキルと優秀さに負の関係（稀）
                        - 95%CIが0を跨がない場合、統計的に有意
                        """)
                    else:
                        st.warning("因果効果を推定できたスキルがありません")

                    # マッチング失敗の理由
                    failed = [r for r in causal_results if r.get('status') != 'success']
                    if len(failed) > 0:
                        with st.expander(f"⚠️ マッチング失敗スキル ({len(failed)}個)"):
                            failed_df = pd.DataFrame(failed)
                            st.dataframe(
                                failed_df[['skill_name', 'interpretation']].rename(columns={
                                    'skill_name': 'スキル名',
                                    'interpretation': '理由'
                                }),
                                use_container_width=True
                            )

        # スキル相互作用タブ
        if tab7 is not None:
            with tab7:
                st.subheader("スキル相互作用分析")

                interaction_results = st.session_state.interaction_results

                if interaction_results is None or len(interaction_results) == 0:
                    st.info("スキル相互作用の結果がありません")
                else:
                    st.markdown("""
                    **相互作用とは**: 2つのスキルを組み合わせることで、
                    それぞれ単独で持つよりも**大きな効果**が得られることです。
                    相乗効果（シナジー）を定量化し、効果的なスキル組み合わせを発見します。
                    """)

                    st.metric("発見された相乗効果", f"{len(interaction_results)}組")

                    # Top15の相互作用を表示
                    st.markdown("### 相乗効果が大きいスキル組み合わせ Top15")

                    interaction_df = pd.DataFrame(interaction_results[:15])

                    if len(interaction_df) > 0:
                        interaction_df_display = interaction_df.copy()
                        interaction_df_display['スキルA'] = interaction_df_display['skill_a_name']
                        interaction_df_display['スキルB'] = interaction_df_display['skill_b_name']
                        interaction_df_display['相乗効果'] = interaction_df_display['synergy'].apply(lambda x: f"+{x*100:.1f}%")
                        interaction_df_display['両方保有時の優秀率'] = interaction_df_display['rate_both'].apply(lambda x: f"{x*100:.0f}%")
                        interaction_df_display['A単独'] = interaction_df_display['rate_a'].apply(lambda x: f"{x*100:.0f}%")
                        interaction_df_display['B単独'] = interaction_df_display['rate_b'].apply(lambda x: f"{x*100:.0f}%")
                        interaction_df_display['どちらもなし'] = interaction_df_display['rate_neither'].apply(lambda x: f"{x*100:.0f}%")

                        st.dataframe(
                            interaction_df_display[['スキルA', 'スキルB', '相乗効果', '両方保有時の優秀率', 'A単独', 'B単独', 'どちらもなし']],
                            use_container_width=True
                        )

                        # 相乗効果の可視化
                        st.markdown("### 相乗効果の可視化")

                        top_interactions = interaction_results[:10]

                        fig = go.Figure()

                        labels = [f"{r['skill_a_name']}\n+\n{r['skill_b_name']}" for r in top_interactions]
                        synergies = [r['synergy'] * 100 for r in top_interactions]

                        fig.add_trace(go.Bar(
                            x=synergies,
                            y=labels,
                            orientation='h',
                            marker_color='purple',
                            text=[f"+{s:.1f}%" for s in synergies],
                            textposition='outside'
                        ))

                        fig.update_layout(
                            title="スキル相乗効果 Top10",
                            xaxis_title="相乗効果 (%)",
                            yaxis_title="スキル組み合わせ",
                            height=500,
                            yaxis={'categoryorder': 'total ascending'}
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # 詳細な内訳を表示
                        st.markdown("### 効果の内訳（Top5）")

                        for idx, interaction in enumerate(interaction_results[:5]):
                            with st.expander(f"{idx+1}. {interaction['skill_a_name']} + {interaction['skill_b_name']}"):
                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    st.metric("どちらもなし", f"{interaction['rate_neither']*100:.0f}%",
                                             help=f"サンプル数: {interaction['n_neither']}名")
                                with col2:
                                    st.metric(f"{interaction['skill_a_name']}のみ",
                                             f"{interaction['rate_a']*100:.0f}%",
                                             delta=f"+{interaction['effect_a']*100:.1f}%",
                                             help=f"サンプル数: {interaction['n_a']}名")
                                with col3:
                                    st.metric(f"{interaction['skill_b_name']}のみ",
                                             f"{interaction['rate_b']*100:.0f}%",
                                             delta=f"+{interaction['effect_b']*100:.1f}%",
                                             help=f"サンプル数: {interaction['n_b']}名")
                                with col4:
                                    st.metric("両方保有",
                                             f"{interaction['rate_both']*100:.0f}%",
                                             delta=f"+{interaction['actual_effect']*100:.1f}%",
                                             help=f"サンプル数: {interaction['n_both']}名")

                                st.markdown(f"""
                                **相加効果**: {interaction['additive_effect']*100:.1f}%
                                （A単独 + B単独の効果を足し合わせたもの）

                                **実際の効果**: {interaction['actual_effect']*100:.1f}%

                                **相乗効果**: {interaction['synergy']*100:.1f}%
                                （実際の効果 - 相加効果 = **追加で得られる効果**）
                                """)

                        st.info("""
                        **解釈のポイント**:
                        - 相乗効果が大きいほど、両スキルを組み合わせる価値が高い
                        - 育成計画では、相乗効果のあるスキルをセットで習得させることが効果的
                        - 単独では効果が小さくても、組み合わせると大きな効果を発揮するスキルペアに注目
                        """)
                    else:
                        st.warning("相乗効果のあるスキル組み合わせが見つかりませんでした")

        # 推奨育成プラン
        st.markdown("---")
        st.header("4️⃣ 推奨育成プラン")

        # 必須スキル（優秀群の80%以上が保有）
        essential_skills = [
            s for s in results['skill_importance']
            if s['excellent_rate'] >= ESSENTIAL_THRESHOLD
        ]

        # 重要スキル（保有率差が大きい）
        important_skills = [
            s for s in results['skill_importance']
            if s['rate_diff'] >= IMPORTANT_DIFF_THRESHOLD
        ][:10]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 必須スキル")
            st.markdown(f"優秀群の{ESSENTIAL_THRESHOLD*100:.0f}%以上が保有しているスキル")

            if len(essential_skills) > 0:
                for skill in essential_skills[:10]:
                    st.markdown(f"- **{skill['skill_name']}** (保有率: {skill['excellent_rate']*100:.0f}%)")
            else:
                st.info("該当するスキルはありません")

        with col2:
            st.subheader("⭐ 差別化スキル")
            st.markdown("優秀群と非優秀群で保有率差が大きいスキル")

            for skill in important_skills:
                st.markdown(f"- **{skill['skill_name']}** (差分: +{skill['rate_diff']*100:.0f}%)")

        # ダウンロード
        st.markdown("---")
        st.header("5️⃣ 結果のダウンロード")

        col1, col2 = st.columns(2)

        with col1:
            # スキル重要度のCSV
            skill_export = pd.DataFrame(results['skill_importance'])
            csv_skills = skill_export.to_csv(index=False, encoding=FILE_ENCODING)

            st.download_button(
                label="📥 重要スキル一覧をダウンロード",
                data=csv_skills,
                file_name=EXPORT_SKILL_FILE,
                mime="text/csv"
            )

        with col2:
            # 社員スコアのCSV
            member_export = pd.DataFrame(results['member_scores'])
            csv_members = member_export.to_csv(index=False, encoding=FILE_ENCODING)

            st.download_button(
                label="📥 社員スコア一覧をダウンロード",
                data=csv_members,
                file_name=EXPORT_MEMBER_FILE,
                mime="text/csv"
            )

else:
    # データ未読み込み時の表示
    st.info("👈 左のサイドバーからCSVファイルをアップロードしてください")

    st.markdown(f"""
    ### 📝 使い方

    1. **データアップロード**
       - 5つのCSVファイルをアップロード
       - データ読み込みボタンをクリック

    2. **優秀人材の選択**
       - 優秀と考える社員を5-{MAX_EXCELLENT_RECOMMENDED}名程度選択
       - または上位N名を自動選択

    3. **分析実行**
       - 学習エポック数を設定（推奨: {DEFAULT_EPOCHS}）
       - 分析開始ボタンをクリック

    4. **結果の確認**
       - 重要スキルランキング
       - 社員スコアランキング
       - スキル比較分析
       - 埋め込み可視化

    5. **結果のダウンロード**
       - CSV形式で結果をダウンロード可能

    ### 🔬 技術的特徴

    - **Graph Neural Network (GNN)** による高度な関係性学習
    - **半教師あり学習** でラベルなしデータも活用
    - **Few-shot学習** で少数サンプルでも高精度
    - CPU環境で動作（GPUは不要）

    ### ⚙️ 推奨設定

    - 優秀群: 5-{MAX_EXCELLENT_RECOMMENDED}名（最低{MIN_EXCELLENT}名）
    - 対象社員: 50名以上推奨
    - 学習エポック: {DEFAULT_EPOCHS}-{MAX_EPOCHS//2}
    """)

# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
GNN優秀人材分析システム v1.0 | Powered by Graph Neural Networks
</div>
""", unsafe_allow_html=True)
