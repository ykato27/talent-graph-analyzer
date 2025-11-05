"""
GNN優秀人材分析システム - Streamlitアプリケーション
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from talent_analyzer.core.analyzer import TalentAnalyzer
from talent_analyzer.config.loader import get_config

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

# ==================== ダッシュボード用コンポーネント関数 ====================
def render_skill_cards(skill_importance, top_n=3):
    """Top Nのスキルをカード型で表示"""
    st.subheader("🎯 Top スキル")

    cols = st.columns(top_n)
    for idx, col in enumerate(cols):
        if idx < len(skill_importance):
            skill = skill_importance[idx]
            with col:
                st.metric(
                    label=skill['skill_name'],
                    value=f"{skill['importance_score']:.2f}",
                    delta=f"差分: {skill['rate_diff']*100:.1f}%"
                )
                st.caption(f"優秀群: {skill['excellent_rate']*100:.1f}%")

def render_analysis_metrics(results, analyzer):
    """分析メトリクスを表示"""
    st.subheader("📊 分析サマリー")

    cols = st.columns(4)

    with cols[0]:
        st.metric("優秀群", f"{results['n_excellent']}名")

    with cols[1]:
        st.metric("分析対象", f"{results['n_total']}名")

    with cols[2]:
        coverage = results['n_excellent'] / results['n_total'] * 100
        st.metric("優秀群比率", f"{coverage:.1f}%")

    with cols[3]:
        if hasattr(analyzer.gnn, 'last_training_time') and analyzer.gnn.last_training_time:
            training_time = analyzer.gnn.last_training_time
            if training_time < 60:
                time_str = f"{training_time:.1f}s"
            else:
                time_str = f"{training_time/60:.1f}m"
            st.metric("学習時間", time_str)

def render_dashboard_charts(results, st_session_state):
    """ダッシュボードグラフを表示"""
    st.subheader("📈 分析グラフ")

    # グラフ1: スキル保有率比較
    col1, col2 = st.columns(2)

    with col1:
        st.write("**スキル保有率比較（Top 10）**")
        top_skills = results['skill_importance'][:10]
        skill_names = [s['skill_name'] for s in top_skills]
        excellent_rates = [s['excellent_rate'] * 100 for s in top_skills]
        non_excellent_rates = [s['non_excellent_rate'] * 100 for s in top_skills]

        fig = go.Figure(data=[
            go.Bar(name='優秀群', y=skill_names, x=excellent_rates, orientation='h', marker_color='#FF6B6B'),
            go.Bar(name='非優秀群', y=skill_names, x=non_excellent_rates, orientation='h', marker_color='#4ECDC4')
        ])
        fig.update_layout(barmode='group', height=400, showlegend=True, margin=dict(l=150))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("**社員スコア分布**")
        member_scores = [m['score'] for m in results['member_scores']]

        fig = go.Figure(data=[
            go.Histogram(x=member_scores, nbinsx=HISTOGRAM_BINS, marker_color='#95E1D3')
        ])
        fig.update_layout(
            title="スコア分布",
            xaxis_title="スコア",
            yaxis_title="人数",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # グラフ3: スキル相互作用（存在する場合）
    if st_session_state.interaction_results:
        st.write("**スキル相互作用トップ（相乗効果）**")
        interactions = st_session_state.interaction_results[:5]

        interaction_labels = [f"{i['skill_a_name']}\n×\n{i['skill_b_name']}" for i in interactions]
        synergy_values = [i['synergy'] for i in interactions]

        fig = go.Figure(data=[
            go.Bar(y=interaction_labels, x=synergy_values, orientation='h', marker_color='#FFA07A')
        ])
        fig.update_layout(
            title="スキル相互作用（相乗効果）",
            xaxis_title="相乗効果",
            height=300,
            margin=dict(l=200)
        )
        st.plotly_chart(fig, use_container_width=True)

def render_model_metrics(evaluation_results):
    """モデル評価メトリクスを表示"""
    st.subheader("🎯 モデル性能")

    if evaluation_results is None:
        st.info("モデル評価は実行されていません")
        return

    if evaluation_results.get('method') == 'holdout':
        cols = st.columns(4)
        train_metrics = evaluation_results.get('train_metrics', {})
        test_metrics = evaluation_results.get('test_metrics', {})

        with cols[0]:
            st.metric("Train AUC", f"{train_metrics.get('auc', 0):.3f}")
        with cols[1]:
            st.metric("Test AUC", f"{test_metrics.get('auc', 0):.3f}")
        with cols[2]:
            st.metric("Precision", f"{test_metrics.get('precision', 0):.3f}")
        with cols[3]:
            st.metric("Recall", f"{test_metrics.get('recall', 0):.3f}")

def render_detailed_analysis(results, st_session_state):
    """詳細分析を折りたたみ型で表示"""

    # 詳細スキル一覧
    with st.expander("📋 詳細スキル一覧"):
        skill_df = pd.DataFrame(results['skill_importance'])
        skill_df_display = skill_df.copy()
        skill_df_display['優秀群保有率'] = skill_df_display['excellent_rate'].apply(lambda x: f"{x*100:.1f}%")
        skill_df_display['非優秀群保有率'] = skill_df_display['non_excellent_rate'].apply(lambda x: f"{x*100:.1f}%")
        skill_df_display['重要度スコア'] = skill_df_display['importance_score'].apply(lambda x: f"{x:.3f}")

        display_cols = ['skill_name', '優秀群保有率', '非優秀群保有率', '重要度スコア']
        st.dataframe(skill_df_display[display_cols], use_container_width=True)

    # 社員ランキング
    with st.expander("👥 社員スコアランキング"):
        member_df = pd.DataFrame(results['member_scores'])
        member_df_display = member_df.copy()
        member_df_display['スコア'] = member_df_display['score'].apply(lambda x: f"{x:.1f}")
        member_df_display['優秀群'] = member_df_display['is_excellent'].apply(lambda x: "✓" if x else "")

        display_cols = ['member_name', 'スコア', '優秀群']
        st.dataframe(member_df_display[display_cols], use_container_width=True)

    # モデル性能
    if st_session_state.evaluation_results:
        with st.expander("📈 詳細モデル性能"):
            render_model_metrics(st_session_state.evaluation_results)

    # 因果効果
    if st_session_state.causal_results:
        with st.expander("🔬 因果効果分析"):
            causal_df = pd.DataFrame(st_session_state.causal_results[:20])
            causal_df_display = causal_df.copy()

            if 'causal_effect' in causal_df_display.columns:
                causal_df_display['因果効果'] = causal_df_display['causal_effect'].apply(lambda x: f"{x:.3f}" if x else "N/A")
                causal_df_display['解釈'] = causal_df_display['interpretation']

                display_cols = ['skill_name', '因果効果', '解釈']
                st.dataframe(causal_df_display[display_cols], use_container_width=True)

    # スキル相互作用
    if st_session_state.interaction_results:
        with st.expander("🔗 スキル相互作用詳細"):
            interaction_df = pd.DataFrame(st_session_state.interaction_results[:20])
            interaction_df_display = interaction_df.copy()
            interaction_df_display['相乗効果'] = interaction_df_display['synergy'].apply(lambda x: f"{x:.3f}")
            interaction_df_display['両方の優秀率'] = interaction_df_display['rate_both'].apply(lambda x: f"{x*100:.1f}%")

            display_cols = ['skill_a_name', 'skill_b_name', '相乗効果', '両方の優秀率']
            st.dataframe(interaction_df_display[display_cols], use_container_width=True)

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

        # ダッシュボード表示
        render_skill_cards(results['skill_importance'], top_n=3)
        st.markdown("---")

        render_analysis_metrics(results, analyzer)
        st.markdown("---")

        render_dashboard_charts(results, st.session_state)
        st.markdown("---")

        if st.session_state.evaluation_results is not None:
            render_model_metrics(st.session_state.evaluation_results)
            st.markdown("---")

        render_detailed_analysis(results, st.session_state)

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
