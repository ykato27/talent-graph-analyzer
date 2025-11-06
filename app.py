"""
GNN優秀人材分析システム - Streamlitアプリケーション
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import logging
from gnn_talent_analyzer import (
    TalentAnalyzer,
    DataValidationError,
    DataLoadingError,
    ModelTrainingError,
    ModelEvaluationError,
    CausalInferenceError,
    AnalysisError
)
from config_loader import get_config

# ロギング設定
logger = logging.getLogger('TalentAnalyzer')

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
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'causal_results' not in st.session_state:
    st.session_state.causal_results = None
if 'interaction_results' not in st.session_state:
    st.session_state.interaction_results = None
if 'member_df' not in st.session_state:
    st.session_state.member_df = None
if 'gnn_trained' not in st.session_state:
    st.session_state.gnn_trained = False
# GNN版の結果
if 'skill_profile_gnn' not in st.session_state:
    st.session_state.skill_profile_gnn = None
if 'hte_results_gnn' not in st.session_state:
    st.session_state.hte_results_gnn = None
if 'insights_gnn' not in st.session_state:
    st.session_state.insights_gnn = None
# 従来版の結果
if 'skill_profile_trad' not in st.session_state:
    st.session_state.skill_profile_trad = None
if 'hte_results_trad' not in st.session_state:
    st.session_state.hte_results_trad = None
if 'insights_trad' not in st.session_state:
    st.session_state.insights_trad = None

# タイトル
st.title(f"{get_config('ui.page_icon', '🎯')} {get_config('ui.page_title', 'GNN優秀人材分析システム')}")
st.markdown("---")

# サイドバー: 機能選択メニュー
st.sidebar.title("📋 機能メニュー")
st.sidebar.markdown("分析したい機能を選択してください")

selected_feature = st.sidebar.radio(
    "機能を選択",
    [
        "📁 データ管理",
        "🔬 GNN埋め込み分析（高度）",
        "📊 従来版因果推論（シンプル）"
    ],
    index=0
)

st.sidebar.markdown("---")

# データ読み込み状態の表示
if st.session_state.data_loaded:
    st.sidebar.success("✅ データ読み込み済み")
    analyzer = st.session_state.analyzer
    st.sidebar.metric("総社員数", len(analyzer.members))
    st.sidebar.metric("スキル種類数", len(analyzer.skill_codes))
else:
    st.sidebar.warning("⚠️ データ未読み込み")
    st.sidebar.info("👆 「📁 データ管理」を選択して\nデータをアップロードしてください")

st.sidebar.markdown("---")
st.sidebar.markdown("**📖 機能説明**")

if selected_feature == "📁 データ管理":
    st.sidebar.info(
        "CSVファイルをアップロードし、\n"
        "データを読み込みます"
    )
elif selected_feature == "🔬 GNN埋め込み分析（高度）":
    st.sidebar.info(
        "GNNで高次元の埋め込み表現を学習し、\n"
        "より精度の高い因果推論分析を行います\n\n"
        "📌 GNN学習が必要です"
    )
else:
    st.sidebar.info(
        "GNN学習不要で生データから\n"
        "直接因果推論分析を行います\n\n"
        "📌 シンプルで解釈しやすい分析"
    )

st.sidebar.markdown("---")

# メインコンテンツ: 選択された機能に応じた表示

# ========================================
# 📁 データ管理画面
# ========================================
if selected_feature == "📁 データ管理":
    st.header("📁 データ管理")

    st.markdown("---")
    st.subheader("1️⃣ CSVファイルのアップロード")

    col1, col2 = st.columns([1, 1])

    with col1:
        member_file = st.file_uploader("社員マスタ (member_skillnote.csv)", type=['csv'], key="member_upload")
        acquired_file = st.file_uploader("スキル習得データ (acquiredCompetenceLevel.csv)", type=['csv'], key="acquired_upload")
        skill_file = st.file_uploader("スキルマスタ (skill_skillnote.csv)", type=['csv'], key="skill_upload")

    with col2:
        education_file = st.file_uploader("教育マスタ (education_skillnote.csv)", type=['csv'], key="education_upload")
        license_file = st.file_uploader("資格マスタ (license_skillnote.csv)", type=['csv'], key="license_upload")

    st.markdown("---")
    st.subheader("2️⃣ データ読み込み")

    uploaded_files = {
        'member': member_file,
        'acquired': acquired_file,
        'skill': skill_file,
        'education': education_file,
        'license': license_file
    }

    if st.button("📊 データ読み込み", type="primary", disabled=not all(uploaded_files.values())):
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

                    st.success("✅ データ読み込み完了！")
                    st.balloons()

            except pd.errors.ParserError as e:
                logger.error(f"CSV解析エラー: {e}", exc_info=True)
                st.error(f"❌ CSV形式が無効です。カラム名と型を確認してください。\n詳細: {str(e)}")
            except (DataValidationError, DataLoadingError) as e:
                logger.error(f"データエラー: {e}", exc_info=True)
                st.error(f"❌ データエラー: {str(e)}")
            except Exception as e:
                logger.error(f"予期しないエラー: {e}", exc_info=True)
                st.error(f"❌ 予期しないエラーが発生しました: {str(e)}")
        else:
            st.warning("⚠️ すべてのCSVファイルをアップロードしてください")

    # データ概要表示
    if st.session_state.data_loaded:
        st.markdown("---")
        st.subheader("3️⃣ データ概要")
        analyzer = st.session_state.analyzer

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

# ========================================
# 🔬 GNN埋め込み分析画面
# ========================================
elif selected_feature == "🔬 GNN埋め込み分析（高度）":
    st.header("🔬 GNN埋め込み分析（高度な分析）")

    if not st.session_state.data_loaded:
        st.warning("⚠️ 先に「📁 データ管理」でデータをアップロードしてください")
    else:
        analyzer = st.session_state.analyzer

        st.markdown("---")
        # 1️⃣ 優秀人材選択
        st.subheader("1️⃣ 優秀人材の選択（GNN学習用）")

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

        selection_method = st.radio(
            "選択方法",
            ["手動選択", "スキル保有数上位を自動選択"],
            horizontal=True,
            key="gnn_selection_method"
        )

        if selection_method == "手動選択":
            selected_members = st.multiselect(
                f"優秀な社員を選択してください（5-{MAX_EXCELLENT_RECOMMENDED}名推奨）",
                options=member_df_display['コード'].tolist(),
                format_func=lambda x: f"{member_df_display[member_df_display['コード']==x]['名前'].values[0]} ({x})",
                key="gnn_manual_select"
            )
        else:
            n_top = st.slider(
                "上位何名を選択しますか？",
                min_value=MIN_EXCELLENT,
                max_value=MAX_EXCELLENT_RECOMMENDED,
                value=10,
                key="gnn_auto_select"
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

        # セッションステートに保存（3️⃣で使用）
        st.session_state.selected_members_gnn = selected_members

        st.markdown("---")

        # 2️⃣ GNN学習
        st.subheader("2️⃣ GNN学習")

        col1, col2 = st.columns([1, 3])

        with col1:
            epochs = st.number_input(
                "学習エポック数",
                min_value=MIN_EPOCHS,
                max_value=MAX_EPOCHS,
                value=DEFAULT_EPOCHS,
                step=50,
                help="学習の反復回数。多いほど精度が上がりますが時間がかかります",
                key="gnn_epochs"
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

        if st.button("🚀 GNN学習を開始", type="primary", disabled=(len(selected_members) < MIN_EXCELLENT), key="gnn_train"):
            try:
                with st.spinner("GNNモデルの学習を実行中..."):
                    analyzer.train(selected_members, epochs_unsupervised=epochs)
                    st.session_state.gnn_trained = True

                st.success("✅ GNN学習完了！次に「3️⃣ 逆向き因果推論分析」を実行してください。")
            except ModelTrainingError as e:
                logger.error(f"モデル学習エラー: {e}", exc_info=True)
                st.error(
                    f"❌ モデル学習中にエラーが発生しました。\n"
                    f"詳細: {str(e)}\n\n"
                    f"対策:\n"
                    f"- エポック数を減らしてみてください\n"
                    f"- 優秀人材の人数を増やしてみてください"
                )
            except Exception as e:
                logger.error(f"予期しないエラー: {e}", exc_info=True)
                st.error(f"❌ 予期しないエラーが発生しました: {str(e)}")

        st.markdown("---")

        # 3️⃣ GNN版逆向き因果推論
        st.subheader("3️⃣ GNN埋め込みを使った逆向き因果推論")

        if not st.session_state.get('gnn_trained', False):
            st.warning("⚠️ まず上の「2️⃣ GNN学習」を完了してください")
        elif not st.session_state.get('selected_members_gnn'):
            st.warning("⚠️ まず上の「1️⃣ 優秀人材の選択」で優秀群を選択してください")
        else:
            # 1️⃣で選択された優秀群を使用
            selected_excellent = st.session_state.selected_members_gnn

            st.info(f"📊 1️⃣で選択された優秀群（{len(selected_excellent)}名）を使用して分析を実行します")

            if st.button("🚀 GNN版 Layer 1-3 分析を実行", type="primary", key="gnn_causal_run"):
                try:
                    with st.spinner("GNN版 Layer 1-3 分析を実行中...（GNN埋め込みを活用）"):
                        skill_profile = analyzer.analyze_skill_profile_of_excellent_members(selected_excellent)
                        hte_results = analyzer.estimate_heterogeneous_treatment_effects_with_gnn(selected_excellent, skill_profile)
                        insights = analyzer.generate_comprehensive_insights(selected_excellent, skill_profile, hte_results)

                        st.session_state.skill_profile_gnn = skill_profile
                        st.session_state.hte_results_gnn = hte_results
                        st.session_state.insights_gnn = insights

                        st.success("✅ GNN版 Layer 1-3 分析が完了しました！")

                except (CausalInferenceError, DataValidationError) as e:
                    logger.error(f"因果推論エラー: {e}", exc_info=True)
                    st.error(
                        f"❌ Layer 1-3 分析の実行中にエラーが発生しました。\n"
                        f"詳細: {str(e)}\n\n"
                        f"対策:\n"
                        f"- 優秀人材の人数を増やしてみてください（推奨: 5-10名）\n"
                        f"- 対象社員の総数が十分か確認してください（推奨: 50名以上）"
                    )
                except Exception as e:
                    logger.error(f"GNN分析実行エラー: {e}", exc_info=True)
                    st.error(f"❌ GNN分析中にエラーが発生しました: {str(e)}")

        # GNN版分析結果の表示
        if hasattr(st.session_state, 'insights_gnn') and st.session_state.insights_gnn is not None:
            insights = st.session_state.insights_gnn
            skill_profile = st.session_state.skill_profile_gnn
            hte_results = st.session_state.hte_results_gnn

            st.markdown("---")
            st.markdown(insights['executive_summary'])

            analysis_tabs = st.tabs([
                "🎯 優秀者スキルプロファイル",
                "👥 メンバー別改善提案",
                "🔗 スキル相乗効果"
            ])

            with analysis_tabs[0]:
                st.subheader("優秀者が持つべきスキル TOP 10")
                top_10_skills = skill_profile[:10]

                for idx, skill in enumerate(top_10_skills, 1):
                    with st.expander(f"{idx}. {skill['skill_name']} ({skill['importance']*100:+.1f}% 差分)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("優秀群での習得率", f"{skill['p_excellent']*100:.0f}%",
                                    f"信頼区間: {skill['ci_excellent'][0]*100:.0f}%-{skill['ci_excellent'][1]*100:.0f}%")
                            st.metric("非優秀群での習得率", f"{skill['p_control']*100:.0f}%",
                                    f"信頼区間: {skill['ci_control'][0]*100:.0f}%-{skill['ci_control'][1]*100:.0f}%")
                        with col2:
                            st.metric("重要度（差分）", f"{skill['importance']*100:+.1f}%")
                            st.metric("統計的有意性", "有意" if skill['significant'] else "有意でない",
                                    f"p-value: {skill['p_value']:.4f}")
                        st.info(skill['interpretation'])

            with analysis_tabs[1]:
                st.subheader("メンバー別改善提案（TOP 20）")
                recommendations = insights['member_recommendations'][:20]

                for rec in recommendations:
                    with st.expander(f"{rec['member_id']}: 改善期待値 {rec['estimated_improvement']*100:+.1f}%"):
                        st.write(rec['summary'])
                        for skill in rec['priority_skills']:
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.write(f"**{skill['rank']}. {skill['skill_name']}**")
                                st.caption(skill['reasoning'])
                            with col2:
                                st.metric("信頼度", skill['confidence'], f"{skill['expected_effect']*100:+.1f}%")

            with analysis_tabs[2]:
                st.subheader("スキル相乗効果（因果推論ベース）")
                st.info("優秀群で共起率が高く、非優秀群との差が大きいスキル組み合わせです")
                synergies = insights['skill_combinations']

                if synergies:
                    for idx, s in enumerate(synergies, 1):
                        with st.expander(
                            f"{idx}. {s['skill1']} × {s['skill2']} "
                            f"(相乗効果スコア: {s['synergy_score']:.3f})"
                        ):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    "優秀群での共起率",
                                    f"{s['co_occurrence_excellent']*100:.1f}%",
                                    f"{s['n_excellent_with_both']}名が両方保有"
                                )
                            with col2:
                                st.metric(
                                    "非優秀群での共起率",
                                    f"{s['co_occurrence_non_excellent']*100:.1f}%",
                                    f"{s['n_non_excellent_with_both']}名が両方保有"
                                )

                            st.markdown(f"**統計的有意性:** {'有意 (p < 0.05)' if s['significant'] else '有意でない'} (p = {s['p_value']:.4f})")
                            st.info(s['interpretation'])
                else:
                    st.info("相乗効果が検出されませんでした")

# ========================================
# 📊 従来版因果推論画面
# ========================================
else:  # 従来版因果推論
    st.header("📊 従来版因果推論（シンプル版）")
    st.info("💡 GNN学習なしで、従来型の因果推論のみを使用した分析です。シンプルかつ高速に実行できます。")

    if not st.session_state.data_loaded:
        st.warning("⚠️ 先に「📁 データ管理」でデータをアップロードしてください")
    else:
        analyzer = st.session_state.analyzer

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
            horizontal=True,
            key="trad_selection_method"
        )

        if selection_method == "手動選択":
            # マルチセレクト
            selected_members_trad = st.multiselect(
                f"優秀な社員を選択してください（5-{MAX_EXCELLENT_RECOMMENDED}名推奨）",
                options=member_df_display['コード'].tolist(),
                format_func=lambda x: f"{member_df_display[member_df_display['コード']==x]['名前'].values[0]} ({x})",
                key="trad_members"
            )
        else:
            # 上位N名を自動選択
            n_top = st.slider(
                "上位何名を選択しますか？",
                min_value=MIN_EXCELLENT,
                max_value=MAX_EXCELLENT_RECOMMENDED,
                value=10,
                key="trad_n_top"
            )
            top_members = member_df_display.nlargest(n_top, 'スキル保有数')
            selected_members_trad = top_members['コード'].tolist()

            st.info(f"スキル保有数上位{n_top}名を自動選択しました")
            st.dataframe(top_members, use_container_width=True)

        st.markdown(f"**選択された社員数: {len(selected_members_trad)}名**")

        if len(selected_members_trad) < MIN_EXCELLENT:
            st.warning(f"⚠️ 最低{MIN_EXCELLENT}名以上の優秀人材を選択してください")
        elif len(selected_members_trad) > MAX_EXCELLENT_RECOMMENDED:
            st.warning(f"⚠️ {MAX_EXCELLENT_RECOMMENDED}名以下での選択を推奨します")

        st.markdown("---")

        # 分析実行
        st.header("2️⃣ 従来版因果推論分析")
        st.info("📊 GNN学習なしで、Layer 1-3 逆向き因果推論分析を実行します")

        if st.button("🚀 分析開始", type="primary", disabled=(len(selected_members_trad) < MIN_EXCELLENT), key="trad_run"):
            try:
                with st.spinner("従来版因果推論分析を実行中..."):
                    # Layer 1: スキルプロファイル分析
                    skill_profile_trad = analyzer.analyze_skill_profile_of_excellent_members(selected_members_trad)

                    # Layer 2: 異質的処置効果推定（従来版）
                    hte_results_trad = analyzer.estimate_heterogeneous_treatment_effects(
                        selected_members_trad,
                        skill_profile_trad
                    )

                    # Layer 3: 総合的な洞察生成（相乗効果分析を含む）
                    insights_trad = analyzer.generate_comprehensive_insights(
                        selected_members_trad,
                        skill_profile_trad,
                        hte_results_trad
                    )

                    # セッションステートに保存
                    st.session_state.skill_profile_trad = skill_profile_trad
                    st.session_state.hte_results_trad = hte_results_trad
                    st.session_state.insights_trad = insights_trad
                    st.session_state.selected_members_trad = selected_members_trad

                st.success("✅ 従来版因果推論分析が完了しました！")

            except Exception as e:
                logger.error(f"従来版分析中にエラーが発生しました: {e}", exc_info=True)
                st.error(
                    f"❌ 分析中にエラーが発生しました。\n"
                    f"詳細: {str(e)}\n\n"
                    f"対策:\n"
                    f"- 優秀人材の人数を増やしてみてください\n"
                    f"- データの品質を確認してください"
                )
                import traceback
                st.error(traceback.format_exc())

        st.markdown("---")

        # 結果表示
        if 'insights_trad' in st.session_state:
            st.header("📈 分析結果")

            insights_trad = st.session_state.insights_trad
            skill_profile_trad = st.session_state.skill_profile_trad

            tab1, tab2, tab3 = st.tabs([
                "🎯 スキルプロファイル",
                "👥 メンバー別推奨",
                "🔗 スキル相乗効果"
            ])

            with tab1:
                st.subheader("優秀人材の特徴的スキル（上位10件）")
                st.info("優秀群で有意に高い習得率を示すスキルを重要度順に表示しています")

                top_skills = skill_profile_trad[:10]

                if len(top_skills) > 0:
                    df_skills = pd.DataFrame([
                        {
                            'スキル': s['skill_name'],
                            '重要度': f"{s['importance']:.3f}",
                            '優秀群習得率': f"{s['p_excellent']*100:.1f}%",
                            '非優秀群習得率': f"{s['p_control']*100:.1f}%",
                            'p値': f"{s['p_value']:.4f}",
                            '統計的有意性': '有意' if s['significant'] else '有意でない'
                        }
                        for s in top_skills
                    ])
                    st.dataframe(df_skills, use_container_width=True)
                else:
                    st.warning("有意なスキルが検出されませんでした")

            with tab2:
                st.subheader("メンバー別スキル推奨（上位20名）")
                st.info("各メンバーに最も効果的なスキル習得を推奨しています")

                recommendations_trad = insights_trad['member_recommendations'][:20]

                if len(recommendations_trad) > 0:
                    for i, rec in enumerate(recommendations_trad, 1):
                        member_name = analyzer.member_names.get(rec['member_id'], '不明')
                        with st.expander(f"{i}. {member_name} ({rec['member_id']}) - 推奨スキル: {rec['recommended_skill']}"):
                            st.markdown(f"**推奨スキル:** {rec['recommended_skill']}")
                            st.markdown(f"**期待効果:** {rec['expected_effect']:.3f}")
                            st.markdown(f"**信頼度:** {rec['confidence']}")
                            st.markdown(f"**理由:**\n{rec['reasoning']}")
                else:
                    st.warning("推奨が生成されませんでした")

            with tab3:
                st.subheader("スキル相乗効果（因果推論ベース）")
                st.info("優秀群で共起率が高く、非優秀群との差が大きいスキル組み合わせです")

                synergies_trad = insights_trad['skill_combinations']

                if len(synergies_trad) > 0:
                    for idx, s in enumerate(synergies_trad, 1):
                        with st.expander(
                            f"{idx}. {s['skill1']} × {s['skill2']} "
                            f"(相乗効果スコア: {s['synergy_score']:.3f})"
                        ):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    "優秀群での共起率",
                                    f"{s['co_occurrence_excellent']*100:.1f}%",
                                    f"{s['n_excellent_with_both']}名が両方保有"
                                )
                            with col2:
                                st.metric(
                                    "非優秀群での共起率",
                                    f"{s['co_occurrence_non_excellent']*100:.1f}%",
                                    f"{s['n_non_excellent_with_both']}名が両方保有"
                                )

                            st.markdown(f"**統計的有意性:** {'有意 (p < 0.05)' if s['significant'] else '有意でない'} (p = {s['p_value']:.4f})")
                            st.info(s['interpretation'])
                else:
                    st.info("相乗効果が検出されませんでした")

# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
GNN優秀人材分析システム v2.0 | 逆向き因果推論 + HTE分析対応 | Powered by Graph Neural Networks
</div>
""", unsafe_allow_html=True)
