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

# タイトル
st.title(f"{get_config('ui.page_icon', '🎯')} {get_config('ui.page_title', 'GNN優秀人材分析システム')}")
st.markdown("---")

# サイドバー
st.sidebar.header("📁 データアップロード")

# ファイルアップロード
uploaded_files = {
    'member': st.sidebar.file_uploader("社員マスタ (member_skillnote.csv)", type=['csv']),
    'acquired': st.sidebar.file_uploader("スキル習得データ (acquiredCompetenceLevel.csv)", type=['csv']),
    'skill': st.sidebar.file_uploader("スキルマスタ (skill_skillnote.csv)", type=['csv']),
    'education': st.sidebar.file_uploader("教育マスタ (education_skillnote.csv)", type=['csv']),
    'license': st.sidebar.file_uploader("資格マスタ (license_skillnote.csv)", type=['csv'])
}

# データ読み込みボタン
if st.sidebar.button("📊 データ読み込み"):
    if all(uploaded_files.values()):
        try:
            with st.spinner("データ読み込み中..."):
                # CSVファイルを読み込み
                try:
                    member_df = pd.read_csv(uploaded_files['member'], encoding=FILE_ENCODING)
                    acquired_df = pd.read_csv(uploaded_files['acquired'], encoding=FILE_ENCODING)
                    skill_df = pd.read_csv(uploaded_files['skill'], encoding=FILE_ENCODING)
                    education_df = pd.read_csv(uploaded_files['education'], encoding=FILE_ENCODING)
                    license_df = pd.read_csv(uploaded_files['license'], encoding=FILE_ENCODING)
                except pd.errors.ParserError as e:
                    logger.error(f"CSV解析エラー: {e}", exc_info=True)
                    st.sidebar.error(
                        f"❌ CSV形式が無効です。カラム名と型を確認してください。\n"
                        f"詳細: {str(e)}"
                    )
                    raise DataLoadingError(f"CSV解析失敗: {e}") from e
                except FileNotFoundError as e:
                    logger.error(f"ファイルが見つかりません: {e}", exc_info=True)
                    st.sidebar.error(f"❌ ファイルが見つかりません: {str(e)}")
                    raise DataLoadingError(f"ファイルが見つかりません: {e}") from e

                # アナライザーの初期化
                analyzer = TalentAnalyzer()
                analyzer.load_data(member_df, acquired_df, skill_df, education_df, license_df)

                st.session_state.analyzer = analyzer
                st.session_state.member_df = member_df
                st.session_state.data_loaded = True

                st.sidebar.success("✅ データ読み込み完了")
        except DataValidationError as e:
            logger.error(f"データ検証エラー: {e}", exc_info=True)
            st.sidebar.error(f"❌ データ検証エラー: {str(e)}")
        except DataLoadingError as e:
            logger.error(f"データ読み込みエラー: {e}", exc_info=True)
            st.sidebar.error(f"❌ データ読み込みエラー: {str(e)}")
        except Exception as e:
            logger.error(f"予期しないエラーが発生しました: {e}", exc_info=True)
            st.sidebar.error(
                f"❌ 予期しないエラーが発生しました。\n"
                f"ファイル形式とデータ内容を確認してください。\n"
                f"詳細: {str(e)}"
            )
    else:
        st.sidebar.warning("⚠️ すべてのCSVファイルをアップロードしてください")

st.sidebar.markdown("---")

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
            with st.spinner("GNNモデルの学習と分析を実行中..."):
                # Layer 1-3 逆向き因果推論分析を実行
                analyzer.train(selected_members, epochs_unsupervised=epochs)

                # Layer 1: 優秀者スキルプロファイル分析
                skill_profile = analyzer.analyze_skill_profile_of_excellent_members(selected_members)

                # Layer 2: 個別メンバーの異質的処置効果推定
                hte_results = analyzer.estimate_heterogeneous_treatment_effects(selected_members, skill_profile)

                # Layer 3: 経営的インサイト生成
                insights = analyzer.generate_comprehensive_insights(selected_members, skill_profile, hte_results)

                # セッション状態に保存
                st.session_state.skill_profile = skill_profile
                st.session_state.hte_results = hte_results
                st.session_state.insights = insights

            st.success("✅ Layer 1-3 分析完了！")
        except ModelTrainingError as e:
            logger.error(f"モデル学習エラー: {e}", exc_info=True)
            st.error(
                f"❌ モデル学習中にエラーが発生しました。\n"
                f"詳細: {str(e)}\n\n"
                f"対策:\n"
                f"- エポック数を減らしてみてください\n"
                f"- 優秀人材の人数を増やしてみてください"
            )
        except (CausalInferenceError, DataValidationError) as e:
            logger.error(f"分析エラー: {e}", exc_info=True)
            st.error(
                f"❌ Layer 1-3 分析の実行中にエラーが発生しました。\n"
                f"詳細: {str(e)}\n\n"
                f"対策:\n"
                f"- 優秀人材の人数を増やしてみてください（推奨: 5-10名）\n"
                f"- 対象社員の総数が十分か確認してください（推奨: 50名以上）"
            )
        except AnalysisError as e:
            logger.error(f"分析エラー: {e}", exc_info=True)
            st.error(
                f"❌ 分析中にエラーが発生しました。\n"
                f"詳細: {str(e)}"
            )
        except Exception as e:
            logger.error(f"予期しないエラーが発生しました: {e}", exc_info=True)
            st.error(
                f"❌ 予期しないエラーが発生しました。\n"
                f"詳細: {str(e)}\n\n"
                f"ログ出力:"
            )
            import traceback
            st.error(traceback.format_exc())

    st.markdown("---")

    # Layer 1-3 逆向き因果推論分析の結果表示
    st.markdown("---")
    st.header("🔄 逆向き因果推論分析（新機能）")

    if st.session_state.data_loaded and st.session_state.analyzer is not None:
        with st.expander("📚 Layer 1-3 分析を実行", expanded=True):

            # 優秀群の選択
            selected_excellent = st.multiselect(
                "優秀群として分析する社員を選択（最低3名）",
                st.session_state.member_df['メンバーコード'].unique(),
                help="統計的に有意な結果を得るため、5-10名の選択を推奨"
            )

            if len(selected_excellent) >= 3 and st.button("🚀 Layer 1-3 分析を実行"):
                try:
                    with st.spinner("Layer 1-3 分析を実行中...（数秒かかります）"):

                        # Layer 1: 優秀者特性の逆向き分析
                        logger.info(f"Layer 1を実行中: {len(selected_excellent)}人の優秀群を分析")
                        skill_profile = st.session_state.analyzer.analyze_skill_profile_of_excellent_members(
                            selected_excellent
                        )

                        # Layer 2: 個別メンバーへの因果効果推定
                        logger.info("Layer 2を実行中: 個別メンバーの因果効果を推定")
                        hte_results = st.session_state.analyzer.estimate_heterogeneous_treatment_effects(
                            selected_excellent,
                            skill_profile
                        )

                        # Layer 3: 説明可能性の強化
                        logger.info("Layer 3を実行中: 包括的な分析洞察を生成")
                        insights = st.session_state.analyzer.generate_comprehensive_insights(
                            selected_excellent,
                            skill_profile,
                            hte_results
                        )

                        # セッション状態に保存
                        st.session_state.skill_profile = skill_profile
                        st.session_state.hte_results = hte_results
                        st.session_state.insights = insights

                        st.success("✅ Layer 1-3 分析が完了しました！")

                except Exception as e:
                    logger.error(f"分析実行エラー: {e}", exc_info=True)
                    st.error(f"❌ 分析中にエラーが発生しました: {str(e)}")

    # 分析結果の表示
    if hasattr(st.session_state, 'insights') and st.session_state.insights is not None:
        insights = st.session_state.insights
        skill_profile = st.session_state.skill_profile
        hte_results = st.session_state.hte_results

        st.markdown("---")

        # Layer 3の結果を表示
        st.markdown(insights['executive_summary'])

        # タブで結果を分割表示
        analysis_tabs = st.tabs([
            "🎯 優秀者スキルプロファイル",
            "👥 メンバー別改善提案",
            "📊 組織スキルギャップ",
            "🔗 スキル相乗効果",
            "🗺️ 開発ロードマップ"
        ])

        # Tab 1: スキルプロファイル（Layer 1）
        with analysis_tabs[0]:
            st.subheader("優秀者が持つべきスキル TOP 10")

            top_10_skills = skill_profile[:10]

            for idx, skill in enumerate(top_10_skills, 1):
                with st.expander(
                    f"{idx}. {skill['skill_name']} "
                    f"({skill['importance']*100:+.1f}% 差分)"
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "優秀群での習得率",
                            f"{skill['p_excellent']*100:.0f}%",
                            f"信頼区間: {skill['ci_excellent'][0]*100:.0f}%-{skill['ci_excellent'][1]*100:.0f}%"
                        )
                        st.metric(
                            "非優秀群での習得率",
                            f"{skill['p_control']*100:.0f}%",
                            f"信頼区間: {skill['ci_control'][0]*100:.0f}%-{skill['ci_control'][1]*100:.0f}%"
                        )

                    with col2:
                        st.metric(
                            "重要度（差分）",
                            f"{skill['importance']*100:+.1f}%"
                        )
                        st.metric(
                            "統計的有意性",
                            "有意" if skill['significant'] else "有意でない",
                            f"p-value: {skill['p_value']:.4f}"
                        )

                    st.info(skill['interpretation'])

        # Tab 2: メンバー別改善提案（Layer 2）
        with analysis_tabs[1]:
            st.subheader("メンバー別改善提案（TOP 20）")

            recommendations = insights['member_recommendations'][:20]

            for rec in recommendations:
                with st.expander(
                    f"{rec['member_id']}: "
                    f"改善期待値 {rec['estimated_improvement']*100:+.1f}%"
                ):
                    st.write(rec['summary'])

                    for skill in rec['priority_skills']:
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.write(f"**{skill['rank']}. {skill['skill_name']}**")
                            st.caption(skill['reasoning'])

                        with col2:
                            st.metric(
                                "信頼度",
                                skill['confidence'],
                                f"{skill['expected_effect']*100:+.1f}%"
                            )

        # Tab 3: 組織スキルギャップ
        with analysis_tabs[2]:
            st.subheader("組織スキルギャップ分析")

            gaps = insights['organizational_gaps']

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("### 🔴 Critical Gap")
                st.write(f"**{len(gaps['critical_gaps'])}個のスキル**")
                for skill in gaps['critical_gaps'][:3]:
                    st.write(f"- {skill['skill_name']}: {skill['gap']*100:+.1f}%")

            with col2:
                st.write("### 🟡 High Potential")
                st.write(f"**{len(gaps['high_potential_skills'])}個のスキル**")
                for skill in gaps['high_potential_skills'][:3]:
                    st.write(f"- {skill['skill_name']}: {skill['importance']*100:+.1f}%")

            with col3:
                st.write("### 🟢 Saturation")
                st.write(f"**{len(gaps['saturation_skills'])}個のスキル**")
                for skill in gaps['saturation_skills'][:3]:
                    st.write(f"- {skill['skill_name']}: {skill['adoption_rate']*100:.0f}%")

        # Tab 4: スキル相乗効果
        with analysis_tabs[3]:
            st.subheader("スキル相乗効果の可能性")

            synergies = insights['skill_combinations']

            if synergies:
                df_synergies = pd.DataFrame([
                    {
                        'スキル組み合わせ': s['skill_combination'],
                        'そのスキル組を習得者': s['member_count_with_both'],
                        'ステータス': s['status']
                    }
                    for s in synergies
                ])

                st.dataframe(df_synergies, use_container_width=True)
            else:
                st.info("相乗効果が検出されませんでした")

        # Tab 5: 開発ロードマップ
        with analysis_tabs[4]:
            st.subheader("スキル開発ロードマップ")

            roadmap = insights['development_roadmap']
            resources = roadmap['resources_required']

            # リソース見積もり
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "開発対象メンバー数",
                    resources['estimated_members_to_develop']
                )

            with col2:
                st.metric(
                    "推奨実施期間",
                    f"{resources['recommended_timeline_months']}ヶ月"
                )

            with col3:
                st.metric(
                    "推定総コスト",
                    f"¥{resources['total_estimated_cost']:,.0f}"
                )

            # 優先度付けスキルプラン
            st.markdown("### 優先度付けスキル習得計画")

            for phase, phase_name in [
                ('immediate_priority', '🔴 即座実施（1ヶ月以内）'),
                ('short_term', '🟡 短期計画（3ヶ月以内）'),
                ('medium_term', '🟢 中期計画（6ヶ月以内）')
            ]:
                with st.expander(phase_name, expanded=(phase == 'immediate_priority')):

                    plans = roadmap[phase][:10]

                    if plans:
                        df_plans = pd.DataFrame([
                            {
                                'メンバーID': p['member_id'],
                                'スキル': p['skill'],
                                '期待効果': f"{p['expected_effect']*100:+.1f}%",
                                '信頼度': p['confidence']
                            }
                            for p in plans
                        ])

                        st.dataframe(df_plans, use_container_width=True)
                    else:
                        st.info(f"{phase_name} に該当するスキルはありません")


# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
GNN優秀人材分析システム v2.0 | 逆向き因果推論 + HTE分析対応 | Powered by Graph Neural Networks
</div>
""", unsafe_allow_html=True)
