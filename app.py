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
        st.info(
            "### 🚧 この画面は次のバージョンで実装予定です\n\n"
            "**実装予定の機能：**\n"
            "1. 優秀人材の選択\n"
            "2. GNNモデルの学習\n"
            "3. GNN埋め込みを使った逆向き因果推論\n"
            "4. 詳細な分析結果の表示"
        )

# ========================================
# 📊 従来版因果推論画面
# ========================================
else:  # 従来版因果推論
    st.header("📊 従来版因果推論（シンプル版）")

    if not st.session_state.data_loaded:
        st.warning("⚠️ 先に「📁 データ管理」でデータをアップロードしてください")
    else:
        st.info(
            "### 🚧 この画面は次のバージョンで実装予定です\n\n"
            "**実装予定の機能：**\n"
            "1. 優秀群の選択\n"
            "2. 従来版逆向き因果推論（GNN不要）\n"
            "3. 詳細な分析結果の表示"
        )

# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
GNN優秀人材分析システム v2.0 | 逆向き因果推論 + HTE分析対応 | Powered by Graph Neural Networks
</div>
""", unsafe_allow_html=True)
