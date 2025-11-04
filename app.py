"""
GNN優秀人材分析システム - Streamlitアプリケーション
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from gnn_talent_analyzer import TalentAnalyzer, load_csv_files
import io

# ページ設定
st.set_page_config(
    page_title="GNN優秀人材分析システム",
    page_icon="🎯",
    layout="wide"
)

# セッション状態の初期化
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'member_df' not in st.session_state:
    st.session_state.member_df = None

# タイトル
st.title("🎯 GNN優秀人材分析システム")
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
                member_df = pd.read_csv(uploaded_files['member'], encoding='utf-8-sig')
                acquired_df = pd.read_csv(uploaded_files['acquired'], encoding='utf-8-sig')
                skill_df = pd.read_csv(uploaded_files['skill'], encoding='utf-8-sig')
                education_df = pd.read_csv(uploaded_files['education'], encoding='utf-8-sig')
                license_df = pd.read_csv(uploaded_files['license'], encoding='utf-8-sig')
                
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
            "優秀な社員を選択してください（5-10名推奨）",
            options=member_df_display['コード'].tolist(),
            format_func=lambda x: f"{member_df_display[member_df_display['コード']==x]['名前'].values[0]} ({x})"
        )
    else:
        # 上位N名を自動選択
        n_top = st.slider("上位何名を選択しますか？", min_value=3, max_value=20, value=10)
        top_members = member_df_display.nlargest(n_top, 'スキル保有数')
        selected_members = top_members['コード'].tolist()
        
        st.info(f"スキル保有数上位{n_top}名を自動選択しました")
        st.dataframe(top_members, use_container_width=True)
    
    st.markdown(f"**選択された社員数: {len(selected_members)}名**")
    
    if len(selected_members) < 3:
        st.warning("⚠️ 最低3名以上の優秀人材を選択してください")
    elif len(selected_members) > 20:
        st.warning("⚠️ 20名以下での選択を推奨します")
    
    st.markdown("---")
    
    # 分析実行
    st.header("2️⃣ 分析実行")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        epochs = st.number_input(
            "学習エポック数",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="学習の反復回数。多いほど精度が上がりますが時間がかかります"
        )
    
    with col2:
        st.info("""
        **推奨設定**
        - 優秀群5名以下: 50-100エポック
        - 優秀群10名程度: 100-200エポック
        - 優秀群20名以上: 200-300エポック
        """)
    
    if st.button("🚀 分析開始", type="primary", disabled=(len(selected_members) < 3)):
        try:
            with st.spinner("GNNモデルの学習と分析を実行中..."):
                # 学習
                analyzer.train(selected_members, epochs_unsupervised=epochs)
                
                # 分析
                results = analyzer.analyze(selected_members)
                st.session_state.results = results
                
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
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 重要スキルランキング",
            "👥 社員スコアランキング",
            "📊 スキル比較分析",
            "🗺️ 埋め込み可視化"
        ])
        
        with tab1:
            st.subheader("優秀群に特徴的なスキル Top20")
            
            skill_df = pd.DataFrame(results['skill_importance'][:20])
            
            # 表示用にフォーマット
            skill_df_display = skill_df.copy()
            skill_df_display['優秀群保有率'] = skill_df_display['excellent_rate'].apply(lambda x: f"{x*100:.1f}%")
            skill_df_display['非優秀群保有率'] = skill_df_display['non_excellent_rate'].apply(lambda x: f"{x*100:.1f}%")
            skill_df_display['差分'] = skill_df_display['rate_diff'].apply(lambda x: f"+{x*100:.1f}%" if x > 0 else f"{x*100:.1f}%")
            skill_df_display['重要度'] = skill_df_display['importance_score'].apply(lambda x: f"{x:.3f}")
            
            st.dataframe(
                skill_df_display[['skill_name', '優秀群保有率', '非優秀群保有率', '差分', '重要度']],
                use_container_width=True
            )
            
            # 棒グラフ
            fig = go.Figure()
            
            top_n = 15
            top_skills = results['skill_importance'][:top_n]
            
            fig.add_trace(go.Bar(
                x=[s['excellent_rate']*100 for s in top_skills],
                y=[s['skill_name'] for s in top_skills],
                orientation='h',
                name='優秀群',
                marker_color='#FF6B6B'
            ))
            
            fig.add_trace(go.Bar(
                x=[s['non_excellent_rate']*100 for s in top_skills],
                y=[s['skill_name'] for s in top_skills],
                orientation='h',
                name='非優秀群',
                marker_color='#4ECDC4'
            ))
            
            fig.update_layout(
                title=f"スキル保有率比較 Top{top_n}",
                xaxis_title="保有率 (%)",
                yaxis_title="スキル名",
                barmode='group',
                height=600,
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
                height=400
            )
            
            # 分布のヒストグラム
            fig = go.Figure()
            
            excellent_scores = member_scores_df[member_scores_df['is_excellent']]['score']
            non_excellent_scores = member_scores_df[~member_scores_df['is_excellent']]['score']
            
            fig.add_trace(go.Histogram(
                x=excellent_scores,
                name='優秀群',
                opacity=0.7,
                marker_color='#FF6B6B',
                nbinsx=20
            ))
            
            fig.add_trace(go.Histogram(
                x=non_excellent_scores,
                name='非優秀群',
                opacity=0.7,
                marker_color='#4ECDC4',
                nbinsx=20
            ))
            
            fig.update_layout(
                title="優秀度スコアの分布",
                xaxis_title="優秀度スコア",
                yaxis_title="人数",
                barmode='overlay',
                height=400
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
            significant_skills = [s for s in results['skill_importance'] if abs(s['rate_diff']) > 0.2][:30]
            
            if len(significant_skills) > 0:
                # スキルカテゴリごとの分析（簡易版）
                st.markdown("### 保有率差が大きいスキル（差分20%以上）")
                
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
                color_discrete_map={True: '#FF6B6B', False: '#4ECDC4'}
            )
            
            fig.update_traces(marker=dict(size=10))
            fig.update_layout(height=600)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **解釈のポイント**
            - 赤い点が優秀群、青い点が非優秀群
            - 近い位置にある社員は似たスキルプロファイルを持つ
            - 優秀群が集まっている領域が「優秀な人材の特徴空間」
            """)
        
        # 推奨育成プラン
        st.markdown("---")
        st.header("4️⃣ 推奨育成プラン")
        
        # 必須スキル（優秀群の80%以上が保有）
        essential_skills = [s for s in results['skill_importance'] if s['excellent_rate'] >= 0.8]
        
        # 重要スキル（保有率差が大きい）
        important_skills = [s for s in results['skill_importance'] if s['rate_diff'] >= 0.3][:10]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 必須スキル")
            st.markdown("優秀群の80%以上が保有しているスキル")
            
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
            csv_skills = skill_export.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="📥 重要スキル一覧をダウンロード",
                data=csv_skills,
                file_name="skill_importance.csv",
                mime="text/csv"
            )
        
        with col2:
            # 社員スコアのCSV
            member_export = pd.DataFrame(results['member_scores'])
            csv_members = member_export.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="📥 社員スコア一覧をダウンロード",
                data=csv_members,
                file_name="member_scores.csv",
                mime="text/csv"
            )

else:
    # データ未読み込み時の表示
    st.info("👈 左のサイドバーからCSVファイルをアップロードしてください")
    
    st.markdown("""
    ### 📝 使い方
    
    1. **データアップロード**
       - 5つのCSVファイルをアップロード
       - データ読み込みボタンをクリック
    
    2. **優秀人材の選択**
       - 優秀と考える社員を5-10名程度選択
       - または上位N名を自動選択
    
    3. **分析実行**
       - 学習エポック数を設定（推奨: 100）
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
    
    - 優秀群: 5-10名（最低3名）
    - 対象社員: 50名以上推奨
    - 学習エポック: 100-200
    """)

# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
GNN優秀人材分析システム v1.0 | Powered by Graph Neural Networks
</div>
""", unsafe_allow_html=True)
