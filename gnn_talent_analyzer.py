"""
GNNベース優秀人材分析システム
半教師あり学習 + Few-shot学習対応
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import fisher_exact, ttest_ind
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import warnings
import logging
import os
import pickle
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from config_loader import get_config
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ロギング設定
def setup_logging():
    """ロギングの設定"""
    log_config = get_config('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # ロガーの設定
    logger = logging.getLogger('TalentAnalyzer')
    logger.setLevel(log_level)

    # コンソールハンドラ
    if log_config.get('console_logging', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)

    # ファイルハンドラ
    if log_config.get('file_logging', True):
        log_dir = Path(get_config('versioning.log_dir', './logs'))
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / log_config.get('log_file', 'talent_analyzer.log')

        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=log_config.get('max_bytes', 10485760),
            backupCount=log_config.get('backup_count', 5)
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    return logger

logger = setup_logging()


class SimpleGNN:
    """
    軽量GNN実装（CPU最適化版）
    Graph Convolutional NetworkとGraphSAGEのハイブリッド
    """

    def __init__(self, n_layers=None, hidden_dim=None, dropout=None, learning_rate=None):
        # 設定ファイルから読み込む、引数があればそれを優先
        self.n_layers = n_layers or get_config('model.n_layers', 3)
        self.hidden_dim = hidden_dim or get_config('model.hidden_dim', 128)
        self.dropout = dropout or get_config('model.dropout', 0.3)
        self.learning_rate = learning_rate or get_config('model.learning_rate', 0.01)
        self.k_neighbors = get_config('model.k_neighbors', 10)

        self.weights = []
        self.trained = False
        self.last_training_time = None  # 最後の学習時間を保存

    def build_graph(self, member_features, skill_matrix, member_attrs):
        """
        異種グラフの構築

        Parameters:
        -----------
        member_features: array-like, shape (n_members, n_features)
            社員の基本特徴量
        skill_matrix: array-like, shape (n_members, n_skills)
            スキル保有マトリクス
        member_attrs: dict
            社員の属性情報（等級、役職など）

        Returns:
        --------
        adjacency_matrix: sparse matrix
            グラフの隣接行列
        node_features: array
            ノードの特徴量
        """
        n_members = skill_matrix.shape[0]

        # 社員間の類似度行列を計算（スキルの類似性）
        member_similarity = np.dot(skill_matrix, skill_matrix.T)
        member_similarity = member_similarity / (
            np.linalg.norm(skill_matrix, axis=1, keepdims=True) @
            np.linalg.norm(skill_matrix, axis=1, keepdims=True).T + 1e-8
        )

        # スパース化（上位k個の接続のみ保持）
        k_neighbors = min(self.k_neighbors, n_members - 1)
        for i in range(n_members):
            threshold = np.partition(member_similarity[i], -k_neighbors)[-k_neighbors]
            member_similarity[i][member_similarity[i] < threshold] = 0

        self.adjacency = member_similarity
        self.skill_matrix = skill_matrix
        self.member_features = member_features

        return member_similarity

    def aggregate_neighbors(self, features, adjacency, layer_idx):
        """
        近傍ノードの特徴量を集約（GraphSAGE風）
        """
        # 次数で正規化
        degree = np.sum(adjacency, axis=1, keepdims=True) + 1e-8
        normalized_adj = adjacency / degree

        # 近傍集約
        aggregated = np.dot(normalized_adj, features)

        # 自身の特徴量と結合
        combined = np.concatenate([features, aggregated], axis=1)

        # 線形変換
        if layer_idx < len(self.weights):
            transformed = np.dot(combined, self.weights[layer_idx])
        else:
            # 重みを初期化
            input_dim = combined.shape[1]
            output_dim = self.hidden_dim
            weight = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
            self.weights.append(weight)
            transformed = np.dot(combined, weight)

        # 活性化関数（ReLU）
        activated = np.maximum(0, transformed)

        # Dropout（学習時のみ）
        if self.training:
            mask = np.random.binomial(1, 1 - self.dropout, activated.shape)
            activated = activated * mask / (1 - self.dropout)

        return activated

    def forward(self, adjacency, features):
        """
        順伝播
        """
        h = features

        for layer_idx in range(self.n_layers):
            h = self.aggregate_neighbors(h, adjacency, layer_idx)

        return h

    def fit_unsupervised(self, adjacency, features, epochs=None, on_epoch_callback=None):
        """
        半教師あり学習（ラベルなしで学習）
        Deep Graph Infomax的なアプローチ

        Parameters:
        -----------
        adjacency: array-like
            隣接行列
        features: array-like
            ノードの特徴量
        epochs: int, optional
            学習エポック数
        on_epoch_callback: callable, optional
            各エポック終了時に呼び出されるコールバック関数
            epoch_info辞書を受け取る
        """
        if epochs is None:
            epochs = get_config('training.default_epochs', 100)

        print("半教師あり事前学習を開始...")
        self.training = True
        n_nodes = features.shape[0]

        # 設定から値を取得
        patience = get_config('training.early_stopping_patience', 20)
        batch_size_edges = get_config('training.batch_size_edges', 1000)

        # 重みの初期化
        self.weights = []
        current_dim = features.shape[1] * 2  # 自己と近傍の結合

        for layer_idx in range(self.n_layers):
            output_dim = self.hidden_dim
            weight = np.random.randn(current_dim, output_dim) * np.sqrt(2.0 / current_dim)
            self.weights.append(weight)
            current_dim = self.hidden_dim * 2

        best_loss = float('inf')
        patience_counter = 0

        # 学習時間の計測開始
        training_start_time = time.time()
        epoch_times = []

        # プログレスバーを使用した学習ループ
        pbar = tqdm(range(epochs), desc="GNN学習中", unit="epoch", ncols=100)

        for epoch in pbar:
            epoch_start_time = time.time()

            # 順伝播
            embeddings = self.forward(adjacency, features)

            # エッジ予測による自己教師あり損失
            pos_edges = np.where(adjacency > 0)
            n_pos = min(batch_size_edges, len(pos_edges[0]))
            pos_samples = np.random.choice(len(pos_edges[0]), n_pos, replace=False)

            pos_u = embeddings[pos_edges[0][pos_samples]]
            pos_v = embeddings[pos_edges[1][pos_samples]]
            pos_scores = np.sum(pos_u * pos_v, axis=1)

            # 負例のサンプリング
            neg_u_idx = np.random.randint(0, n_nodes, n_pos)
            neg_v_idx = np.random.randint(0, n_nodes, n_pos)
            neg_u = embeddings[neg_u_idx]
            neg_v = embeddings[neg_v_idx]
            neg_scores = np.sum(neg_u * neg_v, axis=1)

            # 対照学習損失
            loss = -np.mean(np.log(1 / (1 + np.exp(-pos_scores)) + 1e-8)) - \
                   np.mean(np.log(1 - 1 / (1 + np.exp(-neg_scores)) + 1e-8))

            # 重み更新（簡易的な勾配降下）
            for i, weight in enumerate(self.weights):
                grad = np.random.randn(*weight.shape) * 0.01
                new_weight = weight - self.learning_rate * grad

                # 重みを更新して損失を確認
                old_weight = self.weights[i].copy()
                self.weights[i] = new_weight
                new_embeddings = self.forward(adjacency, features)

                # 新しい損失を計算
                pos_u_new = new_embeddings[pos_edges[0][pos_samples]]
                pos_v_new = new_embeddings[pos_edges[1][pos_samples]]
                pos_scores_new = np.sum(pos_u_new * pos_v_new, axis=1)

                neg_u_new = new_embeddings[neg_u_idx]
                neg_v_new = new_embeddings[neg_v_idx]
                neg_scores_new = np.sum(neg_u_new * neg_v_new, axis=1)

                new_loss = -np.mean(np.log(1 / (1 + np.exp(-pos_scores_new)) + 1e-8)) - \
                           np.mean(np.log(1 - 1 / (1 + np.exp(-neg_scores_new)) + 1e-8))

                # 損失が改善しなければ元に戻す
                if new_loss >= loss:
                    self.weights[i] = old_weight

            # エポック終了時刻を記録
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_time)

            # 経過時間と推定完了時間を計算
            elapsed_time = epoch_end_time - training_start_time
            if epoch > 0:
                avg_epoch_time = np.mean(epoch_times)
                remaining_epochs = epochs - epoch - 1
                estimated_remaining_time = avg_epoch_time * remaining_epochs
                estimated_total_time = elapsed_time + estimated_remaining_time

                # プログレスバーにメタ情報を追加
                pbar.set_postfix({
                    'Loss': f'{loss:.4f}',
                    '経過': self._format_time(elapsed_time),
                    '推定残り': self._format_time(estimated_remaining_time),
                    '推定合計': self._format_time(estimated_total_time)
                })
            else:
                pbar.set_postfix({'Loss': f'{loss:.4f}'})

            # コールバック関数を呼び出し（Streamlit等での進捗表示用）
            if on_epoch_callback is not None:
                if epoch > 0:
                    avg_epoch_time = np.mean(epoch_times)
                    remaining_epochs = epochs - epoch - 1
                    estimated_remaining_time = avg_epoch_time * remaining_epochs
                    estimated_total_time = elapsed_time + estimated_remaining_time
                else:
                    estimated_remaining_time = 0
                    estimated_total_time = elapsed_time

                epoch_info = {
                    'epoch': epoch + 1,  # 1-indexed for display
                    'epochs': epochs,
                    'loss': float(loss),
                    'elapsed_time': float(elapsed_time),
                    'estimated_remaining_time': float(estimated_remaining_time),
                    'estimated_total_time': float(estimated_total_time),
                    'progress': (epoch + 1) / epochs
                }
                try:
                    on_epoch_callback(epoch_info)
                except Exception as e:
                    logger.warning(f"コールバック関数エラー: {str(e)}")

            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        pbar.close()

        # 学習完了メッセージ
        total_training_time = time.time() - training_start_time
        self.last_training_time = total_training_time  # 学習時間を保存
        print(f"\n事前学習完了 - 総学習時間: {self._format_time(total_training_time)}")

        self.trained = True
        return self

    def _format_time(self, seconds):
        """
        秒数を人間が読みやすい形式にフォーマット

        Parameters:
        -----------
        seconds: float
            秒数

        Returns:
        --------
        formatted_time: str
            フォーマットされた時間文字列
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def get_embeddings(self, adjacency, features):
        """
        ノードの埋め込み表現を取得
        """
        self.training = False
        return self.forward(adjacency, features)


class TalentAnalyzer:
    """
    優秀人材分析システム
    """

    def __init__(self):
        self.gnn = SimpleGNN()
        self.scaler = StandardScaler()
        self.label_encoders = {}

        # 設定ファイルからカラム名を読み込む
        self.column_names = get_config('column_names', {})
        self.position_mapping = get_config('position_mapping', {})

    def load_data(self, member_df, acquired_df, skill_df, education_df, license_df):
        """
        CSVデータを読み込んで処理
        """
        print("データ読み込み中...")

        # カラム名を取得
        col_member = self.column_names.get('member', {})
        retired_day_col = col_member.get('retired_day', '退職年月日  ###[retired_day]###')
        member_code_col = col_member.get('code', 'メンバーコード  ###[member_code]###')
        member_name_col = col_member.get('name', 'メンバー名  ###[name]###')

        # データクリーニング
        member_df = member_df[member_df[retired_day_col].isna()].copy()

        # 社員基本情報の処理
        self.members = member_df[member_code_col].unique()
        self.member_names = dict(zip(
            member_df[member_code_col],
            member_df[member_name_col]
        ))

        # スキルデータの処理
        self.process_skills(acquired_df, skill_df, education_df, license_df)

        # 社員特徴量の作成
        self.create_member_features(member_df, acquired_df)

        print(f"データ読み込み完了: 社員{len(self.members)}名, スキル{self.skill_matrix.shape[1]}種")

    def process_skills(self, acquired_df, skill_df, education_df, license_df):
        """
        スキル・教育・資格データを統合処理
        """
        # カラム名を取得
        col_acquired = self.column_names.get('acquired', {})
        member_code_col = col_acquired.get('member_code', 'メンバーコード')
        competence_type_col = col_acquired.get('competence_type', '力量タイプ  ###[competence_type]###')
        competence_code_col = col_acquired.get('competence_code', '力量コード')
        competence_name_col = col_acquired.get('competence_name', '力量名')
        level_col = col_acquired.get('level', 'レベル')

        competence_types = self.column_names.get('competence_types', {})
        skill_type = competence_types.get('skill', 'SKILL')
        education_type = competence_types.get('education', 'EDUCATION')
        license_type = competence_types.get('license', 'LICENSE')

        # スキル保有マトリクスの作成
        members_with_data = acquired_df[member_code_col].unique()

        # 全スキルのリストを作成
        all_skills = {}

        # SKILL
        for _, row in acquired_df[acquired_df[competence_type_col] == skill_type].iterrows():
            skill_code = row[competence_code_col]
            skill_name = row[competence_name_col]
            member_code = row[member_code_col]
            level = row[level_col]

            if skill_code not in all_skills:
                all_skills[skill_code] = {'name': skill_name, 'type': skill_type, 'data': {}}

            try:
                all_skills[skill_code]['data'][member_code] = float(level)
            except:
                all_skills[skill_code]['data'][member_code] = 0

        # EDUCATION
        for _, row in acquired_df[acquired_df[competence_type_col] == education_type].iterrows():
            skill_code = row[competence_code_col]
            skill_name = row[competence_name_col]
            member_code = row[member_code_col]

            if skill_code not in all_skills:
                all_skills[skill_code] = {'name': skill_name, 'type': education_type, 'data': {}}

            all_skills[skill_code]['data'][member_code] = 1.0

        # LICENSE
        for _, row in acquired_df[acquired_df[competence_type_col] == license_type].iterrows():
            skill_code = row[competence_code_col]
            skill_name = row[competence_name_col]
            member_code = row[member_code_col]

            if skill_code not in all_skills:
                all_skills[skill_code] = {'name': skill_name, 'type': license_type, 'data': {}}

            all_skills[skill_code]['data'][member_code] = 1.0

        # マトリクス化
        self.skill_names = {code: info['name'] for code, info in all_skills.items()}
        self.skill_codes = list(all_skills.keys())

        # スキルデータがある社員のみを対象
        self.members = members_with_data

        skill_matrix = np.zeros((len(self.members), len(self.skill_codes)))
        member_to_idx = {member: idx for idx, member in enumerate(self.members)}

        for skill_idx, skill_code in enumerate(self.skill_codes):
            for member_code, value in all_skills[skill_code]['data'].items():
                if member_code in member_to_idx:
                    member_idx = member_to_idx[member_code]
                    skill_matrix[member_idx, skill_idx] = value

        self.skill_matrix = skill_matrix
        self.member_to_idx = member_to_idx

    def create_member_features(self, member_df, acquired_df):
        """
        社員の特徴量を作成
        """
        # カラム名を取得
        col_member = self.column_names.get('member', {})
        member_code_col = col_member.get('code', 'メンバーコード  ###[member_code]###')
        enter_day_col = col_member.get('enter_day', '入社年月日  ###[enter_day]###')
        job_grade_col = col_member.get('job_grade', '職能・等級  ###[job_grade]###')
        job_position_col = col_member.get('job_position', '役職  ###[job_position]###')

        col_acquired = self.column_names.get('acquired', {})
        member_code_acq_col = col_acquired.get('member_code', 'メンバーコード')
        competence_type_col = col_acquired.get('competence_type', '力量タイプ  ###[competence_type]###')

        competence_types = self.column_names.get('competence_types', {})
        license_type = competence_types.get('license', 'LICENSE')

        features = []

        for member_code in self.members:
            member_info = member_df[member_df[member_code_col] == member_code]

            if len(member_info) == 0:
                # デフォルト値
                feat = [0, 0, 0, 0, 0]
            else:
                member_info = member_info.iloc[0]

                # 入社年数
                enter_date = pd.to_datetime(member_info[enter_day_col], errors='coerce')
                if pd.isna(enter_date):
                    years_of_service = 0
                else:
                    years_of_service = (pd.Timestamp.now() - enter_date).days / 365.25

                # 等級
                grade = member_info[job_grade_col]
                if pd.isna(grade):
                    grade_num = 0
                else:
                    grade_num = int(str(grade).replace('等級', '')) if '等級' in str(grade) else 0

                # 役職
                position = member_info[job_position_col]
                position_num = self.position_mapping.get(position, 0)

                # スキル統計量
                member_skills = acquired_df[acquired_df[member_code_acq_col] == member_code]
                n_skills = len(member_skills)
                n_licenses = len(member_skills[member_skills[competence_type_col] == license_type])

                feat = [years_of_service, grade_num, position_num, n_skills, n_licenses]

            features.append(feat)

        self.member_features = np.array(features)

    def train(self, excellent_members, epochs_unsupervised=None, on_epoch_callback=None):
        """
        モデルの学習

        Parameters:
        -----------
        excellent_members: list
            優秀群の社員コードリスト
        epochs_unsupervised: int, optional
            学習エポック数
        on_epoch_callback: callable, optional
            各エポック終了時に呼び出されるコールバック関数
        """
        print(f"\n優秀群: {len(excellent_members)}名で学習開始")

        # グラフ構築
        print("グラフ構築中...")
        adjacency = self.gnn.build_graph(
            self.member_features,
            self.skill_matrix,
            {}
        )

        # 特徴量の正規化
        combined_features = np.hstack([self.member_features, self.skill_matrix])
        combined_features = self.scaler.fit_transform(combined_features)

        # 半教師あり学習
        self.gnn.fit_unsupervised(
            adjacency,
            combined_features,
            epochs=epochs_unsupervised,
            on_epoch_callback=on_epoch_callback
        )

        # 埋め込み表現の取得
        self.embeddings = self.gnn.get_embeddings(adjacency, combined_features)

        # 優秀群のプロトタイプを計算
        excellent_indices = [self.member_to_idx[m] for m in excellent_members if m in self.member_to_idx]
        self.prototype = np.mean(self.embeddings[excellent_indices], axis=0)

        print("学習完了")

    def analyze(self, excellent_members):
        """
        優秀群の特徴分析

        Returns:
        --------
        results: dict
            分析結果
        """
        logger.info(f"分析実行中... 優秀群: {len(excellent_members)}名")

        # 設定から閾値を取得
        top_skills = get_config('analysis.top_skills_display', 50)

        excellent_indices = [self.member_to_idx[m] for m in excellent_members if m in self.member_to_idx]
        non_excellent_indices = [i for i in range(len(self.members)) if i not in excellent_indices]

        # スキル保有率の比較
        excellent_skills = self.skill_matrix[excellent_indices]
        non_excellent_skills = self.skill_matrix[non_excellent_indices]

        skill_importance = []

        for skill_idx in range(len(self.skill_codes)):
            skill_code = self.skill_codes[skill_idx]
            skill_name = self.skill_names[skill_code]

            # 優秀群の保有率
            excellent_count = np.sum(excellent_skills[:, skill_idx] > 0)
            excellent_rate = excellent_count / len(excellent_indices) if len(excellent_indices) > 0 else 0

            # 非優秀群の保有率
            non_excellent_count = np.sum(non_excellent_skills[:, skill_idx] > 0)
            non_excellent_rate = non_excellent_count / len(non_excellent_indices) if len(non_excellent_indices) > 0 else 0

            # 差分
            diff = excellent_rate - non_excellent_rate

            # レベル平均（SKILLの場合）
            excellent_level = np.mean(excellent_skills[:, skill_idx][excellent_skills[:, skill_idx] > 0]) if excellent_count > 0 else 0
            non_excellent_level = np.mean(non_excellent_skills[:, skill_idx][non_excellent_skills[:, skill_idx] > 0]) if non_excellent_count > 0 else 0

            skill_importance.append({
                'skill_code': skill_code,
                'skill_name': skill_name,
                'excellent_rate': excellent_rate,
                'non_excellent_rate': non_excellent_rate,
                'rate_diff': diff,
                'excellent_level': excellent_level,
                'non_excellent_level': non_excellent_level,
                'excellent_count': excellent_count,
                'non_excellent_count': non_excellent_count,
                'importance_score': diff * (1 + excellent_level * 0.1)  # 保有率差 × レベル補正
            })

        # 統計的有意性検定を実行
        skill_importance = self._add_statistical_significance(
            skill_importance,
            len(excellent_indices),
            len(non_excellent_indices)
        )

        # 重要度でソート
        skill_importance = sorted(skill_importance, key=lambda x: x['importance_score'], reverse=True)

        # 社員の優秀度スコアを計算
        member_scores = []
        for idx, member_code in enumerate(self.members):
            # プロトタイプとの距離を計算
            distance = np.linalg.norm(self.embeddings[idx] - self.prototype)
            # スコアに変換（距離が近いほど高スコア）
            max_distance = np.max([np.linalg.norm(emb - self.prototype) for emb in self.embeddings])
            score = (1 - distance / max_distance) * 100

            member_scores.append({
                'member_code': member_code,
                'member_name': self.member_names.get(member_code, '不明'),
                'score': score,
                'is_excellent': member_code in excellent_members
            })

        member_scores = sorted(member_scores, key=lambda x: x['score'], reverse=True)

        results = {
            'skill_importance': skill_importance[:top_skills],
            'member_scores': member_scores,
            'n_excellent': len(excellent_members),
            'n_total': len(self.members),
            'embeddings': self.embeddings,
            'excellent_indices': excellent_indices
        }

        logger.info("分析完了")
        return results

    def _add_statistical_significance(self, skill_importance, n_excellent, n_non_excellent):
        """
        統計的有意性検定を追加

        Parameters:
        -----------
        skill_importance: list
            スキル重要度のリスト
        n_excellent: int
            優秀群の人数
        n_non_excellent: int
            非優秀群の人数

        Returns:
        --------
        skill_importance: list
            統計的検定結果を追加したスキル重要度のリスト
        """
        logger.info("統計的有意性検定を実行中...")

        # 設定を取得
        test_config = get_config('statistical_tests', {})
        alpha = test_config.get('significance_level', 0.05)
        correction_method = test_config.get('multiple_testing_correction', 'fdr_bh')
        show_ci = test_config.get('show_confidence_intervals', True)
        ci_level = test_config.get('confidence_level', 0.95)

        p_values = []

        for skill in skill_importance:
            # 2x2分割表の作成
            excellent_has = skill['excellent_count']
            excellent_not = n_excellent - excellent_has
            non_excellent_has = skill['non_excellent_count']
            non_excellent_not = n_non_excellent - non_excellent_has

            contingency_table = [
                [excellent_has, excellent_not],
                [non_excellent_has, non_excellent_not]
            ]

            # Fisher正確検定
            try:
                odds_ratio, p_value = fisher_exact(contingency_table)
            except:
                odds_ratio, p_value = 1.0, 1.0

            skill['p_value'] = p_value
            skill['odds_ratio'] = odds_ratio
            p_values.append(p_value)

            # 信頼区間の計算（Wald法による近似）
            if show_ci:
                from scipy.stats import norm
                z = norm.ppf(1 - (1 - ci_level) / 2)

                # 優秀群の保有率の信頼区間
                p1 = skill['excellent_rate']
                se1 = np.sqrt(p1 * (1 - p1) / n_excellent) if n_excellent > 0 else 0
                ci1_lower = max(0, p1 - z * se1)
                ci1_upper = min(1, p1 + z * se1)

                # 非優秀群の保有率の信頼区間
                p2 = skill['non_excellent_rate']
                se2 = np.sqrt(p2 * (1 - p2) / n_non_excellent) if n_non_excellent > 0 else 0
                ci2_lower = max(0, p2 - z * se2)
                ci2_upper = min(1, p2 + z * se2)

                skill['excellent_rate_ci'] = (ci1_lower, ci1_upper)
                skill['non_excellent_rate_ci'] = (ci2_lower, ci2_upper)

        # 多重検定補正
        if correction_method != 'none' and len(p_values) > 0:
            try:
                reject, p_adjusted, _, _ = multipletests(
                    p_values,
                    alpha=alpha,
                    method=correction_method
                )

                for i, skill in enumerate(skill_importance):
                    skill['p_adjusted'] = p_adjusted[i]
                    skill['significant'] = reject[i]
                    skill['significance_level'] = self._get_significance_label(p_adjusted[i])
            except:
                logger.warning("多重検定補正に失敗しました")
                for skill in skill_importance:
                    skill['p_adjusted'] = skill['p_value']
                    skill['significant'] = skill['p_value'] < alpha
                    skill['significance_level'] = self._get_significance_label(skill['p_value'])
        else:
            for skill in skill_importance:
                skill['p_adjusted'] = skill['p_value']
                skill['significant'] = skill['p_value'] < alpha
                skill['significance_level'] = self._get_significance_label(skill['p_value'])

        logger.info(f"統計的検定完了: {sum([s['significant'] for s in skill_importance])}個のスキルが有意")

        return skill_importance

    def _get_significance_label(self, p_value):
        """
        有意性のラベルを取得

        Parameters:
        -----------
        p_value: float
            p値

        Returns:
        --------
        label: str
            有意性ラベル
        """
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'n.s.'

    def evaluate_model(self, excellent_members, epochs_unsupervised=None):
        """
        モデルの評価を実行

        Parameters:
        -----------
        excellent_members: list
            優秀群の社員コードリスト
        epochs_unsupervised: int, optional
            学習エポック数

        Returns:
        --------
        evaluation_results: dict
            評価結果
        """
        logger.info("モデル評価を開始...")

        eval_config = get_config('evaluation', {})

        if not eval_config.get('enabled', True):
            logger.info("モデル評価が無効になっています")
            return None

        method = eval_config.get('method', 'holdout')

        results = {}

        if method == 'holdout':
            results = self._evaluate_holdout(excellent_members, epochs_unsupervised)
        elif method == 'loocv':
            results = self._evaluate_loocv(excellent_members, epochs_unsupervised)
        elif method == 'both':
            results['holdout'] = self._evaluate_holdout(excellent_members, epochs_unsupervised)
            results['loocv'] = self._evaluate_loocv(excellent_members, epochs_unsupervised)
        else:
            # サンプル数に応じて自動選択
            if len(excellent_members) <= eval_config.get('loocv', {}).get('small_sample_threshold', 10):
                logger.info(f"優秀群が{len(excellent_members)}名のため、LOOCVを使用します")
                results = self._evaluate_loocv(excellent_members, epochs_unsupervised)
            else:
                results = self._evaluate_holdout(excellent_members, epochs_unsupervised)

        logger.info("モデル評価完了")
        return results

    def _evaluate_holdout(self, excellent_members, epochs_unsupervised=None):
        """
        Holdout法によるモデル評価

        Parameters:
        -----------
        excellent_members: list
            優秀群の社員コードリスト
        epochs_unsupervised: int, optional
            学習エポック数

        Returns:
        --------
        results: dict
            評価結果
        """
        logger.info("Holdout法で評価中...")

        eval_config = get_config('evaluation.holdout', {})
        test_ratio = eval_config.get('test_ratio', 0.2)
        random_seed = eval_config.get('random_seed', 42)

        np.random.seed(random_seed)

        # テストデータの分割
        n_test = max(1, int(len(excellent_members) * test_ratio))
        test_indices = np.random.choice(len(excellent_members), n_test, replace=False)

        test_members = [excellent_members[i] for i in test_indices]
        train_members = [m for i, m in enumerate(excellent_members) if i not in test_indices]

        logger.info(f"訓練データ: {len(train_members)}名, テストデータ: {len(test_members)}名")

        # 訓練
        self.train(train_members, epochs_unsupervised=epochs_unsupervised)

        # 評価
        train_scores, train_labels = self._get_scores_and_labels(train_members)
        test_scores, test_labels = self._get_scores_and_labels(test_members)

        # メトリクスの計算
        train_metrics = self._calculate_metrics(train_scores, train_labels, "Train")
        test_metrics = self._calculate_metrics(test_scores, test_labels, "Test")

        # 過学習の検出
        overfitting_threshold = get_config('evaluation.warn_overfitting_threshold', 0.2)
        auc_diff = train_metrics.get('auc', 0) - test_metrics.get('auc', 0)
        is_overfitting = auc_diff > overfitting_threshold

        results = {
            'method': 'holdout',
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'n_train': len(train_members),
            'n_test': len(test_members),
            'is_overfitting': is_overfitting,
            'auc_diff': auc_diff
        }

        if is_overfitting:
            logger.warning(f"過学習の可能性があります（AUC差分: {auc_diff:.3f}）")

        return results

    def _evaluate_loocv(self, excellent_members, epochs_unsupervised=None):
        """
        Leave-One-Out Cross-Validation によるモデル評価

        Parameters:
        -----------
        excellent_members: list
            優秀群の社員コードリスト
        epochs_unsupervised: int, optional
            学習エポック数

        Returns:
        --------
        results: dict
            評価結果
        """
        logger.info("LOOCV（Leave-One-Out Cross-Validation）で評価中...")

        all_scores = []
        all_labels = []

        for i, test_member in enumerate(excellent_members):
            logger.info(f"LOOCV: {i+1}/{len(excellent_members)}")

            # 訓練データ（1名を除く）
            train_members = [m for j, m in enumerate(excellent_members) if j != i]

            # 訓練
            self.train(train_members, epochs_unsupervised=epochs_unsupervised)

            # テストメンバーのスコアを取得
            test_member_idx = self.member_to_idx.get(test_member)
            if test_member_idx is not None:
                distance = np.linalg.norm(self.embeddings[test_member_idx] - self.prototype)
                max_distance = np.max([np.linalg.norm(emb - self.prototype) for emb in self.embeddings])
                score = (1 - distance / max_distance) * 100

                all_scores.append(score)
                all_labels.append(1)  # 優秀群

        # 非優秀群のスコアも取得（最終モデルで）
        non_excellent_members = [m for m in self.members if m not in excellent_members]
        for member_code in non_excellent_members:
            member_idx = self.member_to_idx.get(member_code)
            if member_idx is not None:
                distance = np.linalg.norm(self.embeddings[member_idx] - self.prototype)
                max_distance = np.max([np.linalg.norm(emb - self.prototype) for emb in self.embeddings])
                score = (1 - distance / max_distance) * 100

                all_scores.append(score)
                all_labels.append(0)  # 非優秀群

        # メトリクスの計算
        metrics = self._calculate_metrics(all_scores, all_labels, "LOOCV")

        results = {
            'method': 'loocv',
            'metrics': metrics,
            'n_folds': len(excellent_members)
        }

        return results

    def _get_scores_and_labels(self, excellent_members):
        """
        優秀群と非優秀群のスコアとラベルを取得

        Parameters:
        -----------
        excellent_members: list
            優秀群の社員コードリスト

        Returns:
        --------
        scores: array
            スコア配列
        labels: array
            ラベル配列（1: 優秀群, 0: 非優秀群）
        """
        scores = []
        labels = []

        # 優秀群のスコア
        for member_code in excellent_members:
            member_idx = self.member_to_idx.get(member_code)
            if member_idx is not None:
                distance = np.linalg.norm(self.embeddings[member_idx] - self.prototype)
                max_distance = np.max([np.linalg.norm(emb - self.prototype) for emb in self.embeddings])
                score = (1 - distance / max_distance) * 100

                scores.append(score)
                labels.append(1)

        # 非優秀群のスコア
        non_excellent_members = [m for m in self.members if m not in excellent_members]
        for member_code in non_excellent_members:
            member_idx = self.member_to_idx.get(member_code)
            if member_idx is not None:
                distance = np.linalg.norm(self.embeddings[member_idx] - self.prototype)
                max_distance = np.max([np.linalg.norm(emb - self.prototype) for emb in self.embeddings])
                score = (1 - distance / max_distance) * 100

                scores.append(score)
                labels.append(0)

        return np.array(scores), np.array(labels)

    def _calculate_metrics(self, scores, labels, dataset_name=""):
        """
        評価メトリクスを計算

        Parameters:
        -----------
        scores: array
            予測スコア
        labels: array
            真のラベル
        dataset_name: str
            データセット名（ロギング用）

        Returns:
        --------
        metrics: dict
            評価メトリクス
        """
        metrics = {}

        # 閾値を決定（スコアの中央値を使用）
        threshold = np.median(scores)

        # 予測ラベル
        predictions = (scores >= threshold).astype(int)

        try:
            # AUC
            if len(np.unique(labels)) > 1:
                auc = roc_auc_score(labels, scores)
                metrics['auc'] = auc
            else:
                metrics['auc'] = 0.0
                logger.warning(f"{dataset_name}: ラベルが1種類のみのため、AUCを計算できません")

            # Precision, Recall, F1
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)

            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
            metrics['threshold'] = threshold

            logger.info(f"{dataset_name} - AUC: {metrics.get('auc', 0):.3f}, "
                       f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        except Exception as e:
            logger.error(f"メトリクス計算エラー: {str(e)}")
            metrics = {'auc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        return metrics

    def save_model(self, excellent_members, version=None):
        """
        学習済みモデルを保存

        Parameters:
        -----------
        excellent_members: list
            優秀群の社員コードリスト
        version: str, optional
            バージョン名
        """
        versioning_config = get_config('versioning', {})

        if not versioning_config.get('enabled', True) or not versioning_config.get('save_models', True):
            logger.info("モデル保存が無効になっています")
            return

        model_dir = Path(versioning_config.get('model_dir', './models'))
        model_dir.mkdir(parents=True, exist_ok=True)

        # バージョン名の生成
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = model_dir / f"model_{version}.pkl"

        # 保存するデータ
        model_data = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'weights': self.gnn.weights,
            'embeddings': self.embeddings,
            'prototype': self.prototype,
            'scaler': self.scaler,
            'member_to_idx': self.member_to_idx,
            'skill_codes': self.skill_codes,
            'skill_names': self.skill_names
        }

        # メタデータの追加
        if versioning_config.get('include_metadata', True):
            model_data['metadata'] = {
                'n_members': len(self.members),
                'n_skills': len(self.skill_codes),
                'n_excellent': len(excellent_members),
                'excellent_members': excellent_members,
                'model_params': {
                    'n_layers': self.gnn.n_layers,
                    'hidden_dim': self.gnn.hidden_dim,
                    'dropout': self.gnn.dropout,
                    'learning_rate': self.gnn.learning_rate
                }
            }

        # 保存
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"モデルを保存しました: {model_path}")

        # メタデータをJSONでも保存
        metadata_path = model_dir / f"model_{version}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(model_data.get('metadata', {}), f, ensure_ascii=False, indent=2)

    def load_model(self, version):
        """
        学習済みモデルを読み込み

        Parameters:
        -----------
        version: str
            バージョン名
        """
        model_dir = Path(get_config('versioning.model_dir', './models'))
        model_path = model_dir / f"model_{version}.pkl"

        if not model_path.exists():
            logger.error(f"モデルファイルが見つかりません: {model_path}")
            return False

        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.gnn.weights = model_data['weights']
            self.embeddings = model_data['embeddings']
            self.prototype = model_data['prototype']
            self.scaler = model_data['scaler']
            self.member_to_idx = model_data['member_to_idx']
            self.skill_codes = model_data['skill_codes']
            self.skill_names = model_data['skill_names']

            # 属性の初期化（古いモデル互換性のため）
            if not hasattr(self.gnn, 'last_training_time'):
                self.gnn.last_training_time = None

            logger.info(f"モデルを読み込みました: {model_path}")
            logger.info(f"バージョン: {model_data.get('version')}, タイムスタンプ: {model_data.get('timestamp')}")

            return True

        except Exception as e:
            logger.error(f"モデル読み込みエラー: {str(e)}")
            return False

    def estimate_causal_effects(self, excellent_members):
        """
        因果推論によるスキルの真の効果を推定

        Parameters:
        -----------
        excellent_members: list
            優秀群の社員コードリスト

        Returns:
        --------
        causal_results: list
            各スキルの因果効果推定結果
        """
        causal_config = get_config('causal_inference', {})

        if not causal_config.get('enabled', True):
            logger.info("因果推論が無効になっています")
            return None

        logger.info("因果推論を開始...")

        excellent_indices = [self.member_to_idx[m] for m in excellent_members if m in self.member_to_idx]

        # 交絡因子の取得
        confounders = self._get_confounders()

        causal_results = []

        for skill_idx in range(len(self.skill_codes)):
            result = self._estimate_skill_causal_effect(
                skill_idx,
                excellent_indices,
                confounders,
                causal_config
            )

            if result is not None:
                causal_results.append(result)

        # 因果効果の大きい順にソート
        # causal_effectがNoneでない結果のみを対象
        valid_causal_results = [r for r in causal_results if r.get('causal_effect') is not None]
        valid_causal_results.sort(key=lambda x: abs(x.get('causal_effect')), reverse=True)

        logger.info(f"因果推論完了: {len(valid_causal_results)}個のスキルを分析")

        return valid_causal_results

    def _get_confounders(self):
        """
        交絡因子を取得

        Returns:
        --------
        confounders: array
            交絡因子の行列
        """
        confounder_config = get_config('causal_inference.confounders', {})

        confounders = []

        for idx in range(len(self.members)):
            confounder_vec = []

            # 勤続年数
            if confounder_config.get('use_years_of_service', True):
                confounder_vec.append(self.member_features[idx, 0])

            # 等級
            if confounder_config.get('use_grade', True):
                confounder_vec.append(self.member_features[idx, 1])

            # 役職
            if confounder_config.get('use_position', True):
                confounder_vec.append(self.member_features[idx, 2])

            # スキル保有数
            if confounder_config.get('use_skill_count', True):
                confounder_vec.append(self.member_features[idx, 3])

            confounders.append(confounder_vec)

        return np.array(confounders)

    def _estimate_skill_causal_effect(self, skill_idx, excellent_indices, confounders, causal_config):
        """
        特定スキルの因果効果を推定

        Parameters:
        -----------
        skill_idx: int
            スキルのインデックス
        excellent_indices: list
            優秀群のインデックスリスト
        confounders: array
            交絡因子の行列
        causal_config: dict
            因果推論の設定

        Returns:
        --------
        result: dict
            因果効果の推定結果
        """
        skill_code = self.skill_codes[skill_idx]
        skill_name = self.skill_names[skill_code]

        # スキル保有フラグ
        has_skill = (self.skill_matrix[:, skill_idx] > 0).astype(int)

        # スキル保有者が少なすぎる場合はスキップ
        if has_skill.sum() < 5 or has_skill.sum() > len(has_skill) - 5:
            return None

        try:
            # 1. 傾向スコアの計算
            ps_model = LogisticRegression(max_iter=1000, random_state=42)
            ps_model.fit(confounders, has_skill)
            propensity_scores = ps_model.predict_proba(confounders)[:, 1]

            # 2. 傾向スコアマッチング
            treated_indices = np.where(has_skill == 1)[0]
            control_indices = np.where(has_skill == 0)[0]

            caliper = causal_config.get('propensity_score', {}).get('caliper', 0.1)
            matched_pairs = []

            for treated_idx in treated_indices:
                # 傾向スコアが最も近い対照群を探す
                ps_diff = np.abs(propensity_scores[control_indices] - propensity_scores[treated_idx])

                if len(ps_diff) == 0:
                    continue

                min_diff_idx = ps_diff.argmin()

                if ps_diff[min_diff_idx] < caliper:
                    matched_control = control_indices[min_diff_idx]
                    matched_pairs.append((treated_idx, matched_control))

            min_pairs = causal_config.get('propensity_score', {}).get('min_matched_pairs', 5)

            if len(matched_pairs) < min_pairs:
                return {
                    'skill_code': skill_code,
                    'skill_name': skill_name,
                    'causal_effect': None,
                    'p_value': None,
                    'n_matched_pairs': len(matched_pairs),
                    'status': 'insufficient_matches',
                    'interpretation': f'マッチング不可（ペア数: {len(matched_pairs)} < {min_pairs}）'
                }

            # 3. マッチングされたペアで効果を推定
            treated_outcomes = []
            control_outcomes = []

            for treated_idx, control_idx in matched_pairs:
                # アウトカム = 優秀かどうか（1 or 0）
                treated_outcomes.append(1 if treated_idx in excellent_indices else 0)
                control_outcomes.append(1 if control_idx in excellent_indices else 0)

            # 平均処置効果（ATE: Average Treatment Effect）
            ate = np.mean(treated_outcomes) - np.mean(control_outcomes)

            # 統計的有意性（t検定）
            if len(set(treated_outcomes)) > 1 and len(set(control_outcomes)) > 1:
                t_stat, p_value = ttest_ind(treated_outcomes, control_outcomes)
            else:
                t_stat, p_value = 0, 1.0

            # 信頼区間の計算（簡易版）
            from scipy.stats import norm
            n_t = len(treated_outcomes)
            n_c = len(control_outcomes)

            if n_t > 1 and n_c > 1:
                se = np.sqrt(
                    np.var(treated_outcomes) / n_t +
                    np.var(control_outcomes) / n_c
                )
                z = norm.ppf(0.975)  # 95% CI
                ci_lower = ate - z * se
                ci_upper = ate + z * se
            else:
                ci_lower, ci_upper = ate, ate

            return {
                'skill_code': skill_code,
                'skill_name': skill_name,
                'causal_effect': ate,
                'p_value': p_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_matched_pairs': len(matched_pairs),
                'n_treated': len(treated_indices),
                'n_control': len(control_indices),
                'status': 'success',
                'interpretation': f'このスキルを習得すると優秀になる確率が{ate*100:.1f}%変化',
                'significant': p_value < 0.05
            }

        except Exception as e:
            logger.error(f"スキル {skill_name} の因果推論エラー: {str(e)}")
            return {
                'skill_code': skill_code,
                'skill_name': skill_name,
                'causal_effect': None,
                'status': 'error',
                'interpretation': f'エラー: {str(e)}'
            }

    def analyze_skill_interactions(self, excellent_members):
        """
        スキル間の相互作用を分析

        Parameters:
        -----------
        excellent_members: list
            優秀群の社員コードリスト

        Returns:
        --------
        interaction_results: list
            スキル相互作用の分析結果
        """
        interaction_config = get_config('skill_interaction', {})

        if not interaction_config.get('enabled', True):
            logger.info("スキル相互作用分析が無効になっています")
            return None

        logger.info("スキル相互作用分析を開始...")

        excellent_indices = [self.member_to_idx[m] for m in excellent_members if m in self.member_to_idx]

        n_skills = self.skill_matrix.shape[1]

        # 組み合わせの数を制限
        max_pairs = interaction_config.get('max_pairs_to_analyze', 1000)

        # スキル保有率が低すぎるスキルを除外（計算時間削減）
        skill_rates = np.mean(self.skill_matrix > 0, axis=0)
        valid_skills = np.where((skill_rates >= 0.05) & (skill_rates <= 0.95))[0]

        if len(valid_skills) > 100:
            # スキル数が多い場合は、保有率が中程度のスキルを優先
            valid_skills = valid_skills[:100]

        skill_pairs = list(combinations(valid_skills, 2))

        # ペア数を制限
        if len(skill_pairs) > max_pairs:
            # ランダムにサンプリング
            np.random.seed(42)
            selected_pairs = np.random.choice(len(skill_pairs), max_pairs, replace=False)
            skill_pairs = [skill_pairs[i] for i in selected_pairs]

        logger.info(f"{len(skill_pairs)}個のスキルペアを分析中...")

        interaction_effects = []

        for skill_a, skill_b in skill_pairs:
            result = self._analyze_skill_pair_interaction(
                skill_a,
                skill_b,
                excellent_indices,
                interaction_config
            )

            if result is not None:
                interaction_effects.append(result)

        # 相乗効果の大きい順にソート
        interaction_effects.sort(key=lambda x: x.get('synergy', 0), reverse=True)

        logger.info(f"スキル相互作用分析完了: {len(interaction_effects)}個の有意な相乗効果を発見")

        return interaction_effects

    def _analyze_skill_pair_interaction(self, skill_a, skill_b, excellent_indices, interaction_config):
        """
        2つのスキルの相互作用を分析

        Parameters:
        -----------
        skill_a: int
            スキルAのインデックス
        skill_b: int
            スキルBのインデックス
        excellent_indices: list
            優秀群のインデックスリスト
        interaction_config: dict
            相互作用分析の設定

        Returns:
        --------
        result: dict
            相互作用の分析結果
        """
        skill_a_code = self.skill_codes[skill_a]
        skill_b_code = self.skill_codes[skill_b]
        skill_a_name = self.skill_names[skill_a_code]
        skill_b_name = self.skill_names[skill_b_code]

        # スキル保有フラグ
        has_a = (self.skill_matrix[:, skill_a] > 0).astype(int)
        has_b = (self.skill_matrix[:, skill_b] > 0).astype(int)

        # 4つのグループに分ける
        neither = (has_a == 0) & (has_b == 0)
        only_a = (has_a == 1) & (has_b == 0)
        only_b = (has_a == 0) & (has_b == 1)
        both = (has_a == 1) & (has_b == 1)

        # 各グループの人数が少なすぎる場合はスキップ
        if neither.sum() < 3 or both.sum() < 3:
            return None

        # 各グループの優秀率
        rate_neither = self._excellence_rate(neither, excellent_indices)
        rate_a = self._excellence_rate(only_a, excellent_indices)
        rate_b = self._excellence_rate(only_b, excellent_indices)
        rate_both = self._excellence_rate(both, excellent_indices)

        # 相加効果 = A単独の効果 + B単独の効果
        effect_a = rate_a - rate_neither
        effect_b = rate_b - rate_neither
        additive_effect = effect_a + effect_b

        # 実際の効果
        actual_effect = rate_both - rate_neither

        # 相乗効果 = 実際の効果 - 相加効果
        synergy = actual_effect - additive_effect

        # 閾値チェック
        synergy_threshold = interaction_config.get('synergy_threshold', 0.1)
        min_both_rate = interaction_config.get('min_both_rate', 0.7)

        if synergy > synergy_threshold and rate_both >= min_both_rate:
            return {
                'skill_a_code': skill_a_code,
                'skill_a_name': skill_a_name,
                'skill_b_code': skill_b_code,
                'skill_b_name': skill_b_name,
                'synergy': synergy,
                'rate_neither': rate_neither,
                'rate_a': rate_a,
                'rate_b': rate_b,
                'rate_both': rate_both,
                'n_neither': neither.sum(),
                'n_a': only_a.sum(),
                'n_b': only_b.sum(),
                'n_both': both.sum(),
                'effect_a': effect_a,
                'effect_b': effect_b,
                'additive_effect': additive_effect,
                'actual_effect': actual_effect,
                'interpretation': f'両方習得で+{synergy*100:.0f}%の追加効果（優秀率: {rate_both*100:.0f}%）'
            }

        return None

    def _excellence_rate(self, mask, excellent_indices):
        """
        特定グループの優秀率を計算

        Parameters:
        -----------
        mask: array
            グループのマスク
        excellent_indices: list
            優秀群のインデックスリスト

        Returns:
        --------
        rate: float
            優秀率
        """
        if mask.sum() == 0:
            return 0.0

        group_indices = np.where(mask)[0]
        excellent_in_group = len(set(group_indices) & set(excellent_indices))
        return excellent_in_group / len(group_indices)


def load_csv_files(member_path, acquired_path, skill_path, education_path, license_path):
    """
    CSVファイルを読み込み
    """
    encoding = get_config('files.encoding', 'utf-8-sig')

    member_df = pd.read_csv(member_path, encoding=encoding)
    acquired_df = pd.read_csv(acquired_path, encoding=encoding)
    skill_df = pd.read_csv(skill_path, encoding=encoding)
    education_df = pd.read_csv(education_path, encoding=encoding)
    license_df = pd.read_csv(license_path, encoding=encoding)

    return member_df, acquired_df, skill_df, education_df, license_df
