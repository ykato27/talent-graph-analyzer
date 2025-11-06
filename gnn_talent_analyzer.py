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
from scipy.stats import fisher_exact, ttest_ind, norm
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import warnings
import logging
import os
import pickle
import json
import re
from datetime import datetime
from pathlib import Path
from config_loader import get_config

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

# ==================== カスタム例外クラス ====================
class TalentAnalyzerError(Exception):
    """タレント分析システムの基底例外"""
    pass


class DataValidationError(TalentAnalyzerError):
    """データ検証エラー"""
    pass


class DataLoadingError(TalentAnalyzerError):
    """データ読み込みエラー"""
    pass


class ConfigurationError(TalentAnalyzerError):
    """設定エラー"""
    pass


class ModelTrainingError(TalentAnalyzerError):
    """モデル学習エラー"""
    pass


class ModelEvaluationError(TalentAnalyzerError):
    """モデル評価エラー"""
    pass


class CausalInferenceError(TalentAnalyzerError):
    """因果推論エラー"""
    pass


class AnalysisError(TalentAnalyzerError):
    """分析エラー"""
    pass


# ==================== 定数定義 ====================
# 数値計算パラメータ
NUMERICAL_EPSILON = 1e-8
HE_INIT_SCALE = 2.0

# データ処理パラメータ
MIN_SKILL_HOLDERS = 5
MIN_SKILL_HOLDERS_FOR_CAUSAL = 5
MIN_GROUP_SIZE = 3
MAX_VALID_SKILLS = 100
MAX_INTERACTION_PAIRS = 1000
DEFAULT_RANDOM_STATE = 42

# 統計分析パラメータ
DEFAULT_SIGNIFICANCE_LEVEL = 0.05
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_SKILL_RATE_LOWER = 0.05
DEFAULT_SKILL_RATE_UPPER = 0.95
DEFAULT_CALIPER = 0.1
MIN_MATCHED_PAIRS = 5

# モデルパラメータ
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_DROPOUT = 0.3
DEFAULT_K_NEIGHBORS = 10
DEFAULT_EPOCHS = 100
DEFAULT_EARLY_STOPPING_PATIENCE = 20

# ファイル関連
DEFAULT_LOG_DIR = './logs'
DEFAULT_MODEL_DIR = './models'
DEFAULT_FILE_ENCODING = 'utf-8-sig'


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

    def fit_unsupervised(self, adjacency, features, epochs=None):
        """
        半教師あり学習（ラベルなしで学習）
        Deep Graph Infomax的なアプローチ
        """
        if epochs is None:
            epochs = get_config('training.default_epochs', 100)

        logger.info("Semi-supervised pre-training started")
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

        for epoch in range(epochs):
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

            if epoch % 20 == 0:
                logger.debug(f"事前学習: Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"早期停止: Epoch {epoch}で学習を停止しました")
                break

        self.trained = True
        logger.info("グラフ事前学習が完了しました")
        return self

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

    def _validate_input_data(self, df, df_name, required_columns=None):
        """
        入力データフレームの検証

        Args:
            df: 検証対象のDataFrame
            df_name: DataFrameの名前（エラーメッセージ用）
            required_columns: 必須カラムのリスト（デフォルト: None）

        Raises:
            DataValidationError: 検証失敗時
        """
        # 型チェック
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(
                f"{df_name} は pandas.DataFrame である必要があります。"
                f"受け取った型: {type(df).__name__}"
            )

        # 空チェック
        if df.empty:
            raise DataValidationError(f"{df_name} は空のDataFrameです")

        # 必須カラム確認
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise DataValidationError(
                    f"{df_name} に必須カラムが不足しています。"
                    f"不足カラム: {missing_cols}. "
                    f"利用可能なカラム: {list(df.columns)}"
                )

        # 欠損値の多さをチェック
        missing_ratio = df.isnull().sum() / len(df)
        if (missing_ratio > 0.5).any():
            high_missing_cols = missing_ratio[missing_ratio > 0.5].index.tolist()
            logger.warning(
                f"{df_name} に50%以上の欠損値があるカラム: {high_missing_cols}. "
                f"分析結果に影響を与える可能性があります"
            )

        logger.info(f"{df_name} の検証成功: {len(df)}行 × {len(df.columns)}列")

    def load_data(self, member_df, acquired_df, skill_df, education_df, license_df):
        """
        CSVデータを読み込んで処理

        Args:
            member_df: 社員マスタDataFrame
            acquired_df: スキル習得DataFrameframe
            skill_df: スキルマスタDataFrame
            education_df: 教育マスタDataFrame
            license_df: 資格マスタDataFrame

        Raises:
            DataValidationError: 入力データの検証失敗時
        """
        logger.info("データ読み込みを開始しました")

        # 入力データの検証
        self._validate_input_data(member_df, 'member_df')
        self._validate_input_data(acquired_df, 'acquired_df')
        self._validate_input_data(skill_df, 'skill_df')
        self._validate_input_data(education_df, 'education_df')
        self._validate_input_data(license_df, 'license_df')

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

        logger.info(f"データ読み込み完了: 社員{len(self.members)}名, スキル{self.skill_matrix.shape[1]}種")

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
            except (ValueError, TypeError) as e:
                logger.debug(
                    f"スキルレベルの変換に失敗 "
                    f"(メンバー: {member_code}, スキル: {skill_code}, 値: {level}): {e}. "
                    f"デフォルト値 0 を使用"
                )
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

    def train(self, excellent_members, epochs_unsupervised=None):
        """
        モデルの学習

        Parameters:
        -----------
        excellent_members: list
            優秀群の社員コードリスト
        epochs_unsupervised: int, optional
            学習エポック数
        """
        logger.info(f"優秀群 {len(excellent_members)}名の学習を開始します")

        # グラフ構築
        logger.info("グラフ構築を開始...")
        adjacency = self.gnn.build_graph(
            self.member_features,
            self.skill_matrix,
            {}
        )

        # 特徴量の正規化
        combined_features = np.hstack([self.member_features, self.skill_matrix])
        combined_features = self.scaler.fit_transform(combined_features)

        # 半教師あり学習
        self.gnn.fit_unsupervised(adjacency, combined_features, epochs=epochs_unsupervised)

        # 埋め込み表現の取得
        self.embeddings = self.gnn.get_embeddings(adjacency, combined_features)

        # 優秀群のプロトタイプを計算
        excellent_indices = [self.member_to_idx[m] for m in excellent_members if m in self.member_to_idx]
        self.prototype = np.mean(self.embeddings[excellent_indices], axis=0)

        logger.info(f"学習完了しました（プロトタイプ次元: {self.prototype.shape[0]}）")

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
            except (ValueError, ZeroDivisionError) as e:
                logger.warning(
                    f"Fisher検定失敗 (スキル: {skill['skill_name']}): {e}. "
                    f"デフォルト値を使用"
                )
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
            except (ValueError, IndexError) as e:
                logger.warning(
                    f"多重検定補正に失敗しました (方法: {correction_method}): {e}. "
                    f"補正なしの p-値を使用"
                )
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

    def _sanitize_version(self, version: str) -> str:
        """
        バージョン文字列をサニタイズ（パストラバーサル対策）

        許可される文字: 英数字、アンダースコア、ハイフン
        Path traversal 試行（"..", "/" 等）を防止

        Args:
            version: バージョン文字列

        Returns:
            サニタイズされたバージョン文字列

        Raises:
            ValueError: サニタイズされたバージョンが不正な形式の場合
        """
        # 許可される文字のみを抽出：英数字、アンダースコア、ハイフン
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', version)

        # サニタイズ前後で変更があるかチェック
        if sanitized != version:
            logger.warning(
                f"バージョン文字列をサニタイズしました: "
                f"'{version}' → '{sanitized}' "
                f"（不正な文字を削除）"
            )

        # 空の文字列チェック
        if not sanitized:
            raise ValueError(
                f"バージョン文字列が無効です（許可される文字なし）: '{version}'"
            )

        return sanitized

    def save_model(self, excellent_members, version=None):
        """
        学習済みモデルを保存

        Parameters:
        -----------
        excellent_members: list
            優秀群の社員コードリスト
        version: str, optional
            バージョン名

        Raises:
            ModelTrainingError: モデル保存に失敗した場合
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

        # バージョン文字列をサニタイズ（セキュリティ対策）
        try:
            version = self._sanitize_version(version)
        except ValueError as e:
            logger.error(f"バージョン文字列の検証失敗: {e}")
            raise ModelTrainingError(f"無効なバージョン名: {e}") from e

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

        Returns:
        --------
        bool: 読み込み成功時はTrue、失敗時はFalse
        """
        # バージョン文字列をサニタイズ（セキュリティ対策）
        try:
            version = self._sanitize_version(version)
        except ValueError as e:
            logger.error(f"バージョン文字列の検証失敗: {e}")
            return False

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

            logger.info(f"モデルを読み込みました: {model_path}")
            logger.info(f"バージョン: {model_data.get('version')}, タイムスタンプ: {model_data.get('timestamp')}")

            return True

        except (FileNotFoundError, IOError) as e:
            logger.error(f"モデルファイルの読み込みに失敗しました: {e}", exc_info=True)
            return False
        except (KeyError, ValueError) as e:
            logger.error(f"モデルデータが不正または欠損しています: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"予期しないエラーが発生しました（モデル読み込み）: {e}", exc_info=True)
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
        causal_results.sort(key=lambda x: abs(x.get('causal_effect', 0)), reverse=True)

        logger.info(f"因果推論完了: {len(causal_results)}個のスキルを分析")

        return causal_results

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

        except (ValueError, ZeroDivisionError) as e:
            logger.warning(
                f"スキル {skill_name} の因果推論計算エラー（数値エラー）: {e}. "
                f"データが不適切である可能性があります"
            )
            return {
                'skill_code': skill_code,
                'skill_name': skill_name,
                'causal_effect': None,
                'status': 'numerical_error',
                'interpretation': f'数値計算エラー（スキル分布が不適切）'
            }
        except Exception as e:
            logger.error(
                f"スキル {skill_name} の因果推論エラー（予期しないエラー）: {e}",
                exc_info=True
            )
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

    # ==================== Layer 1: 優秀者特性の逆向き分析 ====================

    def analyze_skill_profile_of_excellent_members(self, excellent_members):
        """
        Layer 1: 優秀者特性の逆向き分析

        優秀群（n=10）と傾向スコアマッチング後の非優秀群を比較し、
        「優秀者が持つべきスキルプロファイル」を分析

        Parameters:
        -----------
        excellent_members: list
            優秀群の社員コードリスト

        Returns:
        --------
        skill_profile: list[dict]
            スキルプロファイルのリスト（重要度順）
            各要素には以下を含む：
            - skill_code: スキルコード
            - skill_name: スキル名
            - importance: 重要度（0-1）
            - p_excellent: 優秀群での習得率
            - p_non_excellent: 非優秀群での習得率
            - ci_excellent: 優秀群の習得率CI
            - ci_non_excellent: 非優秀群の習得率CI
            - p_value: Fisher検定のp値
            - significant: 統計的有意性フラグ
            - interpretation: 解釈文
        """
        logger.info("=== Layer 1: 優秀者特性の逆向き分析を開始 ===")

        excellent_indices = np.array([
            self.member_to_idx[m] for m in excellent_members
            if m in self.member_to_idx
        ])

        if len(excellent_indices) == 0:
            logger.warning("優秀群のインデックスが空です")
            return []

        # 非優秀群のインデックス
        all_indices = np.arange(len(self.members))
        non_excellent_indices = np.array([
            idx for idx in all_indices if idx not in excellent_indices
        ])

        logger.info(f"優秀群: {len(excellent_indices)}名, 非優秀群: {len(non_excellent_indices)}名")

        # 傾向スコアマッチングで対照群を作成
        matched_non_excellent_indices = self._create_matched_control_group(
            excellent_indices,
            non_excellent_indices
        )

        logger.info(f"マッチング後の対照群: {len(matched_non_excellent_indices)}名")

        # 各スキルについて、優秀群と非優秀群のプロファイルを比較
        skill_profile = []

        for skill_idx in range(len(self.skill_codes)):
            result = self._compare_skill_acquisition(
                skill_idx,
                excellent_indices,
                matched_non_excellent_indices
            )

            if result is not None:
                skill_profile.append(result)

        # 重要度（差分）でソート
        skill_profile.sort(
            key=lambda x: abs(x['importance']),
            reverse=True
        )

        logger.info(f"スキルプロファイル分析完了: {len(skill_profile)}個のスキル")

        return skill_profile

    def _create_matched_control_group(self, excellent_indices, non_excellent_indices):
        """
        傾向スコアマッチングで対照群を作成

        優秀群との共変量バランスを取るため、
        傾向スコアが最も近い非優秀者を対応させる

        Parameters:
        -----------
        excellent_indices: array
            優秀群のインデックス
        non_excellent_indices: array
            非優秀群のインデックス

        Returns:
        --------
        matched_indices: array
            マッチングされた非優秀群のインデックス
        """
        try:
            # 優秀フラグを作成
            is_excellent = np.zeros(len(self.members))
            is_excellent[excellent_indices] = 1

            # 傾向スコアモデルをフィット
            ps_model = LogisticRegression(
                max_iter=1000,
                random_state=DEFAULT_RANDOM_STATE
            )
            ps_model.fit(self.member_features, is_excellent)
            propensity_scores = ps_model.predict_proba(self.member_features)[:, 1]

            # 各優秀者について、最も傾向スコアが近い非優秀者を見つける
            matched_indices = []
            caliper = DEFAULT_CALIPER

            for exc_idx in excellent_indices:
                exc_ps = propensity_scores[exc_idx]

                # マッチング候補の傾向スコア差
                ps_diff = np.abs(propensity_scores[non_excellent_indices] - exc_ps)

                if len(ps_diff) == 0:
                    continue

                # 最小の差を持つ非優秀者を選択
                best_idx = non_excellent_indices[ps_diff.argmin()]

                if ps_diff.min() < caliper:
                    matched_indices.append(best_idx)

            logger.debug(f"傾向スコアマッチング: {len(matched_indices)}/{len(excellent_indices)} ペアが成功")

            return np.array(matched_indices)

        except Exception as e:
            logger.warning(f"傾向スコアマッチング失敗、全非優秀群を使用: {e}")
            return non_excellent_indices

    def _compare_skill_acquisition(self, skill_idx, excellent_indices, control_indices):
        """
        特定のスキルについて、優秀群と対照群のプロファイルを比較

        Parameters:
        -----------
        skill_idx: int
            スキルのインデックス
        excellent_indices: array
            優秀群のインデックス
        control_indices: array
            対照群（マッチング済み非優秀群）のインデックス

        Returns:
        --------
        result: dict
            スキル比較結果
        """
        skill_code = self.skill_codes[skill_idx]
        skill_name = self.skill_names[skill_code]

        # 習得フラグ（スキルレベル > 0 を習得と見なす）
        has_skill_excellent = (self.skill_matrix[excellent_indices, skill_idx] > 0).astype(int)
        has_skill_control = (self.skill_matrix[control_indices, skill_idx] > 0).astype(int)

        # 習得率を計算
        n_excellent = len(excellent_indices)
        n_control = len(control_indices)

        n_skill_excellent = has_skill_excellent.sum()
        n_skill_control = has_skill_control.sum()

        p_excellent = n_skill_excellent / n_excellent if n_excellent > 0 else 0
        p_control = n_skill_control / n_control if n_control > 0 else 0

        # 重要度（差分）
        importance = p_excellent - p_control

        # Wilson信頼区間を計算
        z = norm.ppf(0.975)  # 95% CI
        ci_excellent = self._wilson_confidence_interval(
            n_skill_excellent, n_excellent, z
        )
        ci_control = self._wilson_confidence_interval(
            n_skill_control, n_control, z
        )

        # Fisher正確検定
        try:
            # 分割表を作成
            contingency_table = np.array([
                [n_skill_excellent, n_excellent - n_skill_excellent],
                [n_skill_control, n_control - n_skill_control]
            ])

            odds_ratio, p_value = fisher_exact(contingency_table)
        except (ValueError, ZeroDivisionError):
            logger.debug(f"スキル {skill_name} のFisher検定失敗")
            p_value = 1.0
            odds_ratio = 1.0

        # 統計的有意性
        alpha = DEFAULT_SIGNIFICANCE_LEVEL
        significant = p_value < alpha

        # 解釈文を生成
        interpretation = self._generate_skill_interpretation(
            skill_name,
            p_excellent,
            p_control,
            importance,
            p_value,
            significant
        )

        return {
            'skill_code': skill_code,
            'skill_name': skill_name,
            'importance': importance,
            'p_excellent': p_excellent,
            'p_control': p_control,
            'ci_excellent': ci_excellent,
            'ci_control': ci_control,
            'p_value': p_value,
            'significant': significant,
            'n_excellent': n_excellent,
            'n_skill_excellent': n_skill_excellent,
            'n_control': n_control,
            'n_skill_control': n_skill_control,
            'interpretation': interpretation
        }

    def _wilson_confidence_interval(self, successes, n, z=1.96):
        """
        Wilson スコア法による信頼区間を計算

        Parameters:
        -----------
        successes: int
            成功数
        n: int
            試行数
        z: float
            標準正規分布の分位点（デフォルト: 1.96 = 95%CI）

        Returns:
        --------
        ci: tuple
            (下限, 上限)
        """
        if n == 0:
            return (0, 0)

        p_hat = successes / n
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denominator
        margin = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denominator

        lower = max(0, center - margin)
        upper = min(1, center + margin)

        return (lower, upper)

    def _generate_skill_interpretation(self, skill_name, p_excellent, p_control,
                                       importance, p_value, significant):
        """
        スキル比較結果の解釈文を生成

        Parameters:
        -----------
        skill_name: str
            スキル名
        p_excellent: float
            優秀群での習得率
        p_control: float
            対照群での習得率
        importance: float
            重要度（差分）
        p_value: float
            Fisher検定のp値
        significant: bool
            統計的有意性フラグ

        Returns:
        --------
        interpretation: str
            解釈文
        """
        significance_text = "（統計的に有意）" if significant else ""

        if importance > 0:
            return (
                f"優秀群の {p_excellent*100:.0f}% が習得 vs "
                f"非優秀群の {p_control*100:.0f}% "
                f"（差分: +{importance*100:.1f}%）"
                f"{significance_text}"
            )
        else:
            return (
                f"優秀群の {p_excellent*100:.0f}% が習得 vs "
                f"非優秀群の {p_control*100:.0f}% "
                f"（差分: {importance*100:.1f}%）"
                f"{significance_text}"
            )

    # ==================== Layer 2: 個別メンバーへの因果効果推定（HTE） ====================

    def estimate_heterogeneous_treatment_effects(self, excellent_members, skill_profile):
        """
        Layer 2: 個別メンバーへの因果効果推定（HTE）

        各メンバーについて、各スキル習得の個別効果を推定
        「このメンバーがスキルXを習得したら...」を個別推定

        Parameters:
        -----------
        excellent_members: list
            優秀群の社員コードリスト
        skill_profile: list[dict]
            Layer 1から得られたスキルプロファイル

        Returns:
        --------
        hte_results: dict
            メンバー別のHTE推定結果
            {
                'member_id': {
                    'skills': [
                        {
                            'skill_code': str,
                            'skill_name': str,
                            'estimated_effect': float,
                            'confidence': float,
                            'reasoning': str
                        },
                        ...
                    ],
                    'top_5_skills': [...],  # TOP 5推奨スキル
                }
            }
        """
        logger.info("=== Layer 2: 個別メンバーへの因果効果推定を開始 ===")

        excellent_indices = np.array([
            self.member_to_idx[m] for m in excellent_members
            if m in self.member_to_idx
        ])

        # 各メンバーの優秀フラグ
        is_excellent = np.zeros(len(self.members))
        is_excellent[excellent_indices] = 1

        # 各スキルについて HTE を推定
        hte_matrix = np.zeros((len(self.members), len(self.skill_codes)))

        for skill_idx in range(len(self.skill_codes)):
            hte_matrix[:, skill_idx] = self._estimate_skill_specific_hte(
                skill_idx,
                is_excellent,
                excellent_indices
            )

        # 結果をメンバー別にまとめる
        hte_results = {}

        for member_idx, member_code in enumerate(self.members):
            skill_effects = []

            for skill_idx in range(len(self.skill_codes)):
                skill_code = self.skill_codes[skill_idx]
                skill_name = self.skill_names[skill_code]

                estimated_effect = hte_matrix[member_idx, skill_idx]

                # 対応するスキルプロファイルを取得
                profile_entry = next(
                    (s for s in skill_profile if s['skill_code'] == skill_code),
                    None
                )

                # 信頼度とレベル分けを生成
                confidence = self._calculate_confidence_level(
                    profile_entry,
                    estimated_effect
                )

                reasoning = self._generate_member_specific_reasoning(
                    member_idx,
                    skill_name,
                    estimated_effect,
                    profile_entry,
                    confidence
                )

                skill_effects.append({
                    'skill_code': skill_code,
                    'skill_name': skill_name,
                    'estimated_effect': float(estimated_effect),
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'profile_rank': next(
                        (i for i, s in enumerate(skill_profile) if s['skill_code'] == skill_code),
                        len(skill_profile) + 1
                    )
                })

            # 効果の大きさでソート
            skill_effects.sort(key=lambda x: abs(x['estimated_effect']), reverse=True)

            hte_results[member_code] = {
                'member_id': member_code,
                'is_excellent': bool(is_excellent[member_idx]),
                'skills': skill_effects,
                'top_5_skills': skill_effects[:5],
                'summary': self._generate_member_summary(
                    member_code,
                    skill_effects[:3],
                    is_excellent[member_idx]
                )
            }

        logger.info(f"HTE推定完了: {len(hte_results)}メンバー")

        return hte_results

    def _estimate_skill_specific_hte(self, skill_idx, is_excellent, excellent_indices):
        """
        特定のスキルについて HTE を推定

        Doubly Robust推定量を使用して、バイアスを低減する

        Parameters:
        -----------
        skill_idx: int
            スキルのインデックス
        is_excellent: array
            優秀フラグ
        excellent_indices: array
            優秀群のインデックス

        Returns:
        --------
        hte: array
            各メンバーについての推定効果
        """
        has_skill = (self.skill_matrix[:, skill_idx] > 0).astype(int)

        try:
            # 傾向スコアモデル（スキル習得の傾向）
            ps_model = LogisticRegression(
                max_iter=1000,
                random_state=DEFAULT_RANDOM_STATE
            )
            ps_model.fit(self.member_features, has_skill)
            propensity_scores = ps_model.predict_proba(self.member_features)[:, 1]

            # スキル習得者と未習得者を分離
            skill_acquirers = np.where(has_skill == 1)[0]
            non_acquirers = np.where(has_skill == 0)[0]

            # Doubly Robust推定量
            hte = np.zeros(len(self.members))

            for i in range(len(self.members)):
                ps_i = propensity_scores[i]

                # Propensity scoreが極端な値を避ける
                if ps_i < 0.01 or ps_i > 0.99:
                    hte[i] = 0
                    continue

                # スキル習得者の場合
                if has_skill[i] == 1:
                    hte[i] = is_excellent[i] / ps_i
                # スキル未習得者の場合
                else:
                    hte[i] = -is_excellent[i] / (1 - ps_i)

            return hte

        except Exception as e:
            logger.warning(f"スキル {self.skill_names.get(self.skill_codes[skill_idx], 'Unknown')} の HTE推定失敗: {e}")
            return np.zeros(len(self.members))

    def _calculate_confidence_level(self, profile_entry, estimated_effect):
        """
        推定効果の信頼度をレベル分け

        Parameters:
        -----------
        profile_entry: dict
            スキルプロファイルエントリ
        estimated_effect: float
            推定効果

        Returns:
        --------
        confidence: str
            "Low", "Medium", "High" のいずれか
        """
        if profile_entry is None:
            return "Low"

        p_value = profile_entry.get('p_value', 1.0)
        n_excellent = profile_entry.get('n_excellent', 0)

        # 統計的有意性と サンプルサイズから信頼度を判定
        if p_value < 0.01 and n_excellent >= 10:
            return "High"
        elif p_value < 0.05 and n_excellent >= 5:
            return "Medium"
        else:
            return "Low"

    def _generate_member_specific_reasoning(self, member_idx, skill_name, estimated_effect,
                                           profile_entry, confidence):
        """
        メンバー固有の根拠付き説明を生成

        Parameters:
        -----------
        member_idx: int
            メンバーのインデックス
        skill_name: str
            スキル名
        estimated_effect: float
            推定効果
        profile_entry: dict
            スキルプロファイルエントリ
        confidence: str
            信頼度

        Returns:
        --------
        reasoning: str
            説明文
        """
        if profile_entry is None:
            return f"統計データが不足しているため、{skill_name} の効果を推定できません"

        p_excellent = profile_entry.get('p_excellent', 0)
        p_control = profile_entry.get('p_control', 0)
        significant = profile_entry.get('significant', False)

        sig_text = "（統計的に有意）" if significant else ""

        effect_text = f"+{estimated_effect*100:.1f}%" if estimated_effect > 0 else f"{estimated_effect*100:.1f}%"

        reasoning = (
            f"{skill_name}習得で優秀度が {effect_text} 変化見込み\n"
            f"根拠：優秀者の {p_excellent*100:.0f}% が習得（非優秀群 {p_control*100:.0f}%）{sig_text}\n"
            f"信頼度：{confidence}"
        )

        return reasoning

    def _generate_member_summary(self, member_code, top_skills, is_excellent):
        """
        メンバーの改善提案サマリーを生成

        Parameters:
        -----------
        member_code: str
            メンバーコード
        top_skills: list[dict]
            TOP 3のスキル
        is_excellent: bool
            優秀者フラグ

        Returns:
        --------
        summary: str
            サマリーテキスト
        """
        if is_excellent:
            return f"{member_code}: 既に優秀人材です。さらなるスキル習得で組織への貢献度を向上できます。"

        if not top_skills:
            return f"{member_code}: 改善の余地があります。スキルプロファイルを確認してください。"

        top_skill_names = [s['skill_name'] for s in top_skills[:3]]
        skills_text = "、".join(top_skill_names)

        return f"{member_code}: 優先習得すべきスキルは {skills_text} です。"

    # ==================== Layer 3: 説明可能性の強化 ====================

    def generate_comprehensive_insights(self, excellent_members, skill_profile, hte_results):
        """
        Layer 3: 説明可能性の強化

        人材育成担当者向けの根拠付きスキル開発提案を生成

        Parameters:
        -----------
        excellent_members: list
            優秀群の社員コードリスト
        skill_profile: list[dict]
            Layer 1から得られたスキルプロファイル
        hte_results: dict
            Layer 2から得られたHTE推定結果

        Returns:
        --------
        insights: dict
            包括的な分析洞察
            {
                'executive_summary': str,
                'top_10_skills': list,
                'organizational_gaps': dict,
                'member_recommendations': list,
                'skill_combinations': list,
                'development_roadmap': dict
            }
        """
        logger.info("=== Layer 3: 説明可能性の強化を開始 ===")

        insights = {
            'executive_summary': self._generate_executive_summary(
                excellent_members,
                skill_profile
            ),
            'top_10_skills': skill_profile[:10],
            'organizational_gaps': self._analyze_organizational_gaps(
                skill_profile,
                hte_results
            ),
            'member_recommendations': self._generate_priority_recommendations(
                excellent_members,
                hte_results
            ),
            'skill_combinations': self._identify_skill_synergies(
                skill_profile,
                hte_results
            ),
            'development_roadmap': self._create_development_roadmap(
                excellent_members,
                hte_results
            )
        }

        logger.info("説明可能性の強化が完了しました")

        return insights

    def _generate_executive_summary(self, excellent_members, skill_profile):
        """
        エグゼクティブサマリーを生成

        Parameters:
        -----------
        excellent_members: list
            優秀群の社員コード
        skill_profile: list[dict]
            スキルプロファイル

        Returns:
        --------
        summary: str
            サマリーテキスト
        """
        n_excellent = len(excellent_members)
        n_skills_total = len(skill_profile)
        n_significant = sum(1 for s in skill_profile if s['significant'])

        top_skill = skill_profile[0] if skill_profile else None

        summary = (
            f"## 分析サマリー\n\n"
            f"優秀人材 {n_excellent}名を対象とした逆向き因果推論分析を実施しました。\n\n"
            f"**主要な発見：**\n\n"
            f"1. **優秀者の特性スキル**: {n_skills_total}個のスキル中、"
            f"{n_significant}個が統計的に有意な差異を示しています\n\n"
        )

        if top_skill:
            summary += (
                f"2. **最優先スキル**: {top_skill['skill_name']}\n"
                f"   - 優秀群での習得率: {top_skill['p_excellent']*100:.0f}%\n"
                f"   - 非優秀群での習得率: {top_skill['p_control']*100:.0f}%\n"
                f"   - 差分: +{top_skill['importance']*100:.1f}%\n"
                f"   - 有意性: {'有意 (p < 0.05)' if top_skill['significant'] else '有意でない'}\n\n"
            )

        summary += (
            f"3. **推奨アクション**:\n"
            f"   - 上位スキルを優先的に育成プログラムに組み込む\n"
            f"   - メンバー別の個別化されたスキル開発計画を策定\n"
            f"   - スキル相互作用を活用した効率的な育成\n"
        )

        return summary

    def _analyze_organizational_gaps(self, skill_profile, hte_results):
        """
        組織全体のスキルギャップを分析

        Parameters:
        -----------
        skill_profile: list[dict]
            スキルプロファイル
        hte_results: dict
            HTE推定結果

        Returns:
        --------
        gaps: dict
            スキルギャップ分析結果
        """
        gaps = {
            'critical_gaps': [],  # 習得率が低く重要なスキル
            'high_potential_skills': [],  # 習得率は低いが効果が大きいスキル
            'saturation_skills': [],  # 既に多くが習得しているスキル
        }

        for skill in skill_profile[:20]:  # TOP 20をチェック
            p_excellent = skill['p_excellent']
            p_control = skill['p_control']
            importance = skill['importance']

            # Critical gap: 優秀群での習得率が高いが、全体では低い
            if p_excellent > 0.7 and p_control < 0.3 and importance > 0.3:
                gaps['critical_gaps'].append({
                    'skill_name': skill['skill_name'],
                    'excellent_rate': p_excellent,
                    'overall_rate': p_control,
                    'gap': importance,
                    'status': '優秀者特有スキル'
                })

            # High potential: 習得効果が大きいスキル
            if importance > 0.25 and p_control < 0.5:
                gaps['high_potential_skills'].append({
                    'skill_name': skill['skill_name'],
                    'importance': importance,
                    'current_adoption': p_control,
                    'status': '組織全体での習得を推奨'
                })

            # Saturation: 既に広く習得されているスキル
            if p_control > 0.7 and p_excellent > 0.8:
                gaps['saturation_skills'].append({
                    'skill_name': skill['skill_name'],
                    'adoption_rate': p_control,
                    'status': '基本スキル'
                })

        return gaps

    def _generate_priority_recommendations(self, excellent_members, hte_results):
        """
        優先度付けメンバー改善提案を生成

        Parameters:
        -----------
        excellent_members: list
            優秀群の社員コード
        hte_results: dict
            HTE推定結果

        Returns:
        --------
        recommendations: list[dict]
            メンバー別の優先度付け提案
        """
        recommendations = []

        for member_code, hte_data in hte_results.items():
            if hte_data['is_excellent']:
                continue  # 優秀者はスキップ

            top_3_skills = hte_data['top_5_skills'][:3]

            if not top_3_skills:
                continue

            recommendation = {
                'member_id': member_code,
                'priority_skills': [
                    {
                        'rank': i + 1,
                        'skill_name': skill['skill_name'],
                        'expected_effect': skill['estimated_effect'],
                        'confidence': skill['confidence'],
                        'reasoning': skill['reasoning']
                    }
                    for i, skill in enumerate(top_3_skills)
                ],
                'estimated_improvement': sum(
                    s['estimated_effect'] for s in top_3_skills
                ),
                'summary': hte_data['summary']
            }

            recommendations.append(recommendation)

        # 改善期待値でソート
        recommendations.sort(
            key=lambda x: x['estimated_improvement'],
            reverse=True
        )

        return recommendations[:50]  # TOP 50のメンバーのみ

    def _identify_skill_synergies(self, skill_profile, hte_results):
        """
        スキル相乗効果を特定

        Parameters:
        -----------
        skill_profile: list[dict]
            スキルプロファイル
        hte_results: dict
            HTE推定結果

        Returns:
        --------
        synergies: list[dict]
            スキル相乗効果の分析結果
        """
        synergies = []

        # TOP 10スキルの組み合わせを分析
        top_10_codes = [s['skill_code'] for s in skill_profile[:10]]

        for skill_code_a, skill_code_b in combinations(top_10_codes, 2):
            skill_name_a = self.skill_names[skill_code_a]
            skill_name_b = self.skill_names[skill_code_b]

            # スキルAとBの両方を持つメンバー数
            skill_a_idx = self.skill_codes.index(skill_code_a)
            skill_b_idx = self.skill_codes.index(skill_code_b)

            has_a = (self.skill_matrix[:, skill_a_idx] > 0).astype(int)
            has_b = (self.skill_matrix[:, skill_b_idx] > 0).astype(int)
            has_both = (has_a & has_b).sum()

            if has_both < 3:  # 両方を持つメンバーが少なすぎる
                continue

            synergy = {
                'skill_combination': f"{skill_name_a} + {skill_name_b}",
                'member_count_with_both': int(has_both),
                'status': 'メンバー数が多い' if has_both >= 5 else 'レアな組み合わせ',
                'recommendation': '相乗効果の可能性があります'
            }

            synergies.append(synergy)

        return synergies[:10]  # TOP 10の組み合わせ

    def _create_development_roadmap(self, excellent_members, hte_results):
        """
        スキル開発ロードマップを作成

        Parameters:
        -----------
        excellent_members: list
            優秀群の社員コード
        hte_results: dict
            HTE推定結果

        Returns:
        --------
        roadmap: dict
            開発ロードマップ
        """
        roadmap = {
            'immediate_priority': [],  # 1ヶ月以内
            'short_term': [],  # 3ヶ月以内
            'medium_term': [],  # 6ヶ月以内
            'resources_required': self._estimate_resources(hte_results)
        }

        # 効果が大きい順にスキルを分類
        for member_code, hte_data in hte_results.items():
            if hte_data['is_excellent']:
                continue

            top_skill = hte_data['top_5_skills'][0] if hte_data['top_5_skills'] else None

            if not top_skill:
                continue

            effect = top_skill['estimated_effect']
            confidence = top_skill['confidence']

            skill_plan = {
                'member_id': member_code,
                'skill': top_skill['skill_name'],
                'expected_effect': effect,
                'confidence': confidence
            }

            if effect > 0.15 and confidence in ['High', 'Medium']:
                roadmap['immediate_priority'].append(skill_plan)
            elif effect > 0.10:
                roadmap['short_term'].append(skill_plan)
            else:
                roadmap['medium_term'].append(skill_plan)

        # 各フェーズを件数でソート
        for phase in ['immediate_priority', 'short_term', 'medium_term']:
            roadmap[phase].sort(
                key=lambda x: x['expected_effect'],
                reverse=True
            )
            roadmap[phase] = roadmap[phase][:10]  # 各フェーズMAX 10件

        return roadmap

    def _estimate_resources(self, hte_results):
        """
        開発に必要なリソースを推定

        Parameters:
        -----------
        hte_results: dict
            HTE推定結果

        Returns:
        --------
        resources: dict
            リソース見積もり
        """
        n_members_to_develop = sum(
            1 for data in hte_results.values()
            if not data['is_excellent'] and data['top_5_skills']
        )

        return {
            'estimated_members_to_develop': n_members_to_develop,
            'estimated_training_hours_per_member': 40,
            'total_estimated_hours': n_members_to_develop * 40,
            'recommended_timeline_months': 6,
            'estimated_cost_per_member': 50000,  # JPY
            'total_estimated_cost': n_members_to_develop * 50000,
        }


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
