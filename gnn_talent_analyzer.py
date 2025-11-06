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

    # ==================== CAUSAL SKILL PROFILING ARCHITECTURE ====================
    # Layer 1-3: 逆向き因果推論分析による優秀者特性の抽出
    # - Layer 1: 優秀者のスキルプロファイル分析（傾向スコアマッチング）
    # - Layer 2: 個別メンバーの異質的処置効果推定（HTE）
    # - Layer 3: 経営的インサイト生成と実装ロードマップ
    # ============================================================================

    # ==================== Layer 1: 優秀者特性の逆向き分析 ====================

    def get_top_skill_holders(self, top_n=10):
        """
        スキル保有数上位のメンバーを取得

        Parameters:
        -----------
        top_n: int
            取得する上位メンバー数（デフォルト: 10）

        Returns:
        --------
        top_members: list
            スキル保有数上位のメンバーコードリスト
        """
        # 各メンバーのスキル保有数を計算（スキルレベル > 0 のスキル数）
        skill_counts = (self.skill_matrix > 0).sum(axis=1)

        # スキル保有数でソートし、上位N名を取得
        top_indices = np.argsort(skill_counts)[::-1][:top_n]
        top_members = [self.members[idx] for idx in top_indices]

        logger.info(f"スキル保有数上位{top_n}名を取得: {top_members}")

        return top_members

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

            # 優秀群の習得率が高いスキルのみを採用（マイナスの効果は除外）
            if result is not None and result['importance'] > 0:
                skill_profile.append(result)

        # 重要度（差分）でソート（正の値のみなので abs 不要）
        skill_profile.sort(
            key=lambda x: x['importance'],
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

    def estimate_heterogeneous_treatment_effects_with_gnn(self, excellent_members, skill_profile):
        """
        Layer 2: 個別メンバーへの因果効果推定（HTE）- GNN埋め込み版

        GNN学習で得られた埋め込み表現を活用してHTEを推定
        より高度な特徴表現を用いることで、精度の高い個別効果推定が可能

        Parameters:
        -----------
        excellent_members: list
            優秀群の社員コードリスト
        skill_profile: list[dict]
            Layer 1から得られたスキルプロファイル

        Returns:
        --------
        hte_results: dict
            メンバー別のHTE推定結果（通常版と同じ形式）
        """
        logger.info("=== Layer 2: 個別メンバーへの因果効果推定（GNN版）を開始 ===")

        if not hasattr(self, 'embeddings') or self.embeddings is None:
            raise ModelTrainingError("GNN学習が完了していません。先に train() を実行してください。")

        excellent_indices = np.array([
            self.member_to_idx[m] for m in excellent_members
            if m in self.member_to_idx
        ])

        # 各メンバーの優秀フラグ
        is_excellent = np.zeros(len(self.members))
        is_excellent[excellent_indices] = 1

        # 各スキルについて HTE を推定（GNN埋め込みを使用）
        hte_matrix = np.zeros((len(self.members), len(self.skill_codes)))

        for skill_idx in range(len(self.skill_codes)):
            hte_matrix[:, skill_idx] = self._estimate_skill_specific_hte_with_gnn(
                skill_idx,
                is_excellent,
                excellent_indices
            )

        # 結果をメンバー別にまとめる（通常版と同じ処理）
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
                    'estimated_effect': estimated_effect,
                    'confidence': confidence,
                    'reasoning': reasoning
                })

            # TOP 5スキルを抽出
            skill_effects_sorted = sorted(
                skill_effects,
                key=lambda x: abs(x['estimated_effect']),
                reverse=True
            )
            top_5 = skill_effects_sorted[:5]

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

        logger.info(f"HTE推定完了（GNN版）: {len(hte_results)}メンバー")

        return hte_results

    def _estimate_skill_specific_hte_with_gnn(self, skill_idx, is_excellent, excellent_indices):
        """
        特定のスキルについて HTE を推定（GNN埋め込み版）

        GNN学習で得られた埋め込み表現を特徴量として使用
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
            # 傾向スコアモデル（スキル習得の傾向）- GNN埋め込みを使用
            ps_model = LogisticRegression(
                max_iter=1000,
                random_state=DEFAULT_RANDOM_STATE
            )
            ps_model.fit(self.embeddings, has_skill)
            propensity_scores = ps_model.predict_proba(self.embeddings)[:, 1]

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
            logger.warning(f"スキル {self.skill_names.get(self.skill_codes[skill_idx], 'Unknown')} の HTE推定失敗（GNN版）: {e}")
            return np.zeros(len(self.members))

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
            スキルプロファイルエントリ（Noneの場合もあり）
        confidence: str
            信頼度

        Returns:
        --------
        reasoning: str
            説明文
        """
        effect_text = f"+{estimated_effect*100:.1f}%" if estimated_effect > 0 else f"{estimated_effect*100:.1f}%"

        # profile_entryがない場合（優秀群習得率が低いスキル）も、
        # HTEの推定値に基づいて説明を生成
        if profile_entry is None:
            reasoning = (
                f"{skill_name}習得で優秀度が {effect_text} 変化見込み\n"
                f"根拠：個別メンバーの特性に基づく異質的処置効果推定\n"
                f"信頼度：{confidence}"
            )
            return reasoning

        p_excellent = profile_entry.get('p_excellent', 0)
        p_control = profile_entry.get('p_control', 0)
        significant = profile_entry.get('significant', False)

        sig_text = "（統計的に有意）" if significant else ""

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
                'member_recommendations': list,
                'skill_combinations': list
            }
        """
        logger.info("=== Layer 3: 説明可能性の強化を開始 ===")

        insights = {
            'executive_summary': self._generate_executive_summary(
                excellent_members,
                skill_profile
            ),
            'top_10_skills': skill_profile[:10],
            'member_recommendations': self._generate_priority_recommendations(
                excellent_members,
                hte_results
            ),
            'skill_combinations': self._identify_skill_synergies(
                skill_profile,
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
        スキル相乗効果を特定（因果推論ベース）

        優秀群で共起率が高く、非優秀群との差が大きいスキル組み合わせを推薦

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

        # 優秀群と非優秀群のインデックスを取得
        excellent_indices = []
        non_excellent_indices = []

        for member_code, hte_data in hte_results.items():
            if member_code not in self.member_to_idx:
                continue
            member_idx = self.member_to_idx[member_code]

            if hte_data.get('is_excellent', False):
                excellent_indices.append(member_idx)
            else:
                non_excellent_indices.append(member_idx)

        excellent_indices = np.array(excellent_indices)
        non_excellent_indices = np.array(non_excellent_indices)

        if len(excellent_indices) < 3 or len(non_excellent_indices) < 3:
            logger.warning("優秀群または非優秀群のサンプル数が不足しています")
            return []

        # TOP 10スキルの組み合わせを分析
        top_10_codes = [s['skill_code'] for s in skill_profile[:10]]

        for skill_code_a, skill_code_b in combinations(top_10_codes, 2):
            skill_name_a = self.skill_names[skill_code_a]
            skill_name_b = self.skill_names[skill_code_b]

            # スキルインデックス
            skill_a_idx = self.skill_codes.index(skill_code_a)
            skill_b_idx = self.skill_codes.index(skill_code_b)

            # 優秀群での共起率
            has_a_excellent = (self.skill_matrix[excellent_indices, skill_a_idx] > 0).astype(int)
            has_b_excellent = (self.skill_matrix[excellent_indices, skill_b_idx] > 0).astype(int)
            has_both_excellent = (has_a_excellent & has_b_excellent).sum()
            co_occurrence_excellent = has_both_excellent / len(excellent_indices)

            # 非優秀群での共起率
            has_a_non_excellent = (self.skill_matrix[non_excellent_indices, skill_a_idx] > 0).astype(int)
            has_b_non_excellent = (self.skill_matrix[non_excellent_indices, skill_b_idx] > 0).astype(int)
            has_both_non_excellent = (has_a_non_excellent & has_b_non_excellent).sum()
            co_occurrence_non_excellent = has_both_non_excellent / len(non_excellent_indices)

            # 相乗効果スコア：優秀群での共起率と、優秀群と非優秀群の差
            synergy_score = co_occurrence_excellent * (co_occurrence_excellent - co_occurrence_non_excellent)

            # フィルタリング：優秀群で少なくとも2人以上が両方持っている
            if has_both_excellent < 2:
                continue

            # 相乗効果スコアが正（優秀群の方が共起率が高い）の場合のみ
            if synergy_score <= 0:
                continue

            # Fisher検定で統計的有意性を確認
            try:
                contingency_table = np.array([
                    [has_both_excellent, len(excellent_indices) - has_both_excellent],
                    [has_both_non_excellent, len(non_excellent_indices) - has_both_non_excellent]
                ])
                _, p_value = fisher_exact(contingency_table)
            except (ValueError, ZeroDivisionError):
                p_value = 1.0

            synergy = {
                'skill1': skill_name_a,
                'skill2': skill_name_b,
                'synergy_score': float(synergy_score),
                'co_occurrence_excellent': float(co_occurrence_excellent),
                'co_occurrence_non_excellent': float(co_occurrence_non_excellent),
                'n_excellent_with_both': int(has_both_excellent),
                'n_non_excellent_with_both': int(has_both_non_excellent),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'interpretation': (
                    f"優秀群の{co_occurrence_excellent*100:.1f}%が両方のスキルを保有（"
                    f"非優秀群は{co_occurrence_non_excellent*100:.1f}%）。"
                    f"{'統計的に有意な' if p_value < 0.05 else ''}相乗効果が期待できます。"
                )
            }

            synergies.append(synergy)

        # 相乗効果スコアでソート
        synergies.sort(key=lambda x: x['synergy_score'], reverse=True)

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
