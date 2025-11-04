"""
GNNベース優秀人材分析システム
半教師あり学習 + Few-shot学習対応
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
import warnings
from config_loader import get_config

warnings.filterwarnings('ignore')


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
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.trained = True
        print("事前学習完了")
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
        self.gnn.fit_unsupervised(adjacency, combined_features, epochs=epochs_unsupervised)

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
        print("\n分析実行中...")

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
                'importance_score': diff * (1 + excellent_level * 0.1)  # 保有率差 × レベル補正
            })

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

        print("分析完了")
        return results


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
