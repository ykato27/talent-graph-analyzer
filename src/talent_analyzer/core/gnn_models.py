"""
GNNモデル実装モジュール

Graph Neural Networkモデルの実装を提供します。
CPU最適化版のGraphSAGEベースハイブリッドGNNです。
"""

import numpy as np
import time
from tqdm import tqdm
from talent_analyzer.utils import get_logger, format_time
from talent_analyzer.config import ModelConfig, TrainingConfig, NumericalConfig
from talent_analyzer.config.loader import get_config


logger = get_logger(__name__)


class SimpleGNN:
    """
    軽量GNN実装（CPU最適化版）
    Graph Convolutional NetworkとGraphSAGEのハイブリッド

    Attributes:
    -----------
    n_layers: int
        グラフ畳み込み層数
    hidden_dim: int
        隠れ層の次元数
    dropout: float
        ドロップアウト率
    learning_rate: float
        学習率
    k_neighbors: int
        グラフ構築時の近傍ノード数
    weights: list
        モデルの重みパラメータ
    trained: bool
        モデルが訓練済みかどうか
    last_training_time: float
        最後の学習にかかった時間（秒）
    """

    def __init__(self, n_layers=None, hidden_dim=None, dropout=None, learning_rate=None):
        """
        GNNモデルの初期化

        Parameters:
        -----------
        n_layers: int, optional
            グラフ畳み込み層数
        hidden_dim: int, optional
            隠れ層の次元数
        dropout: float, optional
            ドロップアウト率
        learning_rate: float, optional
            学習率
        """
        # 設定ファイルから読み込む、引数があればそれを優先
        self.n_layers = n_layers or ModelConfig.get_n_layers()
        self.hidden_dim = hidden_dim or ModelConfig.get_hidden_dim()
        self.dropout = dropout or ModelConfig.get_dropout()
        self.learning_rate = learning_rate or ModelConfig.get_learning_rate()
        self.k_neighbors = ModelConfig.get_k_neighbors()

        self.weights = []
        self.trained = False
        self.last_training_time = None

        logger.debug(f"GNN initialized: layers={self.n_layers}, hidden={self.hidden_dim}, lr={self.learning_rate}")

    def build_graph(self, member_features, skill_matrix, member_attrs=None):
        """
        異種グラフの構築

        Parameters:
        -----------
        member_features: array-like, shape (n_members, n_features)
            社員の基本特徴量
        skill_matrix: array-like, shape (n_members, n_skills)
            スキル保有マトリクス
        member_attrs: dict, optional
            社員の属性情報（等級、役職など）

        Returns:
        --------
        adjacency_matrix: ndarray
            グラフの隣接行列
        """
        n_members = skill_matrix.shape[0]
        epsilon = NumericalConfig.get_epsilon()

        # 社員間の類似度行列を計算（スキルの類似性）
        member_similarity = np.dot(skill_matrix, skill_matrix.T)
        norm_product = (
            np.linalg.norm(skill_matrix, axis=1, keepdims=True) @
            np.linalg.norm(skill_matrix, axis=1, keepdims=True).T
        )
        member_similarity = member_similarity / (norm_product + epsilon)

        # スパース化（上位k個の接続のみ保持）
        k_neighbors = min(self.k_neighbors, n_members - 1)
        for i in range(n_members):
            threshold = np.partition(member_similarity[i], -k_neighbors)[-k_neighbors]
            member_similarity[i][member_similarity[i] < threshold] = 0

        self.adjacency = member_similarity
        self.skill_matrix = skill_matrix
        self.member_features = member_features

        logger.debug(f"Graph built: {n_members} nodes, sparsity={1-np.count_nonzero(member_similarity)/(n_members**2):.2%}")
        return member_similarity

    def aggregate_neighbors(self, features, adjacency, layer_idx):
        """
        近傍ノードの特徴量を集約（GraphSAGE風）

        Parameters:
        -----------
        features: ndarray
            ノードの特徴量
        adjacency: ndarray
            隣接行列
        layer_idx: int
            層インデックス

        Returns:
        --------
        activated: ndarray
            活性化後の特徴量
        """
        epsilon = NumericalConfig.get_epsilon()

        # 次数で正規化
        degree = np.sum(adjacency, axis=1, keepdims=True) + epsilon
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
        順伝播（フォワードパス）

        Parameters:
        -----------
        adjacency: ndarray
            隣接行列
        features: ndarray
            ノード特徴量

        Returns:
        --------
        h: ndarray
            出力埋め込み表現
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
        adjacency: ndarray
            隣接行列
        features: ndarray
            ノードの特徴量
        epochs: int, optional
            学習エポック数
        on_epoch_callback: callable, optional
            各エポック終了時に呼び出されるコールバック関数

        Returns:
        --------
        self: SimpleGNN
            自身のインスタンス
        """
        if epochs is None:
            epochs = TrainingConfig.get_epochs()

        logger.info("Starting unsupervised pretraining...")
        self.training = True
        n_nodes = features.shape[0]
        epsilon = NumericalConfig.get_epsilon()

        # 設定から値を取得
        patience = TrainingConfig.get_early_stopping_patience()
        batch_size_edges = TrainingConfig.get_batch_size_edges()

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
            if n_pos == 0:
                logger.warning("No positive edges found in graph")
                break

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
            loss = (-np.mean(np.log(1 / (1 + np.exp(-pos_scores)) + epsilon)) -
                    np.mean(np.log(1 - 1 / (1 + np.exp(-neg_scores)) + epsilon)))

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

                new_loss = (-np.mean(np.log(1 / (1 + np.exp(-pos_scores_new)) + epsilon)) -
                            np.mean(np.log(1 - 1 / (1 + np.exp(-neg_scores_new)) + epsilon)))

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
                    '経過': format_time(elapsed_time),
                    '推定残り': format_time(estimated_remaining_time),
                    '推定合計': format_time(estimated_total_time)
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
                    logger.warning(f"Callback error: {str(e)}")

            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        pbar.close()

        # 学習完了メッセージ
        total_training_time = time.time() - training_start_time
        self.last_training_time = total_training_time
        logger.info(f"Pretraining completed - Total time: {format_time(total_training_time)}")

        self.trained = True
        return self

    def get_embeddings(self, adjacency, features):
        """
        学習済みモデルからノードの埋め込み表現を取得

        Parameters:
        -----------
        adjacency: ndarray
            グラフの隣接行列
        features: ndarray
            ノードの特徴量

        Returns:
        --------
        embeddings: ndarray
            ノードの埋め込み表現（特徴量ベクトル）
        """
        self.training = False
        return self.forward(adjacency, features)
