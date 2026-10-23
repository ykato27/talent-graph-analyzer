"""疎な社員グラフとNumPyによる2層GraphSAGE。

リンク再構成の二値交差エントロピーを解析的に逆伝播する。
ラベルを学習に使用しない。全社員を含むtransductiveな埋め込み。
"""

import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def _negative_edges(graph, count, rng):
    n = graph.shape[0]
    negative_pairs = set()
    # 全結合など非エッジがない場合も必ず有限回で終了する。
    for _ in range(30):
        if len(negative_pairs) >= count:
            break
        u = rng.integers(n, size=max(32, count * 2))
        v = rng.integers(n, size=max(32, count * 2))
        valid = (u != v) & (np.asarray(graph[u, v]).ravel() == 0)
        negative_pairs.update((min(i, j), max(i, j)) for i, j in zip(u[valid], v[valid]))
    candidates = sorted(negative_pairs)
    selected = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
    negatives = [candidates[i] for i in selected]
    return negatives


class GraphEncoder:
    def __init__(self, hidden_dim=32, neighbors=10, seed=42, learning_rate=0.01):
        if hidden_dim < 1 or neighbors < 1 or learning_rate <= 0:
            raise ValueError("次元・近傍数・学習率は正の値が必要です")
        self.hidden_dim = hidden_dim
        self.neighbors = neighbors
        self.seed = seed
        self.learning_rate = learning_rate

    def fit(self, matrix, *, epochs=100):
        if epochs < 1:
            raise ValueError("epochsは1以上が必要です")
        x = normalize(sparse.csr_matrix(matrix, dtype=float), axis=1)
        n, p = x.shape
        if n < 1 or p < 1:
            raise ValueError("社員と力量がそれぞれ1件以上必要です")
        rng = np.random.default_rng(self.seed)
        k = min(self.neighbors + 1, n)
        distances, indices = (
            NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(x).kneighbors(x)
        )
        rows = np.repeat(np.arange(n), k)
        cols = indices.ravel()
        values = np.clip(1 - distances.ravel(), 0, 1)
        keep = (rows != cols) & (values > 1e-10)
        graph = sparse.csr_matrix((values[keep], (rows[keep], cols[keep])), shape=(n, n))
        graph = graph.maximum(graph.T)
        self.adjacency = graph
        a = normalize(graph + sparse.eye(n), axis=1, norm="l1").tocsr()
        c1 = sparse.hstack([x, a @ x], format="csr")
        w0 = rng.normal(0, 1 / np.sqrt(max(1, 2 * p)), (2 * p, self.hidden_dim))
        w1 = rng.normal(0, 1 / np.sqrt(2 * self.hidden_dim), (2 * self.hidden_dim, self.hidden_dim))
        positives = sparse.triu(graph, k=1).tocoo()
        # 固定したペア集合で損失を比較。大きいグラフでも最大2048正例まで。
        count = min(2048, positives.nnz)
        chosen = rng.choice(positives.nnz, size=count, replace=False)
        pu, pv = positives.row[chosen], positives.col[chosen]
        negatives = _negative_edges(graph, count, rng)
        self.losses = []
        self.training_status = "trained" if count and negatives else "no_contrastive_pairs"
        if self.training_status == "trained":
            nu, nv = np.asarray(negatives).T
            u = np.concatenate([pu, nu])
            v = np.concatenate([pv, nv])
            y = np.concatenate([np.ones(count), np.zeros(len(nu))])
            first = [np.zeros_like(w0), np.zeros_like(w1)]
            second = [np.zeros_like(w0), np.zeros_like(w1)]
            scale = np.sqrt(self.hidden_dim)
            for step in range(1, epochs + 1):
                h = np.tanh(c1 @ w0)
                c2 = np.hstack([h, a @ h])
                z = np.tanh(c2 @ w1)
                logits = np.einsum("ij,ij->i", z[u], z[v]) / scale
                loss = np.mean(np.logaddexp(0, logits) - y * logits)
                self.losses.append(float(loss))
                dlogit = (expit(logits) - y) / len(y) / scale
                dz = np.zeros_like(z)
                np.add.at(dz, u, dlogit[:, None] * z[v])
                np.add.at(dz, v, dlogit[:, None] * z[u])
                delta2 = dz * (1 - z * z)
                grad1 = c2.T @ delta2
                dc2 = delta2 @ w1.T
                dh = dc2[:, : self.hidden_dim] + a.T @ dc2[:, self.hidden_dim :]
                grad0 = c1.T @ (dh * (1 - h * h))
                for j, (w, grad) in enumerate(zip((w0, w1), (grad0, grad1))):
                    first[j] = 0.9 * first[j] + 0.1 * grad
                    second[j] = 0.999 * second[j] + 0.001 * grad**2
                    w -= (
                        self.learning_rate
                        * (first[j] / (1 - 0.9**step))
                        / (np.sqrt(second[j] / (1 - 0.999**step)) + 1e-8)
                    )
        h = np.tanh(c1 @ w0)
        self.embeddings = np.tanh(np.hstack([h, a @ h]) @ w1)
        self.weights = (w0, w1)
        return self
