import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix


def dataset():
    from talent_graph.data import prepare_data

    rng = np.random.default_rng(12)
    matrix = rng.integers(0, 4, (24, 12))
    matrix[:6, :4] = 3
    matrix[6:, :4] = 0
    members = pd.DataFrame(
        {"member_code": [f"{i:03}" for i in range(24)], "name": [f"社員{i}" for i in range(24)]}
    )
    records = [
        {
            "member_code": f"{i:03}",
            "competence_type": "SKILL",
            "competence_code": f"{j:03}",
            "competence_name": f"技能{j}",
            "level": str(v),
        }
        for i, row in enumerate(matrix)
        for j, v in enumerate(row)
    ]
    return prepare_data(members, pd.DataFrame(records))


def test_graph_uses_reproducible_real_gradient():
    from talent_graph.graph import GraphEncoder

    x = dataset().matrix
    a = GraphEncoder(hidden_dim=8, neighbors=3, seed=7).fit(x, epochs=50)
    b = GraphEncoder(hidden_dim=8, neighbors=3, seed=7).fit(x, epochs=50)
    assert a.losses[-1] < a.losses[0] - 0.01
    np.testing.assert_allclose(a.embeddings, b.embeddings)
    assert np.isfinite(a.embeddings).all()
    assert a.adjacency.nnz < 24 * 24


def test_zero_graph_and_small_graph_do_not_crash():
    from talent_graph.graph import GraphEncoder

    for x in [csr_matrix((4, 2)), csr_matrix(np.ones((1, 2))), csr_matrix(np.ones((4, 2)))]:
        g = GraphEncoder(hidden_dim=4, neighbors=2).fit(x, epochs=3)
        assert g.embeddings.shape == (x.shape[0], 4)
        assert np.isfinite(g.embeddings).all()


def test_analysis_exports_all_skills_with_fdr_and_oof_metrics():
    from talent_graph.analysis import analyze

    result = analyze(dataset(), [f"{i:03}" for i in range(6)], epochs=20, seed=3)
    assert len(result.skills) == 12
    assert result.skills.q_value.between(0, 1).all()
    assert (result.skills.q_value >= result.skills.p_value - 1e-10).all()
    assert result.members.similarity.between(-1, 1).all()
    assert set(result.metrics["models"]) == {"graph_prototype", "logistic_baseline"}
    assert result.metrics["evaluation"] == "stratified_out_of_fold_transductive"
    assert result.metrics["models"]["graph_prototype"]["samples"] == 24


@pytest.mark.parametrize(
    "selected", [[], ["missing"], ["000", "000"], [f"{i:03}" for i in range(24)]]
)
def test_bad_reference_group_is_rejected(selected):
    from talent_graph.analysis import analyze

    with pytest.raises(ValueError):
        analyze(dataset(), selected, epochs=2)


def test_members_without_skills_are_not_scored():
    from talent_graph.analysis import analyze

    data = dataset()
    data.matrix = data.matrix.tolil()
    data.matrix[-1, :] = 0
    data.matrix = data.matrix.tocsr()
    data.matrix.eliminate_zeros()
    result = analyze(data, ["000", "001", "002"], epochs=3)
    assert pd.isna(result.members.set_index("member_code").loc["023", "similarity"])


def test_single_reference_reports_unavailable_evaluation():
    from talent_graph.analysis import analyze

    r = analyze(dataset(), ["000"], epochs=3)
    assert r.metrics["models"] == {}
    assert r.metrics["reason"]


def test_negative_edges_are_not_biased_towards_early_csv_rows():
    from talent_graph.graph import _negative_edges

    graph = csr_matrix((100, 100))
    pairs = np.asarray(_negative_edges(graph, 100, np.random.default_rng(42)))
    # 一様な社員インデックスの平均は49.5。辞書順切り詰めだと約30になる。
    assert 40 < pairs.mean() < 60
    assert len(set(map(tuple, pairs))) == 100
    assert (pairs[:, 0] != pairs[:, 1]).all()
