"""基準群との記述的比較。相関を因果効果や人材評価の確率として扱わない。"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from .data import DataValidationError, SkillnoteData
from .graph import GraphEncoder


@dataclass
class AnalysisResult:
    members: pd.DataFrame
    skills: pd.DataFrame
    embeddings: np.ndarray
    metrics: dict
    training: dict


def _prototype_scores(embeddings, selected):
    return cosine_similarity(embeddings, embeddings[selected].mean(axis=0, keepdims=True)).ravel()


def analyze(
    data: SkillnoteData,
    reference_codes: list[str],
    *,
    epochs=100,
    seed=42,
    hidden_dim=32,
    neighbors=10,
) -> AnalysisResult:
    codes = data.members.member_code.tolist()
    if not reference_codes or len(set(reference_codes)) != len(reference_codes):
        raise DataValidationError("基準群を重複なく1名以上指定してください")
    if set(reference_codes) - set(codes):
        raise DataValidationError("基準群に在籍社員一覧にないメンバーコードがあります")
    reference = np.array([code in reference_codes for code in codes])
    if reference.all():
        raise DataValidationError("基準群以外の比較対象が1名以上必要です")
    observed = data.matrix.getnnz(axis=1) > 0
    if (reference & ~observed).any():
        raise DataValidationError("基準群に習得済み力量がない社員が含まれています")
    if data.matrix.shape[1] == 0:
        raise DataValidationError("力量データがありません")
    encoder = GraphEncoder(hidden_dim=hidden_dim, neighbors=neighbors, seed=seed).fit(
        data.matrix, epochs=epochs
    )
    similarities = np.clip(_prototype_scores(encoder.embeddings, reference), -1, 1)
    similarities[~observed] = np.nan
    members = data.members[["member_code", "name"]].copy()
    members["is_reference"] = reference
    members["acquired_count"] = data.matrix.getnnz(axis=1)
    members["similarity"] = similarities
    members["status"] = np.where(observed, "scored", "no_acquisitions")
    # 非記録を未習得と同一視した比較であることはレポートへ明記する。
    present = data.matrix > 0
    n_ref = int(reference.sum())
    n_other = int((~reference).sum())
    ref_counts = np.asarray(present[reference].sum(axis=0)).ravel().astype(int)
    other_counts = np.asarray(present[~reference].sum(axis=0)).ravel().astype(int)
    records = []
    for j, row in data.competences.iterrows():
        a, b = int(ref_counts[j]), int(other_counts[j])
        p = fisher_exact([[a, n_ref - a], [b, n_other - b]]).pvalue
        ci_ref = proportion_confint(a, n_ref, method="wilson")
        ci_other = proportion_confint(b, n_other, method="wilson")
        records.append(
            {
                **row.to_dict(),
                "reference_holders": a,
                "other_holders": b,
                "reference_n": n_ref,
                "other_n": n_other,
                "reference_rate": a / n_ref,
                "other_rate": b / n_other,
                "rate_difference": a / n_ref - b / n_other,
                "reference_ci_low": ci_ref[0],
                "reference_ci_high": ci_ref[1],
                "other_ci_low": ci_other[0],
                "other_ci_high": ci_other[1],
                "p_value": float(p),
            }
        )
    skills = pd.DataFrame(records)
    skills["q_value"] = multipletests(skills.p_value, method="fdr_bh")[1]
    skills["significant"] = skills.q_value < 0.05
    skills = skills.sort_values(["rate_difference", "competence_id"], ascending=[False, True])
    metrics = _evaluate(data.matrix, encoder.embeddings, reference, observed, seed)
    return AnalysisResult(
        members.sort_values("similarity", ascending=False, na_position="last"),
        skills,
        encoder.embeddings,
        metrics,
        {
            "status": encoder.training_status,
            "epochs_completed": len(encoder.losses),
            "losses": encoder.losses,
            "graph_edges": int(encoder.adjacency.nnz // 2),
            "seed": seed,
            "hidden_dim": hidden_dim,
            "neighbors": neighbors,
        },
    )


def _evaluate(matrix, embeddings, reference, observed, seed):
    indices = np.flatnonzero(observed)
    y = reference[indices].astype(int)
    counts = np.bincount(y, minlength=2)
    result = {
        "evaluation": "stratified_out_of_fold_transductive",
        "models": {},
        "excluded_no_acquisitions": int((~observed).sum()),
    }
    if counts.min() < 2:
        result["reason"] = "習得記録のある基準群・比較群がそれぞれ2名以上必要です"
        return result
    splits = int(min(5, counts.min()))
    graph_scores = np.zeros(len(indices))
    baseline_scores = np.zeros(len(indices))
    for train, test in StratifiedKFold(splits, shuffle=True, random_state=seed).split(indices, y):
        selected = indices[train[y[train] == 1]]
        graph_scores[test] = cosine_similarity(
            embeddings[indices[test]], embeddings[selected].mean(axis=0, keepdims=True)
        ).ravel()
        baseline = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        baseline.fit(matrix[indices[train]], y[train])
        baseline_scores[test] = baseline.predict_proba(matrix[indices[test]])[:, 1]
    for name, scores in [("graph_prototype", graph_scores), ("logistic_baseline", baseline_scores)]:
        result["models"][name] = {
            "roc_auc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores)),
            "samples": len(y),
            "folds": splits,
        }
    return result
