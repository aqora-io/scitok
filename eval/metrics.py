"""Ranking metrics from a score matrix (queries × docs)."""

import numpy as np


def rankings(scores, depth=100):
    top = np.argpartition(-scores, min(depth, scores.shape[1] - 1), axis=1)[:, :depth]
    order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(top, order, axis=1)


def known_item(scores, targets, doc_index):
    """targets: doc id per query. MRR@10, success@1/@10/@100."""
    ranks = rankings(scores, 100)
    t = np.array([doc_index[x] for x in targets])
    hit = ranks == t[:, None]
    pos = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, 10**9)
    return {
        "MRR@10": float(np.mean(np.where(pos <= 10, 1.0 / pos, 0.0))),
        "S@1": float(np.mean(pos <= 1)),
        "S@10": float(np.mean(pos <= 10)),
        "S@100": float(np.mean(pos <= 100)),
    }


def graded(scores, qrels, doc_ids):
    """qrels: list (per query) of {doc_id: grade}. nDCG@10, MRR@10 (grade ≥ 1), R@100 (grade ≥ 1)."""
    ranks = rankings(scores, 100)
    ndcg, mrr, rec = [], [], []
    for row, rel in zip(ranks, qrels):
        gains = np.array([rel.get(doc_ids[d], 0) for d in row], dtype=float)
        disc = 1.0 / np.log2(np.arange(2, 12))
        dcg = float(((2**gains[:10] - 1) * disc).sum())
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = float(((2 ** np.array(ideal, dtype=float) - 1) * disc[: len(ideal)]).sum())
        ndcg.append(dcg / idcg if idcg else 0.0)
        first = np.flatnonzero(gains[:10] >= 1)
        mrr.append(1.0 / (first[0] + 1) if len(first) else 0.0)
        n_rel = sum(1 for g in rel.values() if g >= 1)
        rec.append(float((gains >= 1).sum()) / n_rel if n_rel else 0.0)
    return {"nDCG@10": float(np.mean(ndcg)), "MRR@10": float(np.mean(mrr)), "R@100": float(np.mean(rec))}


def per_query_ndcg(scores, qrels, doc_ids, k=10):
    ranks = rankings(scores, k)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    out = []
    for row, rel in zip(ranks, qrels):
        gains = np.array([rel.get(doc_ids[d], 0) for d in row], dtype=float)
        ideal = np.array(sorted(rel.values(), reverse=True)[:k], dtype=float)
        idcg = ((2**ideal - 1) * disc[: len(ideal)]).sum()
        out.append(((2**gains - 1) * disc).sum() / idcg if idcg else 0.0)
    return np.array(out)


def paired_bootstrap(a, b, n=10000, seed=0):
    """Mean difference a - b with a 95% CI and the share of resamples where a <= b."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float((means <= 0).mean())
