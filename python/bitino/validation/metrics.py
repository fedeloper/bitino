"""Numerical comparison metrics used by every validation level."""
from __future__ import annotations

import numpy as np


def compare_outputs(reference: np.ndarray, candidate: np.ndarray, labels: np.ndarray | None = None, k: int = 5) -> dict:
    """Compares [N, C] outputs row-wise. Returns JSON-serialisable metrics."""
    ref = np.asarray(reference, np.float64).reshape(len(reference), -1)
    cand = np.asarray(candidate, np.float64).reshape(len(candidate), -1)
    assert ref.shape == cand.shape, (ref.shape, cand.shape)
    diff = np.abs(ref - cand)
    denom = np.maximum(np.abs(ref), 1e-6)
    cos = np.sum(ref * cand, axis=1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(cand, axis=1) + 1e-30)
    top1_r, top1_c = ref.argmax(1), cand.argmax(1)
    topk_r = np.argsort(-ref, axis=1)[:, :k]
    topk_c = np.argsort(-cand, axis=1)[:, :k]
    out = {
        "n": int(len(ref)),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "max_rel_error": float((diff / denom).max()),
        "cosine_min": float(cos.min()),
        "cosine_mean": float(cos.mean()),
        "top1_agreement": float(np.mean(top1_r == top1_c)),
        f"top{k}_set_agreement": float(np.mean([set(a) == set(b) for a, b in zip(topk_r, topk_c)])),
        "n_top1_disagreements": int(np.sum(top1_r != top1_c)),
        "exact_match": bool(np.array_equal(ref, cand)),
    }
    if labels is not None:
        labels = np.asarray(labels)
        out["reference_top1_accuracy"] = float(np.mean(top1_r == labels))
        out["candidate_top1_accuracy"] = float(np.mean(top1_c == labels))
        out["reference_top5_accuracy"] = float(np.mean([labels[i] in topk_r[i] for i in range(len(ref))]))
        out["candidate_top5_accuracy"] = float(np.mean([labels[i] in topk_c[i] for i in range(len(ref))]))
        out["disagreements_changing_correctness"] = int(np.sum((top1_r == labels) != (top1_c == labels)))
    return out
