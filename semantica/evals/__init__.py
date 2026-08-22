"""
Semantica Evals Module

Retrieval evaluation primitives. Currently home to page-aggregated IR metrics
(recall@k / nDCG@k / citation hit rate) for document corpora; additional
evaluation harnesses may land here over time.
"""

from .ir_metrics import (
    citation_hit_rate,
    derive_chunk_truth,
    mean,
    ndcg_at_k,
    pages_of_retrieved,
    recall_at_k,
)

__version__ = "0.1.1"
__status__ = "stable"
__all__ = [
    "citation_hit_rate",
    "derive_chunk_truth",
    "mean",
    "ndcg_at_k",
    "pages_of_retrieved",
    "recall_at_k",
]
