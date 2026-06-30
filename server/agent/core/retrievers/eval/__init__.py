"""Retrieval-quality evaluation harness for the RAG quality phase (YAZ-15).

This package is a benchmark-driven sanity check for the retrieval stages. It builds a small, labelled
corpus on top of the real pgvector store, runs the retrievers under successive configurations
(baseline → +metadata filter → +hybrid → +rerank → +MMR) and measures Precision@K / Recall@K / MRR
before and after. The accompanying test asserts each stage improves its targeted metric, and prints a
before/after table so the lift is reviewable at a glance.

It is deliberately small and deterministic (hand-crafted unit-vector semantics) rather than a
production-grade benchmark – the goal is a living, CI-checked demonstration that each stage pulls in
the intended direction, not a statistically rigorous offline evaluation.
"""
