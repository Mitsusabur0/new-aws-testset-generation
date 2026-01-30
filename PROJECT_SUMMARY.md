# Project Summary and Improvement Ideas

## Objectives

This repository implements a small, end-to-end pipeline for offline evaluation of a RAG system in a Chilean banking/real-estate context, built around AWS Bedrock:

- Generate a synthetic test set by reading Markdown knowledge-base chunks and prompting a Bedrock LLM to create realistic user queries in multiple Chilean Spanish styles (`1_testset_generator.py`).
- Retrieve top-K contexts from a Bedrock Knowledge Base for each synthetic query to form an evaluation set (`2_retriever.py`).
- Score retrieval quality with simple hit-rate, MRR, precision@K, and recall@K metrics, then store results in Parquet (`3_evaluator.py`).
- Provide a Streamlit dashboard for filtering and inspecting metrics and per-query retrieval behavior (`app.py`).

## Potential Improvements

- **Config centralization**: Move all configuration (paths, AWS profiles/regions, KB ID, model ID, K/TOP_K) into a single config file or environment variables to avoid drift between scripts.
- **Reproducibility**: Add deterministic seeding for `random` (and record the seed in the output) to make testset generation repeatable.
- **Encoding cleanup**: Fix mojibake (e.g., `ó`, `¿`) by ensuring UTF-8 source encoding and normalizing prompt strings; add `# -*- coding: utf-8 -*-` if needed.
- **Robust parsing**: Improve XML extraction with stricter validation and fallback prompts; log raw responses when parsing fails.
- **Cost reporting**: Use the defined pricing constants to estimate and log LLM cost per run based on token counts.
- **Chunking strategy**: If KB chunks are large, consider splitting or sampling sections to avoid overly long contexts that blur evaluation.
- **Retrieval evaluation**: Replace substring containment with embedding similarity or lexical overlap (e.g., ROUGE/LCS) for more robust matching.
- **Metrics completeness**: Track additional metrics like Recall@1, nDCG, and average rank; store run metadata for comparison across experiments.
- **Error handling**: Add retries with exponential backoff for Bedrock calls; surface non-fatal errors in a run summary.
- **Testing and linting**: Add minimal unit tests for parsing/metrics and a formatter/linter (ruff/black) to keep scripts consistent.
- **Dashboard UX**: Add trend comparisons across runs, ability to filter by file/source, and export filtered views.

