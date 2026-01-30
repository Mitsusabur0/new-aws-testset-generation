# Changes Log

## 2026-01-30

- Centralized configuration in `config.py` with environment overrides (paths, AWS profiles/region, model, K values).
- Wired `1_testset_generator.py`, `2_retriever.py`, `3_evaluator.py`, and `app.py` to use the shared config.
- Added deterministic seeding for testset generation and recorded the seed in output rows.
- Fixed mojibake in Spanish prompt strings and corrected UI emoji encoding; cleaned roadmap text.
- Hardened XML parsing with strict validation and repair fallback; logged raw parse failures.
- Added Bedrock retry/backoff wrappers and run summary outputs for non-fatal errors.
