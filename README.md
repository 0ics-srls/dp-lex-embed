# dp-lex-embed

Shared Data Platform Process class (Proc-protocol v1.1, Python). Embeds the rows of a dataset table whose vector column is `NULL`
with Jina `jina-embeddings-v3` (1024 dimensions, `retrieval.passage`). The only class that receives the Jina secret.

Parameters of `run`: `connString` (`${dataset.connString}`), `table` (`schema.table`), `keyColumn`, `textColumns[]` (joined by a newline),
`vectorColumn`, `tokenBudget` (hard stop), `batchSize` (default 16), `jinaApiKey` (`${secret.JINA_API_KEY}`), `dryRun`.

- Idempotent: only `vector IS NULL` rows are read; every batch is committed alone, a failed run resumes from the remaining rows.
- `dryRun` with a `connString` counts pending rows and estimated tokens (chars/3), no Jina call, no write; without it nothing is read.
- Metrics: `pending`, `embedded`, `skipped` (blank text), `embed_tokens`, `written.<schema.table>`; check `budget_not_exhausted` (warn).
- Errors: `config_error` (bad config, rejected login, Jina 4xx), `transient_error` (429, 5xx, network, database down), `system_error`.

Development: `python -m venv .venv`, `pip install -r requirements-dev.txt`, `pytest -m "not integration"`, `ruff check .`, `mypy .`;
integration tests need Docker (`pgvector/pgvector:pg16`).
