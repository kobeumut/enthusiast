# enthusiast.

**Production-ready agentic AI framework for e-commerce.**

Enthusiast is an open-source toolkit for building AI-powered agentic workflows. It ships Retrieval-Augmented Generation (RAG), vector search and a workflow orchestrator on a standard **Python / Django / PostgreSQL / React** stack, and works with cloud models (OpenAI) as well as self-hosted LLMs (Mistral, LLaMA, DeepSeek, Ollama).

> This file is the engineering README for this fork. It documents the platform and, in detail, the **newly added RAG features** (pgvector-backed retrieval, batch embedding, content-hash de-dup, robust reindex, and the optional retrieval-quality pipeline). For a per-task technical breakdown see [`TEKNIK_DOKUMAN.md`](./TEKNIK_DOKUMAN.md). For the Turkish version see [`BeniOku.md`](./BeniOku.md).

---

## Table of contents

- [What it is](#what-it-is)
- [Repository layout](#repository-layout)
- [Quick start (Docker)](#quick-start-docker)
- [Configuration](#configuration)
- [The RAG pipeline at a glance](#the-rag-pipeline-at-a-glance)
- [Using the new RAG features](#using-the-new-rag-features)
  - [1. The pgvector store (PostgreSQL, no extra vector DB)](#1-the-pgvector-store-postgresql-no-extra-vector-db)
  - [2. Embedding configuration (fixed 512-dim)](#2-embedding-configuration-fixed-512-dim)
  - [3. Sync → index (with content-hash de-dup)](#3-sync--index-with-content-hash-de-dup)
  - [4. Batched embeddings](#4-batched-embeddings)
  - [5. Backfill / reindex command](#5-backfill--reindex-command)
  - [6. Retrieval-quality pipeline (filters / hybrid / rerank / MMR / HNSW)](#6-retrieval-quality-pipeline-filters--hybrid--rerank--mmr--hnsw)
- [Troubleshooting](#troubleshooting)
- [Where to look in the code](#where-to-look-in-the-code)
- [License](#license)

---

## What it is

Enthusiast gives you the building blocks for AI commerce tooling:

- **AI product discovery** — natural-language product search grounded in your indexed catalog.
- **User manual search** — precise answers from technical documentation.
- **Order intake** — turn scanned purchase orders / notes into structured e-commerce orders.
- **Catalog enrichment** — extract descriptions, attributes, translations from unstructured sheets.

A **plugin architecture** lets you add agents, LLM/embedding providers and product/document sources as standalone Python packages (`plugins/`).

## Repository layout

```
frontend/          # React 18 + TypeScript + Vite
server/            # Django 5 + DRF + Celery  (apps: agent, catalog, account, sync)
plugins/           # Standalone Python packages (agents, models, sources)
docs/              # Nextra documentation site
docker-compose.yml # Production deployment
docker-compose.development.yml
```

Django apps under `server/`:

| App | Responsibility |
|---|---|
| `agent` | Agent orchestration, conversations, messages, WebSocket streaming, retrievers |
| `catalog` | DataSets, Products, Documents, embedding chunk tables, source configs, sync tasks |
| `account` | User model, DRF token auth |
| `sync` | Source synchronization engine (products + documents) |

## Quick start (Docker)

```bash
# 1. Configure environment
cp server/sample.env server/.env
#   then edit server/.env and set OPENAI_API_KEY=sk-... (required for embeddings + LLM)

# 2. Boot the stack (postgres w/ pgvector, redis, api, worker, beat, frontend)
docker compose -f docker-compose.development.yml up

# 3. Open the UI
#    http://localhost:10001  — log in with admin@example.com / changeme
```

Services and ports:

| Service | Port | Notes |
|---|---|---|
| `api` | `10000` | Django + DRF + Daphne. Runs migrations + serves Swagger at `/api/docs/` |
| `frontend` | `10001` | React UI |
| `postgres` | `5432` | `pgvector/pgvector:pg17` image (pgvector extension built in) |
| `redis` | — | Celery broker/result backend |
| `worker` / `beat` | — | Async sync + indexing tasks |

## Configuration

All configuration is environment-driven. Copy `server/sample.env` to `server/.env` and edit:

```ini
# Database (defaults match the bundled Compose file)
ECL_DB_HOST=postgres
ECL_DB_PORT=5432
ECL_DB_USER=enthusiast
ECL_DB_PASSWORD=enthusiast
ECL_DB_NAME=enthusiast

# Django
ECL_DJANGO_SECRET_KEY=change-me
ECL_DJANGO_DEBUG=True
ECL_DJANGO_ALLOWED_HOSTS=["localhost","127.0.0.1"]
ECL_DJANGO_CORS_ALLOWED_ORIGINS=["http://localhost:10001"]

# Celery
ECL_CELERY_BROKER_URL=redis://redis:6379/0
ECL_CELERY_RESULT_BACKEND=redis://redis:6379/0

# Initial admin user
ECL_ADMIN_EMAIL=admin@example.com
ECL_ADMIN_PASSWORD=changeme

# API keys (OpenAI is the default embedding + LLM provider)
OPENAI_API_KEY=sk-...
```

The **embedding dimension is fixed platform-wide at `512`** (`catalog.models.EMBEDDING_VECTOR_DIMENSIONS`). This is a hard product decision, not a per-data-set setting — see [Embedding configuration](#2-embedding-configuration-fixed-512-dim).

---

## The RAG pipeline at a glance

```
            ┌──────────┐   sync    ┌──────────┐  index task  ┌──────────────────────┐
source ───▶ │ sync mgr │ ────────▶ │ Product  │ ───────────▶ │ split into chunks    │
(Shopify,   │          │           │ Document │              │ embed each chunk     │
Medusa, …)  └──────────┘           └──────────┘              │ store vector(512)    │
                  │                                           └──────────┬───────────┘
                  │  content_hash unchanged? → skip re-index              │
                  └──────────────────────────────────────────────────────┘
                                                                             ▼
                                            ┌────────────────────────────────────────┐
   user query ──embed──▶            query vector                              │
                              ┌─────▼──────────────────────────────────┐     │
                              │ pgvector cosine search over chunks     │ ◀───┘
                              │ (+ optional: filter/hybrid/rerank/MMR) │
                              └─────┬──────────────────────────────────┘
                                    ▼
                        distinct products / top-K chunks → agent
```

Two tables form the actual RAG index, each row = a slice of content + its embedding:

| Table | Embeds | Vector column |
|---|---|---|
| `catalog.ProductContentChunk` | product `name`/`description`/… | `embedding = vector(512)` |
| `catalog.DocumentChunk` | document `content` | `embedding = vector(512)` |

Retrieval is always scoped to a single **Data Set**.

---

## Using the new RAG features

This section is the heart of the new work. Each subsection maps to a tracked task — see [`TEKNIK_DOKUMAN.md`](./TEKNIK_DOKUMAN.md) for full implementation notes.

### 1. The pgvector store (PostgreSQL, no extra vector DB)

Enthusiast stores its entire RAG index **inside the existing PostgreSQL database** using the [pgvector](https://github.com/pgvector/pgvector) extension. There is **no separate vector database** to deploy.

- The bundled Compose image is `pgvector/pgvector:pg17`.
- Migration `server/catalog/migrations/0001_install_pgvector.py` runs `CREATE EXTENSION IF NOT EXISTS vector`.
- HNSW ANN indexes on the chunk embedding columns speed up nearest-neighbour search.

Verify the extension is installed:

```bash
docker compose exec postgres psql -U enthusiast -d enthusiast \
  -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"   # expect: vector
```

### 2. Embedding configuration (fixed 512-dim)

Embeddings are configured **per data set** on the `DataSet` model (`server/catalog/models/data_set.py`):

| Field | Default | Meaning |
|---|---|---|
| `embedding_provider` | `OpenAI` | Which provider plugin generates embeddings |
| `embedding_model` | `text-embedding-3-large` | Model name passed to the provider |
| `embedding_vector_dimensions` | `512` | Length of the stored vector (**forced to 512**) |
| `embedding_chunk_size` | `3000` | Max tokens per chunk |
| `embedding_chunk_overlap` | `150` | Overlap tokens between adjacent chunks |

> **The vector dimension is fixed platform-wide at `EMBEDDING_VECTOR_DIMENSIONS` (512).** Every data set stores its chunk embeddings in the same shared `vector(512)` pgvector column (pgvector ANN indexes require a single fixed dimension). Concretely:
> - `embedding_vector_dimensions` is **not** a per-data-set setting: it is forced to `512` at creation time.
> - The embedding provider/model/dimensions are **immutable** on an existing data set — the API rejects changes with a clear `400` error.
> - A non-`512` value is rejected by the create API with a clear message.
> - To use a different dimension you must change `EMBEDDING_VECTOR_DIMENSIONS` in code **and** run a data migration that recreates both chunk embedding columns at the new dimension, then reindex.
> - The `catalog.W001` system check is a defensive backstop that warns about any data set row that drifts from the fixed dimension (e.g. legacy data or direct DB edits).

In the UI (**Manage → Data Sets → New**), the **Vector Size** field is read-only and fixed at `512`.

### 3. Sync → index (with content-hash de-dup)

Indexing = splitting an item into chunks and embedding each chunk. The flow is the same for products and documents:

1. **Sync** imports products/documents from a source plugin and creates/updates `Product` / `Document` rows.
2. For **newly created** items, or **items whose embedded content changed**, a Celery indexing task is queued:
   - products → `catalog.tasks.index_product_task`
   - documents → `catalog.tasks.index_document_task`
3. The task calls `ProductEmbeddingGenerator.index_object` / `DocumentEmbeddingGenerator.index_object` (`catalog/services.py`), which re-splits the item into chunks, **embeds all chunk contents in one batched call**, and stores each `vector(512)`.

> **Cost saving — canonical content hash.** Source sync does **not** re-enqueue an item when its embedded content is byte-for-byte identical. `Product` and `Document` now carry a `content_hash` (sha256 over the content fields that feed the chunker):
> - Product: `name, description, sku, properties, categories, price`
> - Document: `title, content`
>
> The sync managers look up the previously stored hash before `update_or_create` and skip the re-index task when the hash matches. New items and genuinely changed items are still indexed; legacy rows with a `null` hash are backfilled on first sync. Non-embedded catalog fields (price, sku, properties, categories) are still updated on the row without re-indexing.

Sync is triggered from the UI (**Configure → Integrations → Sync**) or the API.

### 4. Batched embeddings

`index_object` resolves the embedding registry/provider **once** outside the chunk loop and embeds all chunk contents in a single batched request via `generate_embeddings_batch` (`catalog/services.py`). The OpenAI provider implementation reuses a single `OpenAI()` client, sends `input=[...]` lists chunked at 2048, and realigns results by API index. This turns N chunk-level API calls into ~1 call per item and removes redundant provider/client construction.

The base contract lives in `enthusiast-common` (`EmbeddingProvider.generate_embeddings_batch`) with a backwards-compatible default that loops `generate_embeddings`, so custom providers keep working without changes (they just won't benefit from batching until they override it).

### 5. Backfill / reindex command

`python manage.py reindex` re-splits and re-embedds items using each data set's current configuration. It runs **synchronously in the foreground** — ideal for an initial backfill or recovering after a model/dimension change, and it does not depend on a running Celery worker.

```bash
# Reindex products AND documents in one data set
docker compose exec api python manage.py reindex --data-set <data_set_id>

# Reindex everything, across all data sets
docker compose exec api python manage.py reindex

# Only one side
docker compose exec api python manage.py reindex --data-set <id> --products
docker compose exec api python manage.py reindex --data-set <id> --documents
```

**Robust by default.** A single bad item (embedding API error, oversize chunk, transient network failure) never aborts the whole run:

- **Per-item error isolation** — each item is wrapped in `try/except`; on terminal failure the error is recorded and the run continues.
- **Retry/backoff** — each item is retried with exponential backoff. Tunable:
  ```bash
  docker compose exec api python manage.py reindex --data-set <id> \
      --max-attempts 3 --retry-backoff 1.0
  ```
  (`--max-attempts` is total tries, first attempt counts; `--retry-backoff` is the base seconds, delays grow as `backoff * 2 ** (attempt - 1)`.)
- **Summary** — at the end the command prints `Reindex summary: 198 ok / 2 fail` plus the list of failed items (product `entry_id` / document `url` and primary key).
- **`--fail-fast`** — stop on the first terminal failure instead of continuing (CI / debugging). Exits non-zero when it aborts.
- **Resume** — `--from-id` / `--limit` process items in primary-key order for batched backfills:
  ```bash
  docker compose exec api python manage.py reindex --data-set <id> --from-id <pk> --limit 1000
  ```
- **Verbosity** — per-item progress prints only at `-v 2` and above; default output stays limited to headings, per-type totals and the final summary.

Async alternative (Celery, on the worker) — re-dispatch per-item indexing for a whole data set:

```python
# docker compose exec api python manage.py shell
from catalog.tasks import index_all_products_task, index_all_documents_task

index_all_products_task.apply_async(args=[<data_set_id>])
index_all_documents_task.apply_async(args=[<data_set_id>])
```

> **Retrieval guardrail.** Chunks with `embedding IS NULL` are **always skipped** at query time (all chunk-distance queries filter `embedding__isnull=False`). A cosine distance computed against `NULL` is `NULL`, so without this guard stale/partial chunks (e.g. an item whose embedding generation failed mid-way) could occupy result slots. Use `manage.py reindex` to backfill any such chunks.

### 6. Retrieval-quality pipeline (filters / hybrid / rerank / MMR / HNSW)

On top of the pure-vector baseline, four optional **config-driven** retrieval-quality stages (plus runtime HNSW tuning) can be enabled per data set/agent. **All stages are off by default**, so the historical behaviour is preserved exactly until a dataset opts in.

```
query
  → embed → vector ranklist (HNSW, ef_search-tunable)
  → [hybrid]  fuse with full-text keyword ranklist via Reciprocal Rank Fusion
  → [rerank]  lexical rerank over the candidate pool
  → [MMR]     diversity selection (documents only) over the candidate embeddings
  → distinct products / top-K chunks
```

| Stage | Knob | What it does | Applies to |
|---|---|---|---|
| **Metadata filter** | `filters=` (call-time) | Pushes category/price (products) or url/title (documents) predicates into the chunk queryset *before* ranking, so out-of-scope chunks never enter it | products + documents |
| **Hybrid (RRF)** | `hybrid_enabled: True` | Fuses the vector ranklist with a PostgreSQL full-text keyword ranklist via Reciprocal Rank Fusion (k=60). Recovers exact-term hits (SKUs, model numbers) the embedding space blurs | products + documents |
| **Rerank** | `reranker_enabled: True` | Applies a cheap lexical reranker over the candidate pool, blending vector similarity with lexical coverage (fraction of query tokens present in the chunk). Never drops candidates — only reorders | products + documents |
| **MMR diversity** | `mmr_enabled: True`, `mmr_lambda: 0.5` | Selects relevant-yet-diverse chunks via Maximal Marginal Relevance so one document's near-duplicate sections don't crowd out others. `lambda` 1.0 = pure relevance, 0.0 = pure novelty | documents only |
| **HNSW tuning** | `ef_search: N` | Runs `SET LOCAL hnsw.ef_search = N` before the vector query, tuning the HNSW candidate-list size at runtime without rebuilding the index (larger = more recall, more latency) | products + documents |

Stages act on a **candidate pool** (default 50 for products, 60 for documents) rather than the final top-K, so rerank/MMR have room to promote good candidates just outside the final cut.

#### Enabling the stages

Stages are wired through `RetrieverConfig.extra_kwargs` in `agent/core/agents/default_config.py`. To opt a dataset/agent into the full pipeline:

```python
from agent.core.agents.default_config import get_default_config
from agent.core.retrievers import DocumentRetriever, ProductRetriever
from enthusiast_common.config import RetrieverConfig, RetrieversConfig

config = get_default_config()
config.retrievers = RetrieversConfig(
    product=RetrieverConfig(
        retriever_class=ProductRetriever,
        extra_kwargs={
            "number_of_products": 12,
            "hybrid_enabled": True,
            "reranker_enabled": True,
            "ef_search": 100,
        },
    ),
    document=RetrieverConfig(
        retriever_class=DocumentRetriever,
        extra_kwargs={
            "max_objects": 12,
            "hybrid_enabled": True,
            "reranker_enabled": True,
            "mmr_enabled": True,
            "mmr_lambda": 0.5,
            "ef_search": 100,
        },
    ),
)
```

#### Using metadata filters at call time

Filters are passed per-call to the retriever (a `RetrievalFilters` value object):

```python
from agent.core.retrievers.filters import RetrievalFilters

# products: categories (any-of, case-insensitive substring) + price range
retriever.find_products_matching_query(
    "running shoes",
    filters=RetrievalFilters(categories=["Running"], price_max=100),
)

# documents: url / title substring scope
retriever.find_content_matching_query(
    "warranty",
    filters=RetrievalFilters(title_contains="AC-2000"),
)
```

`filters=None` (the default) applies no predicate.

#### Plugging a custom reranker

`LexicalReranker` is the default, but the contract is agnostic — subclass `BaseReranker` and inject it via the `reranker=` kwarg:

```python
from agent.core.retrievers.reranking import BaseReranker

class CrossEncoderReranker(BaseReranker):
    def rerank(self, query, candidates):
        # score candidates with your cross-encoder / LLM, return reordered list
        ...

ProductRetriever(..., reranker=CrossEncoderReranker(), ...)
```

#### Important note on `ef_search` today

`EXPLAIN` shows the **production dataset-scoped** retrieval query (which joins the chunk table to its parent and filters by `data_set_id`) does **not** use the HNSW index today — that relational predicate cannot be served by the embedding index. So on the current schema, `ef_search` has no effect on the production path; the real quality lift comes from **metadata filtering, hybrid RRF, reranking and MMR**. The robust fix (denormalising `data_set_id` onto the chunk tables + a per-dataset partial HNSW index, or a two-stage retrieve-then-filter) is tracked as a follow-up.

---

## Troubleshooting

**`type "vector" does not exist`** — the pgvector extension is missing. Run migrations so `0001_install_pgvector.py` executes, confirm the DB image is `pgvector/pgvector:*`, and that the DB role can `CREATE EXTENSION`:

```bash
docker compose exec api python manage.py migrate
```

**Embedding API key missing / invalid** — with the default OpenAI provider, sync and indexing tasks fail at embedding time if `OPENAI_API_KEY` is empty/invalid. Set it in `server/.env` and restart the worker:

```bash
docker compose restart worker
```

**Embedding dimension mismatch** — symptoms are errors mentioning vector length, or suddenly poor/empty results. The dimension is fixed platform-wide at `512` and immutable on an existing data set, so a mismatch means a data set was created/edited by bypassing the API. Recreate the data set at `512` (or realign the row + reindex). The `catalog.W001` check surfaces drifted rows on startup.

**No indexed chunks (search returns nothing)** — items exist but `ProductContentChunk` / `DocumentChunk` have no rows (or rows with `embedding IS NULL`). Confirm the worker is running, then:

```bash
docker compose exec api python manage.py reindex --data-set <id>
```

**Celery worker not running** — sync/indexing tasks are queued but never processed. Ensure `RUN_WORKER=True` / `RUN_BEAT=True` in Compose, then `docker compose ps` and `docker compose logs worker`.

### Manual QA checklist

```bash
# 1. Start the stack + set OPENAI_API_KEY in server/.env
docker compose -f docker-compose.development.yml up

# 2. Log in (admin@example.com / changeme), create a data set (keep OpenAI defaults),
#    add Sample Product/Document sources under Configure → Integrations, click Sync.

# 3. Confirm the index is populated
docker compose exec postgres psql -U enthusiast -d enthusiast
```

```sql
SELECT count(*) FROM catalog_productcontentchunk;                          -- expect: > 0
SELECT count(*) FROM catalog_productcontentchunk WHERE embedding IS NOT NULL;  -- expect: = total
SELECT count(*) FROM catalog_documentchunk WHERE embedding IS NOT NULL;        -- expect: = total
```

```bash
# 4. Query through an agent (Catalog Knowledge / product / user-manual search).
# 5. Force a backfill if needed:
docker compose exec api python manage.py reindex --data-set <data_set_id>
```

---

## Where to look in the code

| Concern | Location |
|---|---|
| Chunk tables + fixed dim | `server/catalog/models/{product_content_chunk,document_chunk,data_set}.py` |
| Embedding indexing (batched) | `server/catalog/services.py` (`*EmbeddingGenerator.index_object`) |
| Sync + content-hash de-dup | `server/sync/{product,document}/manager.py`, `server/catalog/models/{product,document}.py` (`compute_content_hash`) |
| Reindex command | `server/catalog/management/commands/reindex.py` |
| Retrieval repos (cosine + null guard) | `server/agent/core/repositories.py` |
| Retrievers | `server/agent/core/retrievers/{product,document}_retriever.py` |
| Quality stages | `server/agent/core/retrievers/{filters,hybrid,reranking,diversity}.py` |
| Default retriever config | `server/agent/core/agents/default_config.py` |
| Fixed-512 enforcement | `server/catalog/serializers.py`, `server/catalog/checks.py`, `frontend/.../data-set-form.tsx` |
| In-depth design notes | `server/agent/core/retrievers/RETRIEVAL_QUALITY.md` |

## License

MIT — see [`LICENSE.md`](../LICENSE.md). Enthusiast is fully open-source and will always remain free.
