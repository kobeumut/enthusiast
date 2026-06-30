# Vector Store (pgvector in PostgreSQL)

Enthusiast stores the Retrieval-Augmented Generation (RAG) index **inside its existing PostgreSQL database** using the [pgvector](https://github.com/pgvector/pgvector) extension. There is **no separate vector database** to install or operate: product and document chunks, their embeddings, and the similarity search all live in the same PostgreSQL instance as the rest of the application data.

This page describes the concrete setup path, how data gets indexed, how it is queried, how to run a backfill/reindex, and how to troubleshoot common failures.

## Where vectors live

| Concern | Implementation |
|---|---|
| Vector database | PostgreSQL with the `vector` extension (pgvector) |
| Container image | `pgvector/pgvector:pg17` (see `docker-compose.yml` / `docker-compose.development.yml`) |
| Extension install | Django migration `server/catalog/migrations/0001_install_pgvector.py` (`VectorExtension()`) |
| Product chunks | `catalog.ProductContentChunk` — `embedding = pgvector.django.VectorField(null=True)` |
| Document chunks | `catalog.DocumentChunk` — `embedding = pgvector.django.VectorField(null=True)` |
| Similarity search | `pgvector.django.CosineDistance` |

The two chunk tables are the actual RAG index. Each row is a slice of a product/document (`content`) plus its embedding vector (`embedding`). Products and documents are grouped by a [Data Set](/docs/management/data-sets), and retrieval is always scoped to a single data set.

## Local stack setup

### 1. Start PostgreSQL (with pgvector)

The bundled Docker Compose file already uses the pgvector image, so a plain `docker compose up` gives you a pgvector-enabled PostgreSQL:

```bash
# Development (bind-mounts server/, exposes Postgres on 5432)
docker compose -f docker-compose.development.yml up

# Production-style
docker compose up
```

The default database credentials come from `server/sample.env`:

```ini
ECL_DB_HOST=postgres
ECL_DB_PORT=5432
ECL_DB_USER=enthusiast
ECL_DB_PASSWORD=enthusiast
ECL_DB_NAME=enthusiast
```

If you point Enthusiast at an **external** PostgreSQL instead, that server must have the `vector` extension available (PostgreSQL 13+ with pgvector installed) and the connecting role must be allowed to `CREATE EXTENSION`.

### 2. Run migrations (creates the `vector` extension)

The API container runs migrations automatically on startup when `RUN_MIGRATIONS=True` (the default for the `api` service). You can also run them manually:

```bash
docker compose exec api python manage.py migrate
```

Migration `server/catalog/migrations/0001_install_pgvector.py` executes `CREATE EXTENSION IF NOT EXISTS vector`. If this migration has run, the extension is installed:

```bash
docker compose exec postgres psql -U enthusiast -d enthusiast -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"
```

### 3. Configure a data set's embedding model

Embeddings are configured **per data set** on the `DataSet` model (`server/catalog/models/data_set.py`):

| Field | Default | Meaning |
|---|---|---|
| `embedding_provider` | `OpenAI` | Which provider plugin generates embeddings |
| `embedding_model` | `text-embedding-3-large` | The model name passed to the provider |
| `embedding_vector_dimensions` | `512` | Length of the stored vector |
| `embedding_chunk_size` | `3000` | Max tokens per chunk |
| `embedding_chunk_overlap` | `150` | Overlap tokens between adjacent chunks |

Set these when creating a data set in the UI (**Manage → Data Sets → New**) or via the API. The available providers are configured by `CATALOG_EMBEDDING_PROVIDERS` in `pecl/settings.py` (ships with the OpenAI provider). Generating embeddings with OpenAI requires `OPENAI_API_KEY` in `server/.env`.

> **The vector dimension is fixed platform-wide at `EMBEDDING_VECTOR_DIMENSIONS` (512).** Every data set stores its chunk embeddings in the same shared `vector(512)` pgvector column (pgvector ANN indexes require a single fixed dimension), so `embedding_vector_dimensions` is **not** a per-data-set setting: it is forced to `512` at creation time, the embedding provider/model/dimensions are **immutable** on an existing data set, and a non-`512` value is rejected by the API with a clear error. To use a different dimension you must change `EMBEDDING_VECTOR_DIMENSIONS` in code **and** run a data migration that recreates both chunk embedding columns at the new dimension, then reindex. The `catalog.W001` system check is a defensive backstop that warns about any data set row that drifts from the fixed dimension (e.g. legacy data or direct DB edits).

## How content gets indexed

Indexing = splitting an item into chunks and embedding each chunk. The flow is the same for products and documents:

1. **Sync** imports products/documents from a source plugin (Shopify, Medusa, the Sample source, …) and creates/updates `Product` / `Document` rows.
2. For **newly created** items, or **updated items whose embedded content changed**, a Celery indexing task is queued:
   - products → `catalog.tasks.index_product_task` (only when `name`/`description` changed — see `Product.get_content`)
   - documents → `catalog.tasks.index_document_task` (only when `content` changed — see `Document.split`)

   Source sync does **not** re-enqueue an item when its embedded content is unchanged, so a routine re-sync that brings back identical data does not trigger a redundant re-split + embedding API call. Non-embedded catalog fields (price, sku, properties, categories) are still updated on the row without re-indexing. To regenerate embeddings for items that were never indexed (or failed), run a [backfill/reindex](#backfill--reindex).
3. The task calls `ProductEmbeddingGenerator.index_object` / `DocumentEmbeddingGenerator.index_object` (`catalog/services.py`), which:
   - **re-splits** the item into chunks (`Product.split` / `Document.split` using LangChain's `TokenTextSplitter`, bounded by the data set's chunk size/overlap), deleting any previous chunks;
   - **embeds** each chunk with the data set's configured provider/model/dimensions;
   - **stores** the vector in the chunk's `embedding` column.

Sync is triggered from the UI (**Configure → Integrations → Sync**) or the API (`POST /api/sync`, `POST /api/product_sources/sync`, …). The Celery **worker** service (`celery -A pecl.celery worker`) executes both sync and indexing tasks, with Redis as the broker.

### Backfill & reindex

Two ways to regenerate embeddings, depending on whether you want it on the worker or in the foreground:

**Management command (foreground, no worker required).** Re-splits and re-embeds using each data set's current configuration. Ideal for an initial backfill or recovering after a model/dimension change. If an individual item fails to index (e.g. an embedding API error), the command logs the failing item and continues with the rest, then exits with a non-zero status and a failure summary so partial failures are never silent:

```bash
# Reindex products AND documents in one data set
docker compose exec api python manage.py reindex --data-set <data_set_id>

# Reindex everything, across all data sets
docker compose exec api python manage.py reindex

# Only one side
docker compose exec api python manage.py reindex --data-set <id> --products
docker compose exec api python manage.py reindex --data-set <id> --documents
```

The backfill is **resilient**: a single bad item (embedding API error, oversize chunk, transient network failure) never aborts the whole run. Each item is retried with exponential backoff, and on terminal failure the error is recorded while the run continues. At the end the command prints a summary like `Reindex summary: 198 ok / 2 fail` followed by the list of failed items (product `entry_id` / document `url` and their primary key) so you can investigate or re-run just those.

```bash
# Stop on the first terminal failure instead of continuing (CI / debugging).
# The command exits non-zero when it aborts.
docker compose exec api python manage.py reindex --data-set <id> --fail-fast

# Tune retry behaviour. --max-attempts is the total tries per item (first attempt
# counts), --retry-backoff is the base seconds for exponential backoff.
docker compose exec api python manage.py reindex --data-set <id> --max-attempts 3 --retry-backoff 1.0

# Resume a large backfill in batches. Items are processed in primary-key order
# when either option is set, so --from-id continues exactly where a batch stopped.
docker compose exec api python manage.py reindex --data-set <id> --from-id <pk> --limit 1000
```

Per-item progress is verbose and only printed at `-v 2` and above (`python manage.py reindex -v 2 ...`); the default output stays limited to headings, per-type totals, and the final summary.

**Celery tasks (async, on the worker).** Re-dispatch per-item indexing for a whole data set:

```python
# docker compose exec api python manage.py shell
from catalog.tasks import index_all_products_task, index_all_documents_task

index_all_products_task.apply_async(args=[<data_set_id>])
index_all_documents_task.apply_async(args=[<data_set_id>])
```

## How retrieval works

At query time the agent embeds the user's question with the **same** data set provider/model/dimensions, then runs an approximate-nearest-neighbor search against the chunk tables using pgvector's cosine distance:

- `agent.core.repositories.DjangoProductChunkRepository.get_chunk_by_distance_for_data_set` — `ORDER BY` `CosineDistance("embedding", query_vector)`, scoped to the data set.
- `agent.core.repositories.DjangoDocumentChunkRepository.get_chunk_by_distance_for_data_set` — same, for documents.
- The product retriever can additionally combine vector distance with PostgreSQL full-text ranking (`SearchRank` / `SearchVector`) via `get_chunk_by_distance_and_keyword_for_data_set`.

**Chunks with `embedding IS NULL` are always skipped.** A cosine distance computed against `NULL` is `NULL`, so without this guard stale/partial chunks (e.g. an item whose embedding generation failed mid-way) could occupy result slots even though they carry no vector. All three chunk-query methods filter `embedding__isnull=False`, so failed-to-index content never surfaces in search results. Use `python manage.py reindex` to backfill any such chunks.

The retrievers that wire this into agents live in `server/agent/core/retrievers/document_retriever.py` and the product retriever shipped with the product-search plugin (see [Concept: Product search Agent](/docs/customization/concept-product-search)).

## Production migration runbook: vector(512) + HNSW indexes

`catalog.0014_pgvector_ann_indexes` and `catalog.0016_hnsw_indexes_concurrent` turn the two chunk embedding columns into a fixed `vector(EMBEDDING_VECTOR_DIMENSIONS)` (512) and build the cosine HNSW ANN indexes. On a **fresh** database `migrate` applies both with no extra steps. On an **existing** database that already stores embeddings, two production risks are handled explicitly:

1. **`0014` rewrites each chunk `embedding` column to `vector(512)`** (`ALTER COLUMN ... TYPE vector(512)`). pgvector rejects that ALTER with `expected 512 dimensions, not N` if any stored row already carries a different dimension, so the offending chunks must be re-embedded or removed first.
2. **`0016` builds the HNSW indexes with `CREATE INDEX CONCURRENTLY`** (split into its own `atomic = False` migration because `CONCURRENTLY` cannot run inside a transaction). Concurrent builds take no table-level `ACCESS EXCLUSIVE` lock, so they are safe on the large chunk tables while read/write traffic continues.

Apply the migration to an existing installation in this order.

**Step 1 — Preflight (detect non-512 data before the ALTER).** Run the preflight command against the target database. It reports stored chunk vectors and data sets whose dimension differs from 512, and exits non-zero on blocking findings so it can gate a deploy/CI step:

```bash
docker compose exec api python manage.py preflight_embedding_dimensions
# add --strict to also fail on non-blocking (mismatched data set config) warnings
```

- **`BLOCKING ... stored vector(s) with dimension != 512`** → the `0014` ALTER will fail. Re-embed the affected chunks (Step 2), then re-run the preflight until it is clean.
- **`DataSet: ... embedding_vector_dimensions != 512` (warning)** → a data set was created by bypassing the API (legacy data or a direct DB edit). Its chunks cannot be stored in the fixed `vector(512)` column. Recreate the data set with the matching `512` dimension and re-sync/reindex it; with `--strict` the preflight treats this as a failure.
- **`Preflight OK`** → safe to proceed to Step 3.

**Step 2 — Dimension validation / re-embed (only if Step 1 reported blocking rows).** Until `0014` runs the column is the legacy unbounded `vector`, so offending chunks still hold their original (non-512) vectors. Regenerate them at the platform dimension so the ALTER succeeds. The simplest path is a reindex, which re-splits and re-embeds using each data set's current configuration:

```bash
docker compose exec api python manage.py reindex                  # everything
docker compose exec api python manage.py reindex --data-set <id>   # one data set
```

For bulk cleanup of stray rows you can also drop the offending chunks directly — they are regenerated on the next sync/reindex:

```sql
DELETE FROM catalog_documentchunk        WHERE embedding IS NOT NULL AND vector_dims(embedding) <> 512;
DELETE FROM catalog_productcontentchunk  WHERE embedding IS NOT NULL AND vector_dims(embedding) <> 512;
```

Re-run Step 1 to confirm `Preflight OK` before continuing.

**Step 3 — Apply the migration (dimension ALTER + concurrent index build).** With the preflight clean, run the migration. `0014` applies the dimension ALTER and the data_set filter indexes; `0016` then builds the two HNSW indexes concurrently:

```bash
docker compose exec api python manage.py migrate catalog
```

Because `0016` uses `CREATE INDEX CONCURRENTLY`, expect it to take longer than a normal migration on large tables (it makes two passes over the data) — this is expected and does **not** lock the tables. If `0016` fails with `relation "..." already exists`, a database that applied an earlier revision of these migrations (when the HNSW indexes were created inside `0014`) already has them — drop the existing indexes and re-run `migrate`. The same recovery applies if `0016` is interrupted mid-build:

```sql
DROP INDEX IF EXISTS document_chunk_embedding_idx;
DROP INDEX IF EXISTS product_chunk_embedding_idx;
```

**Verify.** After the migration, confirm the columns are `vector(512)` and the HNSW indexes exist:

```sql
SELECT format_type(a.atttypid, a.atttypmod)
FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
WHERE c.relname = 'catalog_documentchunk' AND a.attname = 'embedding';   -- expect: vector(512)

SELECT indexname, indexdef FROM pg_indexes
WHERE indexname IN ('document_chunk_embedding_idx', 'product_chunk_embedding_idx');
-- expect: ... USING hnsw (embedding vector_cosine_ops)
```

## Troubleshooting

**`vector` extension / `type "vector" does not exist`**
The pgvector extension is missing. Run migrations (`docker compose exec api python manage.py migrate`) so `server/catalog/migrations/0001_install_pgvector.py` executes. Confirm the database image is `pgvector/pgvector:*`, not plain `postgres`, and that the DB role can `CREATE EXTENSION`.

**Embedding API key missing / `OPENAI_API_KEY`**
With the default OpenAI provider, sync and indexing tasks fail at embedding time if the key is empty or invalid. Set `OPENAI_API_KEY` in `server/.env` and restart the worker: `docker compose restart worker`.

**Embedding dimension mismatch**
Symptoms: errors mentioning vector length, or suddenly poor/empty results, after changing `embedding_model` or `embedding_vector_dimensions`. The embedding dimension is fixed platform-wide at `EMBEDDING_VECTOR_DIMENSIONS` (512) and is immutable on an existing data set, so a mismatch means a data set was created/edited by bypassing the API (legacy data or a direct DB edit). Stored vectors keep the old length until regenerated. Recreate the data set with the matching `512` dimension (or realign the row + run a [reindex](#backfill--reindex)) so every chunk matches the fixed column. The `catalog.W001` system check surfaces such drifted rows on startup.

**No indexed chunks (search returns nothing)**
Products/documents exist but `ProductContentChunk` / `DocumentChunk` have no rows (or rows with `embedding IS NULL`). The sync imported data but indexing tasks didn't run or finish. Check the worker is running, then run `python manage.py reindex --data-set <id>` and verify with the SQL in the [QA checklist](#manual-qa-checklist).

**Celery worker not running**
Sync/indexing tasks are queued but never processed, so nothing gets embedded. The `worker` (and `beat`) services must be up — `RUN_WORKER=True` / `RUN_BEAT=True` in Compose. Check with `docker compose ps` and `docker compose logs worker`.

## Manual QA checklist

End-to-end check that the RAG path is grounded in pgvector-stored chunks, using the bundled sample data.

1. **Start the stack:** `docker compose -f docker-compose.development.yml up` (postgres, redis, api, worker, beat, frontend).
2. **Set `OPENAI_API_KEY`** in `server/.env` (required for embeddings and the LLM).
3. **Create a data set** — sign in (`admin@example.com` / `changeme`), **Manage → Data Sets → New**, keep the default OpenAI embedding settings.
4. **Add sample sources** — **Configure → Integrations → Add Integration**: add a *Sample Product Source* and a *Sample Document Source*, then click **Sync** on each.
5. **Confirm import** — the **Products** and **Documents** tabs show the imported rows.
6. **Confirm the pgvector index is populated.** Connect to the DB and check the extension and chunk/embedding counts:

   ```bash
   docker compose exec postgres psql -U enthusiast -d enthusiast
   ```

   ```sql
   SELECT extname FROM pg_extension WHERE extname = 'vector';          -- expect: vector
   SELECT count(*) FROM catalog_productcontentchunk;                   -- expect: > 0
   SELECT count(*) FROM catalog_documentchunk;                         -- expect: > 0
   SELECT count(*) FROM catalog_productcontentchunk WHERE embedding IS NOT NULL;  -- expect: = total
   SELECT count(*) FROM catalog_documentchunk       WHERE embedding IS NOT NULL;  -- expect: = total
   ```

7. **Force a backfill if needed:** `docker compose exec api python manage.py reindex --data-set <data_set_id>`.
8. **Query through an agent** — **Ask → Catalog Knowledge Agent** (or product/user-manual search agent) and ask something answerable only from the imported sample data; confirm the answer is grounded in that content.
9. **(Optional) Direct retrieval check** from a shell, exercising the exact cosine-distance path used at runtime:

   ```python
   # docker compose exec api python manage.py shell
   from pgvector.django import CosineDistance
   from agent.core.repositories import DjangoProductChunkRepository
   from agent.core.registries.embeddings import EmbeddingProviderRegistry
   from catalog.models import DataSet, ProductContentChunk

   data_set = DataSet.objects.first()
   provider = EmbeddingProviderRegistry().provider_for_dataset(data_set.id)(data_set.embedding_model, data_set.embedding_vector_dimensions)
   query_vec = provider.generate_embeddings("red running shoes")

   repo = DjangoProductChunkRepository(ProductContentChunk)
   for hit in repo.get_chunk_by_distance_for_data_set(data_set.id, CosineDistance("embedding", query_vec))[:5]:
       print(round(hit.distance, 4), hit.content)
   ```

   You should see the most semantically similar chunks returned first, each with a small `distance`.
