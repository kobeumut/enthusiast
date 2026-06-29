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

> If you change `embedding_model` or `embedding_vector_dimensions` on an existing data set, [reindex](#backfill--reindex) so stored vectors match the new configuration.

## How content gets indexed

Indexing = splitting an item into chunks and embedding each chunk. The flow is the same for products and documents:

1. **Sync** imports products/documents from a source plugin (Shopify, Medusa, the Sample source, …) and creates/updates `Product` / `Document` rows.
2. For each imported item, a Celery indexing task is queued:
   - products → `catalog.tasks.index_product_task`
   - documents → `catalog.tasks.index_document_task`
3. The task calls `ProductEmbeddingGenerator.index_object` / `DocumentEmbeddingGenerator.index_object` (`catalog/services.py`), which:
   - **re-splits** the item into chunks (`Product.split` / `Document.split` using LangChain's `TokenTextSplitter`, bounded by the data set's chunk size/overlap), deleting any previous chunks;
   - **embeds** each chunk with the data set's configured provider/model/dimensions;
   - **stores** the vector in the chunk's `embedding` column.

Sync is triggered from the UI (**Configure → Integrations → Sync**) or the API (`POST /api/sync`, `POST /api/product_sources/sync`, …). The Celery **worker** service (`celery -A pecl.celery worker`) executes both sync and indexing tasks, with Redis as the broker.

### Backfill & reindex

Two ways to regenerate embeddings, depending on whether you want it on the worker or in the foreground:

**Management command (foreground, no worker required).** Re-splits and re-embeds using each data set's current configuration. Ideal for an initial backfill or recovering after a model/dimension change:

```bash
# Reindex products AND documents in one data set
docker compose exec api python manage.py reindex --data-set <data_set_id>

# Reindex everything, across all data sets
docker compose exec api python manage.py reindex

# Only one side
docker compose exec api python manage.py reindex --data-set <id> --products
docker compose exec api python manage.py reindex --data-set <id> --documents
```

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

The retrievers that wire this into agents live in `server/agent/core/retrievers/document_retriever.py` and the product retriever shipped with the product-search plugin (see [Concept: Product search Agent](/docs/customization/concept-product-search)).

## Troubleshooting

**`vector` extension / `type "vector" does not exist`**
The pgvector extension is missing. Run migrations (`docker compose exec api python manage.py migrate`) so `server/catalog/migrations/0001_install_pgvector.py` executes. Confirm the database image is `pgvector/pgvector:*`, not plain `postgres`, and that the DB role can `CREATE EXTENSION`.

**Embedding API key missing / `OPENAI_API_KEY`**
With the default OpenAI provider, sync and indexing tasks fail at embedding time if the key is empty or invalid. Set `OPENAI_API_KEY` in `server/.env` and restart the worker: `docker compose restart worker`.

**Embedding dimension mismatch**
Symptoms: errors mentioning vector length, or suddenly poor/empty results, after changing `embedding_model` or `embedding_vector_dimensions`. Stored vectors keep the old length until regenerated. Run a [reindex](#backfill--reindex) so every chunk matches the current configuration, and make sure the data set's `embedding_vector_dimensions` is one the provider allows.

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
