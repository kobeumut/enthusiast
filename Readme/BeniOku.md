# enthusiast.

**E-ticaret için production-ready agentic AI çerçevesi.**

Enthusiast, AI destekli agentic iş akışları kurmak için açık kaynaklı bir araç setidir. Retrieval-Augmented Generation (RAG), vektör arama ve iş akışı orkestratörünü standart bir **Python / Django / PostgreSQL / React** yığınında sunar; hem bulut modelleriyle (OpenAI) hem de kendi sunduğunuz LLM'lerle (Mistral, LLaMA, DeepSeek, Ollama) çalışır.

> Bu dosya bu fork için mühendislik README'sidir. Platformu ve özellikle **yeni eklenen RAG özelliklerini** (pgvector tabanlı retrieval, batch embedding, content-hash ile tekilleştirme, dayanıklı reindex ve opsiyonel retrieval-quality hattı) ayrıntılı olarak belgeler. Görev bazlı teknik döküm için [`TEKNIK_DOKUMAN.md`](./TEKNIK_DOKUMAN.md). İngilizce sürüm için [`README.md`](./README.md).

---

## İçindekiler

- [Nedir?](#nedir)
- [Repo düzeni](#repo-düzeni)
- [Hızlı başlangıç (Docker)](#hızlı-başlangıç-docker)
- [Yapılandırma](#yapılandırma)
- [RAG hattına genel bakış](#rag-hattına-genel-bakış)
- [Yeni RAG özelliklerini kullanmak](#yeni-rag-özelliklerini-kullanmak)
  - [1. pgvector deposu (PostgreSQL, ayrı vektör DB yok)](#1-pgvector-deposu-postgresql-ayrı-vektör-db-yok)
  - [2. Embedding yapılandırması (sabit 512 boyut)](#2-embedding-yapılandırması-sabit-512-boyut)
  - [3. Sync → index (content-hash ile tekilleştirme)](#3-sync--index-content-hash-ile-tekilleştirme)
  - [4. Batch (toplu) embedding](#4-batch-toplu-embedding)
  - [5. Backfill / reindex komutu](#5-backfill--reindex-komutu)
  - [6. Retrieval-quality hattı (filtre / hybrid / rerank / MMR / HNSW)](#6-retrieval-quality-hattı-filtre--hybrid--rerank--mmr--hnsw)
- [Sorun giderme](#sorun-giderme)
- [Kodda nereye bakmalı](#kodda-nereye-bakmalı)
- [Lisans](#lisans)

---

## Nedir?

Enthusiast, AI ticaret araçlarının yapı taşlarını verir:

- **AI ürün keşfi** — indekslenmiş kataloğa dayalı, doğal dilde ürün arama.
- **Kullanım kılavuzu arama** — teknik dokümantasyondan kesin cevaplar.
- **Sipariş alımı (order intake)** — taranmış sipariş formlarını / notları yapılandırılmış e-ticaret siparişlerine çevirme.
- **Katalog zenginleştirme** — yapılandırılmamış fişlerden açıklama, nitelik, çeviriler çıkarma.

Bir **eklenti (plugin) mimarisi** sayesinde ajanları, LLM/embedding sağlayıcılarını ve ürün/doküman kaynaklarını bağımsız Python paketleri olarak (`plugins/`) ekleyebilirsiniz.

## Repo düzeni

```
frontend/          # React 18 + TypeScript + Vite
server/            # Django 5 + DRF + Celery  (uygulamalar: agent, catalog, account, sync)
plugins/           # Bağımsız Python paketleri (ajanlar, modeller, kaynaklar)
docs/              # Nextra dokümantasyon sitesi
docker-compose.yml # Production dağıtımı
docker-compose.development.yml
```

`server/` altındaki Django uygulamaları:

| Uygulama | Sorumluluk |
|---|---|
| `agent` | Ajan orkestrasyonu, konuşmalar, mesajlar, WebSocket akışı, retriever'lar |
| `catalog` | DataSet'ler, Ürünler, Dokümanlar, embedding chunk tabloları, kaynak config'leri, sync görevleri |
| `account` | User modeli, DRF token auth |
| `sync` | Kaynak senkronizasyon motoru (ürün + doküman) |

## Hızlı başlangıç (Docker)

```bash
# 1. Ortamı yapılandır
cp server/sample.env server/.env
#   ardından server/.env içine OPENAI_API_KEY=sk-... yaz (embedding + LLM için gerekli)

# 2. Yığını başlat (pgvector'lu postgres, redis, api, worker, beat, frontend)
docker compose -f docker-compose.development.yml up

# 3. UI'ı aç
#    http://localhost:10001  — admin@example.com / changeme ile giriş yap
```

Servisler ve portlar:

| Servis | Port | Notlar |
|---|---|---|
| `api` | `10000` | Django + DRF + Daphne. Migration'ları çalıştırır, Swagger `/api/docs/` |
| `frontend` | `10001` | React UI |
| `postgres` | `5432` | `pgvector/pgvector:pg17` imajı (pgvector eklentisi gömülü) |
| `redis` | — | Celery broker/result backend |
| `worker` / `beat` | — | Asenkron sync + index görevleri |

## Yapılandırma

Tüm yapılandırma ortam değişkenleriyledir. `server/sample.env` dosyasını `server/.env` olarak kopyalayıp düzenleyin:

```ini
# Veritabanı (varsayılanlar bundle Compose ile uyumlu)
ECL_DB_HOST=postgres
ECL_DB_PORT=5432
ECL_DB_USER=enthusiast
ECL_DB_PASSWORD=enthusiast
ECL_DB_NAME=enthusiast

# Django
ECL_DJANGO_SECRET_KEY=degistir-beni
ECL_DJANGO_DEBUG=True
ECL_DJANGO_ALLOWED_HOSTS=["localhost","127.0.0.1"]
ECL_DJANGO_CORS_ALLOWED_ORIGINS=["http://localhost:10001"]

# Celery
ECL_CELERY_BROKER_URL=redis://redis:6379/0
ECL_CELERY_RESULT_BACKEND=redis://redis:6379/0

# İlk admin kullanıcı
ECL_ADMIN_EMAIL=admin@example.com
ECL_ADMIN_PASSWORD=changeme

# API anahtarları (OpenAI varsayılan embedding + LLM sağlayıcısıdır)
OPENAI_API_KEY=sk-...
```

**Embedding boyutu platform genelinde `512` olarak sabitlenmiştir** (`catalog.models.EMBEDDING_VECTOR_DIMENSIONS`). Bu katı bir ürün kararıdır, data-set başına ayar değildir — bkz. [Embedding yapılandırması](#2-embedding-yapılandırması-sabit-512-boyut).

---

## RAG hattına genel bakış

```
            ┌──────────┐   sync    ┌──────────┐  index task  ┌──────────────────────┐
kaynak ───▶ │ sync mgr │ ────────▶ │ Product  │ ───────────▶ │ chunk'lara böl       │
(Shopify,   │          │           │ Document │              │ her chunk'ı embed et │
Medusa, …)  └──────────┘           └──────────┘              │ vector(512) sakla    │
                  │                                           └──────────┬───────────┘
                  │  content_hash aynıysa? → re-index atla               │
                  └──────────────────────────────────────────────────────┘
                                                                             ▼
                                            ┌────────────────────────────────────────┐
   kullanıcı sorgusu ──embed──▶      sorgu vektörü                             │
                              ┌─────▼──────────────────────────────────┐     │
                              │ chunk'lar üzerinden pgvector cosine    │ ◀───┘
                              │ (+ opsiyonel: filter/hybrid/rerank/MMR)│
                              └─────┬──────────────────────────────────┘
                                    ▼
                        distinct ürünler / top-K chunk → ajana
```

Asıl RAG indeksini iki tablo oluşturur; her satır = bir içerik dilimi + embedding'i:

| Tablo | Embed edilen | Vektör sütunu |
|---|---|---|
| `catalog.ProductContentChunk` | ürün `name`/`description`/… | `embedding = vector(512)` |
| `catalog.DocumentChunk` | doküman `content` | `embedding = vector(512)` |

Retrieval her zaman tek bir **Data Set** ile sınırlıdır.

---

## Yeni RAG özelliklerini kullanmak

Bu bölüm yeni işin özüdür. Her alt bölüm bir takip görevine karşılık gelir — tam uygulama notları için [`TEKNIK_DOKUMAN.md`](./TEKNIK_DOKUMAN.md).

### 1. pgvector deposu (PostgreSQL, ayrı vektör DB yok)

Enthusiast tüm RAG indeksini **mevcut PostgreSQL veritabanının içinde**, [pgvector](https://github.com/pgvector/pgvector) eklentisiyle saklar. Kurmanız gereken **ayrı bir vektör veritabanı yoktur**.

- Bundle Compose imajı `pgvector/pgvector:pg17`'dir.
- Migration `server/catalog/migrations/0001_install_pgvector.py`, `CREATE EXTENSION IF NOT EXISTS vector` çalıştırır.
- Chunk embedding sütunlarındaki HNSW ANN indeksleri en-yakın-komşu aramasını hızlandırır.

Eklentinin kurulu olduğunu doğrulayın:

```bash
docker compose exec postgres psql -U enthusiast -d enthusiast \
  -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"   # beklenen: vector
```

### 2. Embedding yapılandırması (sabit 512 boyut)

Embedding'ler data-set bazında `DataSet` modelinde yapılandırılır (`server/catalog/models/data_set.py`):

| Alan | Varsayılan | Anlamı |
|---|---|---|
| `embedding_provider` | `OpenAI` | Hangi sağlayıcı eklentisi embedding üretir |
| `embedding_model` | `text-embedding-3-large` | Sağlayıcıya geçirilen model adı |
| `embedding_vector_dimensions` | `512` | Saklanan vektörün uzunluğu (**512'ye zorlanır**) |
| `embedding_chunk_size` | `3000` | Chunk başına maks. token |
| `embedding_chunk_overlap` | `150` | Komşu chunk'lar arası örtüşme token'ı |

> **Vektör boyutu platform genelinde `EMBEDDING_VECTOR_DIMENSIONS` (512) olarak sabittir.** Her data set chunk embedding'lerini aynı paylaşımlı `vector(512)` pgvector sütununda saklar (pgvector ANN indeksleri tek bir sabit boyut gerektirir). Açıkçası:
> - `embedding_vector_dimensions` bir data-set ayarı **değildir**: oluşturma sırasında `512`'ye zorlanır.
> - Mevcut bir data set'te embedding sağlayıcı/model/boyut **değiştirilemez** — API değişiklikleri net bir `400` hatasıyla reddeder.
> - `512` dışındaki bir değer, oluşturma API'si tarafından net bir mesajla reddedilir.
> - Farklı bir boyut kullanmak için `EMBEDDING_VECTOR_DIMENSIONS`'ı kodda değiştirmeniz **ve** her iki chunk embedding sütununu yeni boyutta yeniden oluşturan bir data migration çalıştırmanız, ardından reindex yapmanız gerekir.
> - `catalog.W001` sistem kontrolü, sabit boyuttan sapan data-set satırlarını (ör. eski veri veya doğrudan DB düzenlemesi) uyaran savunma amaçlı bir backstop'tur.

UI'da (**Manage → Data Sets → New**), **Vector Size** alanı salt okunurdur ve `512`'ye sabitlenmiştir.

### 3. Sync → index (content-hash ile tekilleştirme)

Index = bir öğeyi chunk'lara bölüp her chunk'ı embed etmektir. Akış ürün ve doküman için aynıdır:

1. **Sync**, kaynak eklentisinden ürün/doküman içe aktarır ve `Product` / `Document` satırlarını oluşturur/günceller.
2. **Yeni oluşturulan** veya **embed edilen içeriği değişen** öğeler için bir Celery index görevi kuyruğa alınır:
   - ürünler → `catalog.tasks.index_product_task`
   - dokümanlar → `catalog.tasks.index_document_task`
3. Görev, `ProductEmbeddingGenerator.index_object` / `DocumentEmbeddingGenerator.index_object` (`catalog/services.py`) çağırır; öğeyi chunk'lara böler, **tüm chunk içeriklerini tek bir batch çağrıyla** embed eder ve her `vector(512)`'yi saklar.

> **Maliyet tasarrufu — canonical content hash.** Kaynak sync'i, embed edilen içerik bayt bayt aynı olduğunda öğeyi yeniden kuyruğa **almaz**. `Product` ve `Document` artık bir `content_hash` taşır (chunker'a giden içerik alanları üzerinden sha256):
> - Ürün: `name, description, sku, properties, categories, price`
> - Doküman: `title, content`
>
> Sync yöneticileri, `update_or_create` öncesi daha önce saklanan hash'e bakar ve hash eşleşirse re-index görevini atlar. Yeni öğeler ve gerçekten değişen öğeler yine indekslenir; `null` hash'li eski satırlar ilk sync'te backfill edilir. Embed edilmeyen katalog alanları (price, sku, properties, categories) satırda re-index olmadan güncellenmeye devam eder.

Sync UI'dan (**Configure → Integrations → Sync**) veya API'den tetiklenir.

### 4. Batch (toplu) embedding

`index_object`, embedding registry/sağlayıcısını chunk döngüsünün **dışında bir kez** çözer ve tüm chunk içeriklerini `generate_embeddings_batch` ile tek bir batch isteğinde embed eder (`catalog/services.py`). OpenAI sağlayıcı uygulaması tek bir `OpenAI()` istemcisini yeniden kullanır, `input=[...]` listelerini 2048 parçalı gönderir ve sonuçları API indeksine göre yeniden hizalar. Bu, N adet chunk seviyesi API çağrısını öğe başına ~1 çağrıya çevirir ve yedek sağlayıcı/istemci kurulumunu ortadan kaldırır.

Temel sözleşme `enthusiast-common`'dadır (`EmbeddingProvider.generate_embeddings_batch`) ve `generate_embeddings`'i döngüye alan geriye-uyumlu bir varsayılanla gelir; böylece özel sağlayıcılar değişiklik yapmadan çalışmaya devam eder (override etmedikçe batch'ten faydalanmazlar).

### 5. Backfill / reindex komutu

`python manage.py reindex`, her data set'in mevcut yapılandırmasıyla öğeleri yeniden bölüp embed eder. **Senkron olarak ön planda** çalışır — ilk backfill veya model/boyut değişikliği sonrası toparlanma için idealdir ve Celery worker'a bağlı değildir.

```bash
# Tek data set'te ürün VE dokümanları reindex et
docker compose exec api python manage.py reindex --data-set <data_set_id>

# Tüm data set'lerde her şeyi reindex et
docker compose exec api python manage.py reindex

# Sadece bir taraf
docker compose exec api python manage.py reindex --data-set <id> --products
docker compose exec api python manage.py reindex --data-set <id> --documents
```

**Varsayılan olarak dayanıklıdır.** Tek bir bozuk öğe (embedding API hatası, aşırı büyük chunk, geçici ağ arızası) tüm çalışmayı durdurmaz:

- **Öğe başına hata izolasyonu** — her öğe `try/except` içine alınır; kalıcı hata durumunda hata kaydedilir ve çalışma devam eder.
- **Retry/backoff** — her öğe eksponansiyel backoff ile yeniden denenir. Ayarlanabilir:
  ```bash
  docker compose exec api python manage.py reindex --data-set <id> \
      --max-attempts 3 --retry-backoff 1.0
  ```
  (`--max-attempts` toplam denemedir, ilk deneme sayılır; `--retry-backoff` taban saniyedir, gecikmeler `backoff * 2 ** (attempt - 1)` olarak büyür.)
- **Özet** — komut sonunda `Reindex summary: 198 ok / 2 fail` artı başarısız öğelerin listesini (ürün `entry_id` / doküman `url` ve primary key) yazdırır.
- **`--fail-fast`** — devam etmek yerine ilk kalıcı hatada dur (CI / hata ayıklama). Durdurduğunda non-zero çıkar.
- **Devam etme (resume)** — `--from-id` / `--limit` öğeleri toplu backfill için primary-key sırasında işler:
  ```bash
  docker compose exec api python manage.py reindex --data-set <id> --from-id <pk> --limit 1000
  ```
- **Verbosity** — öğe başına ilerleme yalnızca `-v 2` ve üzeri seviyede yazdırılır; varsayılan çıktı başlıklar, tür başına toplamlar ve son özetle sınırlı kalır.

Asenkron alternatif (Celery, worker üzerinde) — bir data set için öğe başına index görevlerini yeniden dağıt:

```python
# docker compose exec api python manage.py shell
from catalog.tasks import index_all_products_task, index_all_documents_task

index_all_products_task.apply_async(args=[<data_set_id>])
index_all_documents_task.apply_async(args=[<data_set_id>])
```

> **Retrieval guardrail.** `embedding IS NULL` olan chunk'lar sorgu zamanında **her zaman atlanır** (tüm chunk-distance sorguları `embedding__isnull=False` filtresi uygular). `NULL`'a göre hesaplanan bir cosine distance `NULL`'dur; bu koruma olmadan, embedding üretimi yarım kalmış bir öğenin eski/kısmi chunk'ları sonuç yuvalarını işgal edebilirdi. Bu tarz chunk'ları backfill etmek için `manage.py reindex` kullanın.

### 6. Retrieval-quality hattı (filtre / hybrid / rerank / MMR / HNSW)

Saf-vektör taban çizgisinin üzerine, data-set/ajan başına etkinleştirilebilen dört opsiyonel **config-driven** retrieval-quality aşaması (artı runtime HNSW ayarı) eklenebilir. **Tüm aşamalar varsayılan olarak kapalıdır**, böylece bir data set opt-in olmadan tarihsel davranış birebir korunur.

```
sorgu
  → embed → vektör ranklist (HNSW, ef_search ayarlanabilir)
  → [hybrid]  tam-metin anahtar-kelime ranklist ile Reciprocal Rank Fusion ile birleştir
  → [rerank]  aday havuzu üzerinde leksikal rerank
  → [MMR]     çeşitlilik seçimi (yalnızca dokümanlar) aday embedding'leri üzerinden
  → distinct ürünler / top-K chunk
```

| Aşama | Ayar | Ne yapar | Geçerli |
|---|---|---|---|
| **Metadata filtre** | `filters=` (çağrı anında) | Kategori/fiyat (ürün) veya url/başlık (doküman) koşullarını ranklamadan *önce* chunk queryset'ine iter, böylece kapsam dışı chunk'lar hiç giremez | ürün + doküman |
| **Hybrid (RRF)** | `hybrid_enabled: True` | Vektör ranklist'i bir PostgreSQL tam-metin anahtar-kelime ranklist'i ile Reciprocal Rank Fusion (k=60) ile birleştirir. Embed uzayının bulanıklaştığı tam terim eşleşmelerini (SKU, model numaraları) kurtarır | ürün + doküman |
| **Rerank** | `reranker_enabled: True` | Aday havuzu üzerinde ucuz bir leksikal reranker uygular; vektör benzerliğini leksikal kapsama (sorgu token'larının chunk'ta bulunma oranı) ile harmanlar. Aday asla düşürmez — yalnızca yeniden sıralar | ürün + doküman |
| **MMR çeşitlilik** | `mmr_enabled: True`, `mmr_lambda: 0.5` | İlgili-ama-çeşitli chunk'ları Maximal Marginal Relevance ile seçer; tek bir dokümanın yinelenen bölümleri diğerlerini ezmesin. `lambda` 1.0 = saf ilgi, 0.0 = saf yenilik | yalnızca doküman |
| **HNSW ayarı** | `ef_search: N` | Vektör sorgusundan önce `SET LOCAL hnsw.ef_search = N` çalıştırır; HNSW aday-liste boyutunu runtime'da indeksi yeniden kurmadan ayarlar (büyük = daha çok recall, daha çok gecikme) | ürün + doküman |

Aşamalar, son top-K yerine bir **aday havuzu** (ürünler için varsayılan 50, dokümanlar için 60) üzerinde çalışır; böylece rerank/MMR'in son cut dışındaki iyi adayları öne çıkarma alanı olur.

#### Aşamaları etkinleştirme

Aşamalar `agent/core/agents/default_config.py` içindeki `RetrieverConfig.extra_kwargs` ile devreye alınır. Bir data-set/ajan'ı tam hattı için opt-in etmek için:

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

#### Çağrı anında metadata filtrelerini kullanma

Filtreler retriever'a çağrı başına geçirilir (bir `RetrievalFilters` değer nesnesi):

```python
from agent.core.retrievers.filters import RetrievalFilters

# ürünler: kategoriler (herhangi biri, büyük-küçük harf duyarsız alt-dize) + fiyat aralığı
retriever.find_products_matching_query(
    "koşu ayakkabısı",
    filters=RetrievalFilters(categories=["Running"], price_max=100),
)

# dokümanlar: url / başlık alt-dize kapsamı
retriever.find_content_matching_query(
    "garanti",
    filters=RetrievalFilters(title_contains="AC-2000"),
)
```

`filters=None` (varsayılan) hiçbir koşul uygulamaz.

#### Özel reranker takma

`LexicalReranker` varsayılandır, ancak sözleşme uygulamadan bağımsızdır — `BaseReranker`'ı alt sınıfla ve `reranker=` kwarg'u ile enjekte et:

```python
from agent.core.retrievers.reranking import BaseReranker

class CrossEncoderReranker(BaseReranker):
    def rerank(self, query, candidates):
        # adayları kendi cross-encoder / LLM'inle puanla, yeniden sıralı liste döndür
        ...

ProductRetriever(..., reranker=CrossEncoderReranker(), ...)
```

#### Bugün `ef_search` ile ilgili önemli not

`EXPLAIN`, **production data-set kapsamlı** retrieval sorgusunun (chunk tablosunu ebeveynine join edip `data_set_id` ile filtreleyen) bugün HNSW indeksini **kullanmadığını** gösterir — bu ilişkisel koşul embedding indeksi tarafından hizmet edilemez. Dolayısıyla mevcut şemada `ef_search`'in production yolu üzerinde etkisi yoktur; asıl kalite artışı **metadata filtre, hybrid RRF, rerank ve MMR**'dan gelir. Sağlam düzeltme (`data_set_id`'yi chunk tablolarına denormalize etmek + data-set başına partial HNSW indeksi, veya iki aşamalı retrieve-then-filter) bir takip olarak izleniyor.

---

## Sorun giderme

**`type "vector" does not exist`** — pgvector eklentisi eksik. `0001_install_pgvector.py` çalışsın diye migration'ları çalıştırın, DB imajının `pgvector/pgvector:*` olduğunu ve DB rolünün `CREATE EXTENSION` yapabildiğini doğrulayın:

```bash
docker compose exec api python manage.py migrate
```

**Embedding API anahtarı eksik / geçersiz** — varsayılan OpenAI sağlayıcısıyla, `OPENAI_API_KEY` boş/geçersizse sync ve index görevleri embedding aşamasında başarısız olur. `server/.env` içinde ayarlayın ve worker'ı yeniden başlatın:

```bash
docker compose restart worker
```

**Embedding boyut uyumsuzluğu** — belirtiler vektör uzunluğundan bahseden hatalar veya aniden kötü/boş sonuçlardır. Boyut platform genelinde `512`'ye sabitlenmiştir ve mevcut data-set'te değiştirilemez; uyumsuzluk, API atlanarak oluşturulmuş/edilmiş bir data set anlamına gelir. Data set'i `512` ile yeniden oluşturun (veya satırı hizalayıp reindex edin). `catalog.W001` kontrolü sapmış satırları başlangıçta gösterir.

**İndeksli chunk yok (arama hiçbir şey döndürmüyor)** — öğeler var ama `ProductContentChunk` / `DocumentChunk`'da satır yok (veya `embedding IS NULL` satırlar var). Worker'ın çalıştığını doğrulayın, ardından:

```bash
docker compose exec api python manage.py reindex --data-set <id>
```

**Celery worker çalışmıyor** — sync/index görevleri kuyrukta ama işlenmiyor. Compose'ta `RUN_WORKER=True` / `RUN_BEAT=True` olduğundan emin olun, sonra `docker compose ps` ve `docker compose logs worker`.

### Manuel QA kontrol listesi

```bash
# 1. Yığını başlat + server/.env içine OPENAI_API_KEY yaz
docker compose -f docker-compose.development.yml up

# 2. Giriş yap (admin@example.com / changeme), bir data set oluştur (OpenAI varsayılanlarını koru),
#    Configure → Integrations altına Sample Product/Document kaynakları ekle, Sync'e tıkla.

# 3. İndeksin dolduğunu doğrula
docker compose exec postgres psql -U enthusiast -d enthusiast
```

```sql
SELECT count(*) FROM catalog_productcontentchunk;                          -- beklenen: > 0
SELECT count(*) FROM catalog_productcontentchunk WHERE embedding IS NOT NULL;  -- beklenen: = toplam
SELECT count(*) FROM catalog_documentchunk WHERE embedding IS NOT NULL;        -- beklenen: = toplam
```

```bash
# 4. Bir ajan üzerinden sor (Catalog Knowledge / ürün / kullanım kılavuzu arama).
# 5. Gerekirse backfill'i zorla:
docker compose exec api python manage.py reindex --data-set <data_set_id>
```

---

## Kodda nereye bakmalı

| Konu | Konum |
|---|---|
| Chunk tabloları + sabit boyut | `server/catalog/models/{product_content_chunk,document_chunk,data_set}.py` |
| Embedding indexing (batch) | `server/catalog/services.py` (`*EmbeddingGenerator.index_object`) |
| Sync + content-hash tekilleştirme | `server/sync/{product,document}/manager.py`, `server/catalog/models/{product,document}.py` (`compute_content_hash`) |
| Reindex komutu | `server/catalog/management/commands/reindex.py` |
| Retrieval repoları (cosine + null guard) | `server/agent/core/repositories.py` |
| Retriever'lar | `server/agent/core/retrievers/{product,document}_retriever.py` |
| Kalite aşamaları | `server/agent/core/retrievers/{filters,hybrid,reranking,diversity}.py` |
| Varsayılan retriever config | `server/agent/core/agents/default_config.py` |
| Sabit-512 zorlaması | `server/catalog/serializers.py`, `server/catalog/checks.py`, `frontend/.../data-set-form.tsx` |
| Derinlemesine tasarım notları | `server/agent/core/retrievers/RETRIEVAL_QUALITY.md` |

## Lisans

MIT — bkz. [`LICENSE.md`](../LICENSE.md). Enthusiast tamamen açık kaynaklıdır ve her zaman ücretsiz kalacaktır.
