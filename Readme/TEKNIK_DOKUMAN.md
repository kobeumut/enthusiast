# Teknik Doküman — RAG Altyapısı ve Eklenen Görevler

Bu doküman, bu fork'ta upstream Enthusiast üzerine eklenen **RAG (Retrieval-Augmented Generation)** çalışmalarını görev bazlı olarak teknik olarak belgeler. Her bölüm bir Multica görevine (YAZ-\*) karşılık gelir; değişikliğin amacı, dokunduğu dosyalar, davranış ve nasıl doğrulanacağı yer alır.

> Okuma sırası: önce [Genel Bakış](#genel-bakış) ve [Mimari Değişiklikler](#mimari-değişiklikler), ardından görevler kronolojik/mantıksal sırada. Kullanım odaklı özet için [`BeniOku.md`](./BeniOku.md) ve [`README.md`](./README.md).

---

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Mimari Değişiklikler](#mimari-değişiklikler)
- [Görevler](#görevler)
  - [YAZ-5 — Ürün aramasını pgvector retrieval ile destekle](#yaz-5)
  - [YAZ-6 — RAG reindex/retrieval yolunu güvenilir yap](#yaz-6)
  - [YAZ-7 — PostgreSQL pgvector RAG kurulumunu belgele ve doğrula](#yaz-7)
  - [YAZ-10 — İçerik değişmeyen öğeler için re-index'i atla (content hash)](#yaz-10)
  - [YAZ-11 — NULL embedding'leri retrieval'dan hariç tut](#yaz-11)
  - [YAZ-12 — Indexing'de batch embedding + provider/client yeniden kullanımı](#yaz-12)
  - [YAZ-14 — Sabit 512 embedding boyutunu uçtan uca zorla](#yaz-14)
  - [YAZ-15 — RAG kalite fazı: metadata filtre + hybrid/rerank/MMR + HNSW ayarı](#yaz-15)
  - [YAZ-18 — Reindex'i dayanıklı yap (izolasyon/özet/fail-fast/retry/resume)](#yaz-18)
  - [Ek — pgvector HNSW ANN indeksleri](#ek)
- [Test Katmanı](#test-katmanı)
- [Bilinen Sınırlar ve Takip Maddeleri](#bilinen-sınırlar-ve-takip-maddeleri)

---

## Genel Bakış

Upstream Enthusiast, e-ticaret için agentic AI çerçevesidir. Bu fork, **RAG indeksini PostgreSQL + pgvector içinde** tutacak şekilde, retrieval yolunu LLM-üretilen SQL'den **vektör kosinüs sıralamasına** taşıdı ve etrafına bir dizi güvenilirlik/verim/kalite iyileştirmesi inşa etti.

Toplam 9 takip edilen görev (YAZ-5, 6, 7, 10, 11, 12, 14, 15, 18) artı HNSW indeks eki. Hepsi aynı temaya hizmet eder: **veriyi bir kez doğru şekilde embed et, gereksiz yere yeniden embed etme, ve retrieval'i hem güvenilir hem de kalite olarak ayarlanabilir kıl.**

| Görev | Tür | Tek cümle |
|---|---|---|
| YAZ-5 | feat | Doğal-dil ürün aramasını LLM SQL yerine pgvector kosinüs sıralamasına taşı |
| YAZ-6 | feat | Null embedding guard'ı, sync de-dup, reindex hata raporlaması ekle |
| YAZ-7 | docs | pgvector kurulumunu belgele; `reindex` komutu + `index_all_products_task` ekle |
| YAZ-10 | feat | İçerik aynıysa re-index'i atlamak için canonical `content_hash` |
| YAZ-11 | fix | Vektör retrieval'da `embedding IS NULL` chunk'ları hariç tut |
| YAZ-12 | feat | Indexing'de batch embedding + provider/client yeniden kullanımı |
| YAZ-14 | feat | Sabit 512 embedding boyutunu uçtan uca (API + UI + check) zorla |
| YAZ-15 | feat | Opsiyonel retrieval-quality hattı: filtre + hybrid/RRF + rerank + MMR + HNSW `ef_search` |
| YAZ-18 | feat | Reindex komutunu dayanıklı yap: izolasyon, özet, fail-fast, retry, resume |

## Mimari Değişiklikler

### Veri modeli

- **Chunk tabloları** (`catalog.ProductContentChunk`, `catalog.DocumentChunk`): `embedding = pgvector.django.VectorField(dimensions=EMBEDDING_VECTOR_DIMENSIONS, null=True)`. `EMBEDDING_VECTOR_DIMENSIONS = 512` (`catalog/models/data_set.py`).
- **HNSW ANN indeksleri**: her iki chunk tablosunun `embedding` sütunu üzerinde HNSW indeksleri (cosine distance).
- **`content_hash`** (`CharField(max_length=64, null=True)`): `Product` ve `Document` üzerinde, embed edilen içerik alanlarının sha256digest'i. Migration `0015_product_document_content_hash.py` ile eklendi.

### Retrieval katmanı

- `server/agent/core/repositories.py` — chunk-distance sorguları (kosinüs, anahtar-kelime, hibrit), `embedding__isnull=False` guard'ı, opsiyonel `filters` ve runtime `ef_search`.
- `server/agent/core/retrievers/` — `ProductRetriever`, `DocumentRetriever` artık opsiyonel kalite aşamalarını (`filters.py`, `hybrid.py`, `reranking.py`, `diversity.py`) orkestre ediyor.

### Indexing katmanı

- `server/catalog/services.py` — `*EmbeddingGenerator.index_object` artık tek batch çağrıyla embed ediyor.
- `server/catalog/tasks.py` — `index_product_task` / `index_document_task` (hata loglama + re-raise), `index_all_products_task` / `index_all_documents_task` (data-set kapsamlı dağıtım).
- `server/catalog/management/commands/reindex.py` — dayanıklı ön plan backfill.
- `server/sync/{product,document}/manager.py` — `content_hash` ile re-index de-dup.

### Yapılandırma/güvenlik

- `server/catalog/serializers.py` — sabit 512 boyut zorlaması (create + update immutability).
- `server/catalog/checks.py` — `catalog.W001` backstop uyarısı.
- `frontend/.../data-set-form.tsx` — Vector Size salt-okunur, 512.

---

## Görevler

<a id="yaz-5"></a>
### YAZ-5 — Ürün aramasını pgvector retrieval ile destekle

**Amaç.** Doğal-dil ürün araması, daha önce LLM tarafından üretilen `WHERE` SQL koşullarıyla yapılıyordu (kirli, hata yapmaya meyilli, veriye bağlı). Bu görev, ürün aramasını doküman retrieval ile aynı **pgvector kosinüs sıralama** yoluna taşır.

**Değişiklik.** `ProductRetriever.find_products_matching_query` artık:
1. Sorguyu data set'in embedding sağlayıcı/modeli ile embed eder (`EmbeddingProviderRegistry.provider_for_dataset`).
2. `ProductContentChunk.embedding` üzerinde `CosineDistance` ile sıralar.
3. Distinct ürünleri, mevcut data set ile sınırlı, `number_of_products` kadar döndürür; mevcut `product_details_as_json` şeklinde serileştirir.

Ürün arama aracı (`enthusiast-agent-product-search`) zaten `find_products_matching_query` çağırdığı için otomatik beslenir. Açık raw-SQL yolu (`find_products_with_sql` / `ProductSQLSearchTool`) değişmeden kalır; doğal-dil arama artık üretilen SQL'e bağlı değil.

**Yan düzeltmeler (test sırasında ortaya çıkanlar).**
- `enthusiast_common.agents.config`: `AgentConfig` import'u `TYPE_CHECKING` altına alındı — `agent.core.retrievers` import'unu bloklayan gizli `config <-> agents` döngüsel import kırıldı.
- `catalog/tests`: embedding sütunları migration 0014'ten beri `vector(512)`; `test_pgvector_retrieval.py` ve `test_reindex_command.py` 3-boyutlu vektör kullanıyordu (DB reddediyor) → 512-boyut unit/constant vektörlere geçildi.
- `catalog/models/__init__.py`: küçük `ruff I001` (merge imports) düzeltmesi.

**Dosyalar.** `server/agent/core/retrievers/product_retriever.py`, `server/agent/core/agents/default_config.py`, `plugins/enthusiast_common/.../agents/config.py`, `server/catalog/tests/test_pgvector_retrieval.py`, `server/catalog/tests/test_reindex_command.py`, `server/agent/core/retrievers/tests/test_product_vector_retrieval.py` (yeni).

**Doğrulama.** `test_product_vector_retrieval.py`: mesafeye göre sıralama, sonuç yok, data set izolasyonu, ürün başına çok chunk'ta dedup, `number_of_products` limiti, `product_details_as_json` şekli, data set embedding config'inin yeniden kullanımı.

---

<a id="yaz-6"></a>
### YAZ-6 — RAG reindex/retrieval yolunu güvenilir yap

**Amaç.** Şema/model değişikliği veya başarısız bir index koşusundan sonra chunk satırları `embedding IS NULL` kalabiliyor; sync her öğeyi her seferinde yeniden indeksliyordu. Bu görev backfill + retrieval yolunu güvenilir ve gözlemlenebilir kılar.

**Değişiklikler.**
- **Retrieval guardrail**: `DjangoDocumentChunkRepository` / `DjangoProductChunkRepository` üç chunk-distance sorgusunun hepsine `embedding__isnull=False` ekledi. `NULL`'a cosine distance `NULL`'dur; bu koruma olmadan eski/kısmi chunk'lar sonuçlara sızıyordu.
- **Indexing hatası görünürlüğü**: `index_product_task` / `index_document_task` başlangıç + hatayı (object ve data_set id'leriyle) loglar ve Celery görevi FAILED işaretlesin diye **re-raise** eder (sessizce chunksuz bırakmak yerine).
- **`manage.py reindex` artık tek kötü öğede abort etmez**: her hatayı öğe tanımlayıcısıyla loglar, devam eder ve sonunda non-zero + hata özetiiyle çıkar (kısmi backfill sessiz kalmaz).
- **Sync'te gereksiz reindex önleme**: `ProductSyncManager` / `DocumentSyncManager` artık yalnızca yeni öğeler veya **embed edilen içeriği gerçekten değişen** öğeler (ürün name/description; doküman content) için index görevi kuyruğa alır. Embed edilmeyen katalog alanları re-index olmadan güncellenmeye devam eder.

**Dosyalar.** `server/agent/core/repositories.py`, `server/catalog/management/commands/reindex.py`, `server/catalog/tasks.py`, `server/sync/product/manager.py`, `server/sync/document/manager.py`, `docs/content/docs/management/vector-store.md`, ve testler.

**Doğrulama.** `test_tasks.py` (data-set kapsamlı dağıtım, re-raise), `test_pgvector_retrieval.py` (null hariç tutma), `test_reindex_command.py` (tek kötü öğe abort etmez, `CommandError` sonda). Sync de-dup davranışı daha sonra YAZ-10'da `sync/tests/test_sync_content_hash.py` ile değiştirildi/geliştirildi (eski `test_managers.py` kaldırıldı).

> Not: YAZ-6'nın sync de-dup mantığı, YAZ-10'da daha kesin **canonical content hash** yaklaşımıyla değiştirildi/geliştirildi.

---

<a id="yaz-7"></a>
### YAZ-7 — PostgreSQL pgvector RAG kurulumunu belgele ve doğrula

**Amaç.** pgvector tabanlı RAG deposu için somut, uygulamaya-uygun bir kılavuz; yeni geliştiriciler ayrı bir vektör DB kurmadan ürün/doküman retrieval'ı ayağa kaldırabilsin ve doğrulayabilsin.

**Değişiklikler.**
- Yeni docs sayfası `docs/content/docs/management/vector-store.md`: vektörler nerede yaşar (PostgreSQL'de pgvector), lokal yığın kurulumu, env değişkenleri, `0001 VectorExtension` migration'ı, data-set başına embedding config'i, sync → index hattı, backfill/reindex, kosinüs retrieval, sorun giderme (eksik eklenti, eksik API anahtarı, boyut uyumsuzluğu, indeksli chunk yok, worker çalışmıyor) ve SQL + shell içeren manuel QA kontrol listesi.
- `README`, `system-architecture` ve `data-sets` yeni sayfaya çapraz bağlantı veriyor; `docs/content/docs/management/_meta.ts` sayfayı listeye ekliyor.
- Belgenen backfill/reindex yolunun gerçek olması için `index_all_products_task` (ZATEN `index_all_documents_task`'ı yansıtır) ve `python manage.py reindex` komutu eklendi (Celery worker olmadan çalışır).

**Doğrulama.** `test_reindex_command.py` (backfill pgvector chunk'ları oluşturur/değiştirir; ürün/doküman kapsamı) ve `test_pgvector_retrieval.py` (ürün + doküman retrieval sonuçlarını data set başına `CosineDistance`'a göre sıralar).

---

<a id="yaz-10"></a>
### YAZ-10 — İçerik değişmeyen öğeler için re-index'i atla (content hash)

**Amaç.** No-op yeniden sync'ler (aynı içerik) daha önce `index_product_task` / `index_document_task`'ı koşulsuz kuyruğa alıyordu; her chunk'ı silip yeniden kuruyor ve embedding API'sini tekrar çalıştırıyordu — değişmeyen içerik için saf maliyet.

**Değişiklikler.** `Product` ve `Document`'a canonical `content_hash` (sha256, chunker/embedder'a giden içerik alanları üzerinden) eklendi:
- Product: `name, description, sku, properties, categories, price`
- Document: `title, content`

Sync yöneticileri artık `update_or_create` öncesi daha önce saklanan hash'e bakar ve mevcut bir öğenin içeriği bayt bayt aynıysa re-index görevini **atlar**. Yeni öğeler ve gerçekten değişen öğeler her zamanki gibi indekslenir; `null` hash'li eski satırlar ilk sync'te backfill edilir. `reindex` yönetim komutu değişmedi ve her zaman zorla yeniden indeksler.

`compute_content_hash` sınıf metodu `Product`/`Document` üzerinde deterministik bir hex digest üretir; saklanan alan `content_hash = CharField(max_length=64, null=True, blank=True)`. Migration: `catalog/migrations/0015_product_document_content_hash.py`.

**Dosyalar.** `server/catalog/models/{product,document}.py`, `server/sync/product/manager.py`, `server/sync/document/manager.py`, migration `0015`, `server/sync/tests/test_sync_content_hash.py`.

**Doğrulama.** `test_sync_content_hash.py` (mock'lu index görevleriyle): değişmeyen ikinci sync → enqueue yok; değişen içerik → enqueue; eski null hash → backfill.

---

<a id="yaz-11"></a>
### YAZ-11 — NULL embedding'leri retrieval'dan hariç tut

**Amaç.** Retrieval repository metodları yalnızca `CosineDistance` annotate edip ona göre sıralıyordu; `embedding` sütunu `null=True` olduğundan henüz embed edilmemiş satırlar mevcut. `NULL` embedding'ler sahte/`NULL` mesafeler üretip retrieval sonuçlarını ve sorgu planını kirletiyordu.

**Değişiklikler.** Üç metodun da `data_set` filtresinin yanına `embedding__isnull=False` eklendi:
- `DjangoDocumentChunkRepository.get_chunk_by_distance_for_data_set`
- `DjangoProductChunkRepository.get_chunk_by_distance_for_data_set`
- `DjangoProductChunkRepository.get_chunk_by_distance_and_keyword_for_data_set`

**Doğrulama.** `TestNullEmbeddingsAreExcludedFromRetrieval` üç yolu da (doküman vektör, ürün vektör, ürün hibrit anahtar-kelime+vektör) kapsar; filtre olmadan fail eder, filtreyle geçer; mevcut `test_pgvector_retrieval.py` davranışı korunur.

> Bu, YAZ-6'nın null guard'ının ayrı, odaklı bir takibi olarak ayrı bir PR'da geldi (commit `693a8a7`).

---

<a id="yaz-12"></a>
### YAZ-12 — Indexing'de batch embedding + provider/client yeniden kullanımı

**Amaç.** Her chunk için ayrı registry/provider çözümü ve ayrı embedding API çağrısı maliyetli ve yavaştı.

**Değişiklikler.**
- `enthusiast-common`: `EmbeddingProvider.generate_embeddings_batch()` eklendi; varsayılan impl `generate_embeddings`'i döngüye alır (geriye uyumlu).
- `enthusiast-model-openai`: gerçek batch impl — tek `OpenAI()` istemcisi, `input=[...]` listesi, sonuçlar API indeksine göre yeniden hizalanır, 2048'lik parçalara bölünür.
- `catalog/services.index_object`: registry/provider'ı chunk döngüsü dışında bir kez çözer ve tüm chunk içeriklerini tek batch çağrıyla embed eder.

**Dosyalar.** `plugins/enthusiast_common/.../registry/embeddings.py`, `plugins/enthusiast_model_openai/embedding.py`, `server/catalog/services.py`, testler.

**Doğrulama.** `test_embedding_batching.py` (OpenAI/ABC batch testleri), `test_embedding_indexing.py` (`services.index_object` unit — mock provider batch yolunu ve tek-istemciyi assert eder), `test_reindex_command.py` (sahte sağlayıcısına batch eklendi, yeşil kalır).

---

<a id="yaz-14"></a>
### YAZ-14 — Sabit 512 embedding boyutunu uçtan uca zorla

**Amaç.** Chunk tablosu embedding sütunları tek paylaşımlı `vector(EMBEDDING_VECTOR_DIMENSIONS=512)` sütunu (pgvector ANN indeksleri tek sabit boyut gerektirir). Ancak `DataSet.embedding_vector_dimensions` serbest bir tamsayı gibi davranıyordu: 512 dışı bir data set oluşturulabiliyor (chunk insert runtime'da çakışıyordu) ve mevcut data set'te embedding config düzenlemek saklı embedding'leri sessizce geçersiz kılıyordu.

**Değişiklikler.**
- **Backend**
  - `serializers`: `DataSetCreateSerializer` artık `embedding_vector_dimensions != 512`'yi net mesajla `400` olarak reddeder (provider kısıtlamalarından önce kontrol edilir). Yeni `DataSetUpdateSerializer`, mevcut data set'te embedding provider/model/dimensions değişikliklerini reddeder (immutable); değişmeyen değerlere ve embedding-dışı düzenlemelere izin verir.
  - `views`: `DataSetDetailView.patch` artık `DataSetUpdateSerializer` kullanır ve sessizce alan filtrelemek yerine embedding değişikliğinde `400` döner.
  - `models/data_set.py`: `embedding_vector_dimensions`'ın `EMBEDDING_VECTOR_DIMENSIONS`'a eşit olması ve oluşturma sonrası immutable olduğu belgelendi.
  - `checks.py`: `catalog.W001`, API'yi atlayan satırlar için savunma backstop'u olarak netleştirildi (serializer birincil koruma); davranış değişmedi.
- **Frontend**
  - `data-set-form.tsx`: Vector Size artık hem oluşturma hem düzenlemede her zaman 512'ye sabitli salt-okunur alan. Model-tabanlı vektör boyutu otomatik seçimi kaldırıldı; form varsayılan değeri sabit olduğundan her zaman 512 submit edilir.
- **Dokümanlar**: `data-sets.md`, `vector-store.md` boyutun platform geneli sabit, oluşturma sonrası immutable olduğu ve nasıl değiştirileceği (kod + migration + reindex) belgelendi.

**Dosyalar.** `server/catalog/serializers.py`, `server/catalog/views.py`, `server/catalog/models/data_set.py`, `server/catalog/checks.py`, `frontend/.../data-set-form.tsx`, docs, testler.

**Doğrulama.** `test_data_set_list_view.py` (kısıtlı vektör boyutu testleri 512'ye güncellendi, global-dışı boyut reddi + varsayılan fallback testleri eklendi); `test_data_set_detail_view.py` (yeni — PATCH immutability: provider/model/dimensions değişikliği reddi, değişmeyen değerlere ve embedding-dışı düzenlemelere izin).

---

<a id="yaz-15"></a>
### YAZ-15 — RAG kalite fazı: metadata filtre + hybrid/rerank/MMR + HNSW ayarı

**Amaç.** Saf-vektor baseline doğru ama kaliteyi masada bırakıyor: açık terimler (SKU, model no), örtük kapsam (kategori, fiyat, kaynak) ve yinelenen içerik (örtüşen chunk'lar) için. Bu faz, her retrieval-quality kaldıracını **açık, config-driven ve varsayılan kapalı** yapar; tarihsel davranış bir data set opt-in olmadan birebir korunur.

**Yeni strateji modülleri** (`server/agent/core/retrievers/`):
- `filters.py`: `RetrievalFilters` değer nesnesi + `Q`-koşul pushdown (ürün kategori/fiyat; doküman url/başlık), ranklamadan önce uygulanır.
- `hybrid.py`: **Reciprocal Rank Fusion** (RRF, k=60) vektör + tam-metin anahtar-kelime ranklist'lerini birleştirir — tam terim eşleşmelerini (SKU, kodlar) kurtarır.
- `reranking.py`: `BaseReranker` + `LexicalReranker`; aday havuzu üzerinde vektör benzerliğini leksikal kapsama ile harmanlar (cross-encoder takılabilir).
- `diversity.py`: aday embedding'leri üzerinde **Maximal Marginal Relevance**; yakın-kopya doküman bölümleri çeşitli olanları ezmesin.

**Repository** (`server/agent/core/repositories.py`):
- `get_chunk_by_distance_for_data_set`: opsiyonel `filters` + `ef_search` kazandı (`SET LOCAL hnsw.ef_search`, transaction kapsamlı).
- Yeni `get_chunks_by_keyword_for_data_set` (vektör+anahtar-kelime paritesi): `ts_rank` gürültüsünden kaçmak için `SearchVector @@ SearchQuery` ile filtreler.

**Aşamalar ve ayarlar (hepsi opt-in).**

| Aşama | Ayar (extra_kwargs) | Geçerli | Not |
|---|---|---|---|
| Metadata filtre | `filters=` (çağrı anında) | ürün + doküman | kategori/fiyat veya url/başlık |
| Hybrid (RRF k=60) | `hybrid_enabled: True` | ürün + doküman | vektör + tam-metin füzyonu |
| Rerank | `reranker_enabled: True` veya `reranker=<BaseReranker>` | ürün + doküman | `LexicalReranker` varsayılan; `vector_weight`/`lexical_weight` |
| MMR çeşitlilik | `mmr_enabled: True`, `mmr_lambda: 0..1` (vars. 0.5) | yalnızca doküman | 1.0 saf ilgi, 0.0 saf yenilik |
| HNSW ayarı | `ef_search: N` | ürün + doküman | `SET LOCAL hnsw.ef_search`; bugün production yolunda etkisiz (bkz. aşağıda) |

Aşamalar sabit bir sırada birleşir:

```
sorgu → embed → vektör ranklist (HNSW) → [hybrid] RRF → [rerank] leksikal → [MMR] çeşitlilik (doküman) → distinct/top-K
```

**Önemli bulgu — data-set filtre ↔ HNSW etkileşimi.** `EXPLAIN`, **production data-set kapsamlı retrieval sorgusunun HNSW indeksini kullanmadığını** gösterdi: chunk tablosunu ebeveynine join edip `data_set_id` ile filtreliyor — bu ilişkisel koşulu embedding HNSW indeksi hizmet edemiyor. Planlayıcı bu yüzden FK indeksinden erişilebilen chunk'ları tarayıp kosinüs mesafeye göre sıralar (`enable_seqscan = off` olsa bile). *(Saf NN sorgusu — data set join'i olmayan — HNSW indeksini kullanır; EXPLAIN testi bunu sabitler.)*

**Sonuç:** mevcut şemada `ef_search` yalnızca HNSW indeks plana girdiğinde etkili olur; bugün production data-set kapsamlı yol üzerinde etkisi yok. Mevcut şemada asıl kalite artışı **metadata filtre, hybrid RRF, rerank ve MMR**'dan gelir. Sağlam düzeltme (`data_set_id`'yi chunk tablolarına denormalize + data-set başına partial HNSW indeks, veya iki aşamalı retrieve-then-filter) takip olarak işaretlendi.

**Dosyalar.** `server/agent/core/retrievers/{filters,hybrid,reranking,diversity}.py`, `server/agent/core/retrievers/{product,document}_retriever.py`, `server/agent/core/repositories.py`, `plugins/enthusiast_common/.../repositories/base.py`, `RETRIEVAL_QUALITY.md` (yeni), ve kapsamlı testler/eval.

**Doğrulama (benchmark-driven).**
- 35 saf-modül unit testi (RRF, filtre, rerank, MMR).
- Entegrasyon: metadata filtreler, kalite aşamaları, defaults kontratı.
- `catalog/tests/test_pgvector_explain.py`: `EXPLAIN` HNSW kullanımını ve data-set-filtre ↔ HNSW etkileşimini karakterize eder.
- `agent/core/retrievers/eval/`: etiketli korpus üzerinde before/after Precision@K / Recall@K / MRR; her aşama hedef metriğini yükseltir. 289 test geçer; ruff temiz.

Detaylı tasarım notları: `server/agent/core/retrievers/RETRIEVAL_QUALITY.md`.

---

<a id="yaz-18"></a>
### YAZ-18 — Reindex'i dayanıklı yap (izolasyon/özet/fail-fast/retry/resume)

**Amaç.** Reindex backfill daha önce `queryset.iterator()` ile dolaşıp `index_object()`'i `try/except`'siz çağırıyordu; tek kötü öğe (embedding API hatası, aşırı büyük chunk, geçici ağ arızası) tüm koşuyu durduruyordu.

**Değişiklikler.** Tüm dayanıklılık mantığı yönetim komutuna confine edildi; `catalog/services.py`'daki `index_object` sözleşmesi dokunulmadı (T-BATCH uyumlu).
- **Öğe başına hata izolasyonu**: her öğe `try/except`'e sarılır; kalıcı hata kaydedilir, çalışma devam eder.
- **`--fail-fast`**: ilk kalıcı hatada dur ve non-zero çık (CI/debug).
- **Özet**: sonunda `X ok / Y fail` artı başarısız öğelerin listesini (ürün `entry_id` / doküman `url` ve pk) yazdırır.
- **Retry/backoff**: her öğe eksponansiyel backoff ile yeniden denenir (`--max-attempts` vars. 3; `--retry-backoff` vars. 1.0). Yeniden denemek güvenlidir çünkü `index_object` yeniden bölmeden önce eski chunk'ları siler.
- **Resume**: `--from-id` / `--limit` öğeleri pk sırasında işler (toplu backfill'ler için).

Mevcut davranış (`--data-set`/`--products`/`--documents`, `-v 2+` öğe başına log) korunur; varsayılan verbosity'de çıktı temiz kalır.

**Komut bayrakları.**

| Bayrak | Anlamı |
|---|---|
| `--data-set <id>` | Yalnızca bir data set'i reindex et (verilmezse tüm data set'ler) |
| `--products` / `--documents` | Yalnızca bir taraf |
| `--fail-fast` | İlk kalıcı hatada dur, non-zero çık |
| `--max-attempts <n>` | Öğe başına toplam deneme (ilk dahil, vars. 3) |
| `--retry-backoff <s>` | Eksponansiyel backoff taban saniye (vars. 1.0) |
| `--from-id <pk>` | pk >= bu değer olan öğeleri işle (resume) |
| `--limit <n>` | En fazla bu kadar öğe işle (resume ile birlikte) |
| `-v 2` | Öğe başına ilerleme logu |

**Dosyalar.** `server/catalog/management/commands/reindex.py`, `server/catalog/tests/test_reindex_command.py`, `docs/content/docs/management/vector-store.md`.

**Doğrulama.** 6 test: izolasyon, fail-fast, retry (başarı + tükenme), resume, temiz varsayılan çıktı.

---

<a id="ek"></a>
### Ek — pgvector HNSW ANN indeksleri

(Commit `b151507`, ayrı bir YAZ etiketi olmadan, RAG temel altyapısının parçası.)

**Amaç.** Saf kosinüs NN sorgusu için chunk embedding sütunlarında HNSW ANN indeksleri ekler; yaklaşık-en-yakın-komşu aramasını sabit-sıralamaya göre hızlandırır.

**Not.** Üretim sorgusunun bugün HNSW'i kullanmaması (bkz. [YAZ-15](#yaz-15)) bu indeksin değerini azaltmıyor; data-set denormalizasyon takibi indeksin production yolunda da kullanılmasını sağlayacak.

---

## Test Katmanı

Tüm RAG çalışması, mümkün olduğunca **gerçek pgvector PostgreSQL**'e karşı test edilir (vektör deposu mock'lanmaz).

| Katman | Dosya(lar) | Ne kanıtlar |
|---|---|---|
| Saf-modül unit | `agent/core/retrievers/tests/test_{hybrid_rrf,filters,reranking,diversity}.py` | RRF formülü, filtre çevirisi, rerank harmanı, MMR seçimi — hızlı, DB'siz |
| Vektör retrieval | `agent/core/retrievers/tests/test_product_vector_retrieval.py`, `catalog/tests/test_pgvector_retrieval.py` | Mesafeye göre sıralama, null hariç tutma, data-set izolasyonu, dedup, limit |
| Metadata filtreler | `agent/core/retrievers/tests/test_metadata_filters.py` | Filtre koşulları doğru pushdown edilir |
| Kalite aşamaları | `agent/core/retrievers/tests/test_retriever_quality_stages.py` | Her aşama sonucu istenen yöne taşır; defaults saf-vektör sırasını korur |
| EXPLAIN | `catalog/tests/test_pgvector_explain.py` | HNSW saf NN'de kullanılır; runtime `ef_search` kabul edilir; data-set-filtre etkileşimi karakterize edilir |
| Indexing | `catalog/tests/test_embedding_{indexing,batching}.py` | Batch yolu, tek-istemci, `index_object` |
| Sync de-dup | `sync/tests/test_sync_content_hash.py` | Aynı içerik → enqueue yok; değişen içerik → enqueue; null hash → backfill |
| Reindex | `catalog/tests/test_reindex_command.py` | İzolasyon, fail-fast, retry, resume, temiz çıktı |
| Eval harness | `agent/core/retrievers/eval/test_eval_retrieval_quality.py` | Etiketli korpus üzerinde before/after Precision@K / Recall@K / MRR |
| API zorlama | `catalog/tests/test_views/test_data_set_{list,detail}_view.py` | 512 boyut zorlaması + embedding immutability |

Toplam: 289+ test geçer; ruff temiz (server + plugins).

## Bilinen Sınırlar ve Takip Maddeleri

1. **HNSW, üretim yolunda kullanılmıyor.** Data-set kapsamlı sorgu (`data_set_id` join'i) HNSW'i atlıyor; `ef_search` bu yüzden üretimde etkisiz. Takip: `data_set_id`'yi chunk tablolarına denormalize et + data-set başına partial HNSW indeksi, veya iki aşamalı retrieve-then-filter. (`RETRIEVAL_QUALITY.md`'de belgelendi; EXPLAIN testi bu değişiklik gelince pozitife dönecek.)
2. **Embedding boyutu kodda sabit (512).** Farklı boyut için `EMBEDDING_VECTOR_DIMENSIONS` kod değişikliği + data migration + reindex gerekir.
3. **MMR yalnızca dokümanlar.** Ürün retriever'ında MMR yok (ürünler zaten distinct'e collapse ediliyor).
4. **Özel sağlayıcılar batch'ten faydalanmaz.** `generate_embeddings_batch` override edilmediği sürece varsayılan impl `generate_embeddings`'i döngüye alır.
