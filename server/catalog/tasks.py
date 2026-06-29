import logging

from celery import shared_task

from .models import DataSet, Document, Product
from .services import DocumentEmbeddingGenerator, ProductEmbeddingGenerator

logger = logging.getLogger(__name__)


@shared_task
def index_document_task(document_id: int):
    document = Document.objects.get(id=document_id)
    logger.info("Indexing document id=%s (data_set=%s)", document_id, document.data_set_id)
    try:
        DocumentEmbeddingGenerator.index_object(document)
    except Exception:
        # Log with full context and re-raise so Celery marks the task FAILED and the failure is
        # visible (Flower/logs/reindex command) instead of silently leaving chunks without embeddings.
        logger.exception("Failed to index document id=%s (data_set=%s)", document_id, document.data_set_id)
        raise


@shared_task
def index_all_documents_task(data_set_id: int):
    data_set = DataSet.objects.get(id=data_set_id)
    document_ids = list(data_set.documents.values_list("id", flat=True))
    logger.info("Queuing indexing for %s documents (data_set=%s)", len(document_ids), data_set_id)
    for document_id in document_ids:
        index_document_task.apply_async([document_id])


@shared_task
def index_product_task(product_id: int):
    product = Product.objects.get(id=product_id)
    logger.info("Indexing product id=%s (data_set=%s)", product_id, product.data_set_id)
    try:
        ProductEmbeddingGenerator.index_object(product)
    except Exception:
        logger.exception("Failed to index product id=%s (data_set=%s)", product_id, product.data_set_id)
        raise


@shared_task
def index_all_products_task(data_set_id: int):
    data_set = DataSet.objects.get(id=data_set_id)
    product_ids = list(data_set.products.values_list("id", flat=True))
    logger.info("Queuing indexing for %s products (data_set=%s)", len(product_ids), data_set_id)
    for product_id in product_ids:
        index_product_task.apply_async([product_id])
