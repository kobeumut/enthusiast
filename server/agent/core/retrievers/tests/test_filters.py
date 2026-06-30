"""Unit tests for the metadata-filter value object (``agent.core.retrievers.filters``).

The ``*_filter_q`` translators are thin, so this file only checks their None/empty short-circuit and
return type. The actual predicate correctness (does the category/price predicate keep the right
rows?) is verified end-to-end in ``test_metadata_filters.py`` against real pgvector data.
"""

from django.db.models import Q

from agent.core.retrievers.filters import RetrievalFilters, document_filter_q, product_filter_q


class TestRetrievalFilters:
    def test_default_instance_is_empty(self):
        assert RetrievalFilters().is_empty is True

    def test_is_empty_false_when_any_field_set(self):
        assert RetrievalFilters(categories=["shoes"]).is_empty is False
        assert RetrievalFilters(price_max=50).is_empty is False
        assert RetrievalFilters(price_min=10).is_empty is False
        assert RetrievalFilters(url_contains="manual").is_empty is False
        assert RetrievalFilters(title_contains="ac2000").is_empty is False


class TestFilterTranslatorsShortCircuit:
    def test_product_none_or_empty_returns_none(self):
        assert product_filter_q(None) is None
        assert product_filter_q(RetrievalFilters()) is None

    def test_document_none_or_empty_returns_none(self):
        assert document_filter_q(None) is None
        assert document_filter_q(RetrievalFilters()) is None

    def test_product_set_filters_return_a_q(self):
        assert isinstance(product_filter_q(RetrievalFilters(categories=["shoes"])), Q)
        assert isinstance(product_filter_q(RetrievalFilters(price_max=99)), Q)

    def test_document_set_filters_return_a_q(self):
        assert isinstance(document_filter_q(RetrievalFilters(url_contains="m")), Q)
        assert isinstance(document_filter_q(RetrievalFilters(title_contains="m")), Q)
