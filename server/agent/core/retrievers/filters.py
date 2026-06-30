"""Metadata filters applied as predicate pushdown *before* vector ranking.

The retrieval layer ranks chunks by cosine distance to an embedded query. Many real queries are
implicitly scoped ("red running shoes under $50", "manuals for the AC-2000"), and pushing those
predicates into the SQL filter — instead of filtering the final ranked list in Python – both
improves precision (filter-violating chunks never enter the ranking) and lets Postgres plan around
the predicate. This module owns the (deliberately small) filter vocabulary and the translation from
a :class:`RetrievalFilters` value to Django ``Q`` predicates for the product and document querysets.

Filters are optional everywhere (``None`` means "no predicate", the historical behaviour), and the
chosen fields are the ones already indexed/queried by retrieval:

* products – ``categories`` (free-form ``CharField``, matched case-insensitively so it works whether
  a source stores CSV or JSON) and ``price`` (numeric range).
* documents – ``url`` / ``title`` substring scopes.

Source plugins store ``categories`` as an opaque string, so matching is deliberately a substring
(``__icontains``) test rather than a structured join.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from django.db.models import Q


@dataclass(frozen=True)
class RetrievalFilters:
    """Optional pre-retrieval predicates, pushed into the chunk queryset as ``Q`` filters.

    Every field defaults to "no constraint"; a default-constructed instance applies no filter.
    """

    # Product scope — matched against ``Product`` fields via the chunk's ``product`` relation.
    categories: list[str] = field(default_factory=list)
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    # Document scope — substring tests against ``Document`` fields via the chunk's ``document``.
    url_contains: Optional[str] = None
    title_contains: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        """True when no predicate is set, i.e. the filters should be a no-op."""
        return not (
            self.categories
            or self.price_min is not None
            or self.price_max is not None
            or self.url_contains
            or self.title_contains
        )


def product_filter_q(filters: Optional[RetrievalFilters]) -> Optional[Q]:
    """Translate product-oriented filters into a Django ``Q`` over ``product__*`` fields.

    Args:
        filters: The filters to translate. ``None`` or an empty instance yields ``None`` so callers
            can skip annotating the queryset entirely.

    Returns:
        A ``Q`` combining category (any-of) and price-range predicates, or ``None`` when there is
        nothing to apply.
    """
    if filters is None or filters.is_empty:
        return None

    predicates: list[Q] = []
    if filters.categories:
        # ``categories`` is a free-form string (CSV/JSON depending on the source plugin), so match
        # case-insensitively on any of the requested category labels.
        category_q = Q()
        for category in filters.categories:
            category_q |= Q(product__categories__icontains=category)
        predicates.append(category_q)
    if filters.price_min is not None:
        predicates.append(Q(product__price__gte=filters.price_min))
    if filters.price_max is not None:
        predicates.append(Q(product__price__lte=filters.price_max))
    return _combine(predicates)


def document_filter_q(filters: Optional[RetrievalFilters]) -> Optional[Q]:
    """Translate document-oriented filters into a Django ``Q`` over ``document__*`` fields.

    Args:
        filters: The filters to translate. ``None`` or an empty instance yields ``None``.

    Returns:
        A ``Q`` combining url/title substring predicates, or ``None`` when there is nothing to apply.
    """
    if filters is None or filters.is_empty:
        return None

    predicates: list[Q] = []
    if filters.url_contains:
        predicates.append(Q(document__url__icontains=filters.url_contains))
    if filters.title_contains:
        predicates.append(Q(document__title__icontains=filters.title_contains))
    return _combine(predicates)


def _combine(predicates: list[Q]) -> Optional[Q]:
    """AND-merge a list of ``Q`` objects into one, returning ``None`` for an empty list."""
    if not predicates:
        return None
    combined = predicates[0]
    for predicate in predicates[1:]:
        combined &= predicate
    return combined
