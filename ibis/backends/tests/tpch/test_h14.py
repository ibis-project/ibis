from __future__ import annotations

import pytest

import ibis

from .conftest import add_date, tpch_test


@tpch_test
@pytest.mark.notimpl(
    ["snowflake"],
    raises=AssertionError,
    reason="ibis doesn't preserve decimal types in aggregations",
)
@pytest.mark.xfail_version(
    duckdb=["sqlalchemy>=2"],
    trino=["sqlalchemy>=2"],
    reason="slightly different code is generated for sqlalchemy 2 for aggregations",
)
def test_tpc_h14(part, lineitem):
    """Promotion Effect Query (Q14)

    This query monitors the market response to a promotion such as TV
    advertisements or a special campaign."""

    DATE = "1995-09-01"

    q = lineitem
    q = q.join(part, lineitem.l_partkey == part.p_partkey)
    q = q.filter([q.l_shipdate >= ibis.date(DATE), q.l_shipdate < add_date(DATE, dm=1)])

    revenue = q.l_extendedprice * (1 - q.l_discount)
    promo_revenue = q.p_type.like("PROMO%").ifelse(revenue, 0)

    q = q.aggregate(promo_revenue=100 * promo_revenue.sum() / revenue.sum())
    return q
