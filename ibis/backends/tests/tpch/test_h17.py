from __future__ import annotations

import pytest

from .conftest import tpch_test


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
def test_tpc_h17(lineitem, part):
    """Small-Quantity-Order Revenue Query (Q17)

    This query determines how much average yearly revenue would be lost if
    orders were no longer filled for small quantities of certain parts. This
    may reduce overhead expenses by concentrating sales on larger shipments."""
    BRAND = "Brand#23"
    CONTAINER = "MED BOX"

    q = lineitem.join(part, part.p_partkey == lineitem.l_partkey)

    innerq = lineitem
    innerq = innerq.filter([innerq.l_partkey == q.p_partkey])

    q = q.filter(
        [
            q.p_brand == BRAND,
            q.p_container == CONTAINER,
            q.l_quantity < (0.2 * innerq.l_quantity.mean()),
        ]
    )
    q = q.aggregate(avg_yearly=q.l_extendedprice.sum() / 7.0)
    return q
