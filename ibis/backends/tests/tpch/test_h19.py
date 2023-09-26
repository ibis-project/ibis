from __future__ import annotations

from .conftest import tpch_test


@tpch_test
def test_tpc_h19(lineitem, part):
    """Discounted Revenue Query (Q19)

    The Discounted Revenue Query reports the gross discounted revenue
    attributed to the sale of selected parts handled in a particular manner.
    This query is an example of code such as might be produced programmatically
    by a data mining tool."""

    QUANTITY1 = 1
    QUANTITY2 = 10
    QUANTITY3 = 20
    BRAND1 = "Brand#12"
    BRAND2 = "Brand#23"
    BRAND3 = "Brand#34"

    q = lineitem.join(part, part.p_partkey == lineitem.l_partkey)

    q1 = (
        (q.p_brand == BRAND1)
        & (q.p_container.isin(("SM CASE", "SM BOX", "SM PACK", "SM PKG")))
        & (q.l_quantity >= QUANTITY1)
        & (q.l_quantity <= QUANTITY1 + 10)
        & (q.p_size.between(1, 5))
        & (q.l_shipmode.isin(("AIR", "AIR REG")))
        & (q.l_shipinstruct == "DELIVER IN PERSON")
    )

    q2 = (
        (q.p_brand == BRAND2)
        & (q.p_container.isin(("MED BAG", "MED BOX", "MED PKG", "MED PACK")))
        & (q.l_quantity >= QUANTITY2)
        & (q.l_quantity <= QUANTITY2 + 10)
        & (q.p_size.between(1, 10))
        & (q.l_shipmode.isin(("AIR", "AIR REG")))
        & (q.l_shipinstruct == "DELIVER IN PERSON")
    )

    q3 = (
        (q.p_brand == BRAND3)
        & (q.p_container.isin(("LG CASE", "LG BOX", "LG PACK", "LG PKG")))
        & (q.l_quantity >= QUANTITY3)
        & (q.l_quantity <= QUANTITY3 + 10)
        & (q.p_size.between(1, 15))
        & (q.l_shipmode.isin(("AIR", "AIR REG")))
        & (q.l_shipinstruct == "DELIVER IN PERSON")
    )

    q = q.filter([q1 | q2 | q3])
    q = q.aggregate(revenue=(q.l_extendedprice * (1 - q.l_discount)).sum())
    return q
