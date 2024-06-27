from __future__ import annotations

import ibis
from ibis.backends.tests.tpc.conftest import tpc_test


@tpc_test("h")
def test_03(customer, orders, lineitem):
    """Shipping Priority Query (Q3)"""
    MKTSEGMENT = "BUILDING"
    DATE = ibis.date("1995-03-15")

    q = customer.join(orders, customer.c_custkey == orders.o_custkey)
    q = q.join(lineitem, lineitem.l_orderkey == orders.o_orderkey)
    q = q.filter(
        [q.c_mktsegment == MKTSEGMENT, q.o_orderdate < DATE, q.l_shipdate > DATE]
    )
    qg = q.group_by([q.l_orderkey, q.o_orderdate, q.o_shippriority])
    q = qg.aggregate(revenue=(q.l_extendedprice * (1 - q.l_discount)).sum()).relocate(
        "revenue", after="l_orderkey"
    )
    q = q.order_by([ibis.desc(q.revenue), q.o_orderdate])
    q = q.limit(10)

    return q
