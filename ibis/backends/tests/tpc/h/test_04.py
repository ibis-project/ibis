from __future__ import annotations

import ibis
from ibis.backends.tests.tpc.conftest import add_date, tpc_test


@tpc_test("h")
def test_04(orders, lineitem):
    """Order Priority Checking Query (Q4)"""
    DATE = "1993-07-01"
    cond = (lineitem.l_orderkey == orders.o_orderkey) & (
        lineitem.l_commitdate < lineitem.l_receiptdate
    )
    q = orders.filter(
        [
            cond.any(),
            orders.o_orderdate >= ibis.date(DATE),
            orders.o_orderdate < add_date(DATE, dm=3),
        ]
    )
    q = q.group_by([orders.o_orderpriority])
    q = q.aggregate(order_count=lambda t: t.count())
    q = q.order_by([orders.o_orderpriority])
    return q
