from __future__ import annotations

import ibis

from .conftest import add_date, tpch_test


@tpch_test
def test_tpc_h04(orders, lineitem):
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
    q = q.aggregate(order_count=orders.count())
    q = q.order_by([orders.o_orderpriority])
    return q
