from __future__ import annotations

import ibis

from .conftest import tpch_test


@tpch_test
def test_tpc_h18(customer, orders, lineitem):
    """Large Volume Customer Query (Q18)

    The Large Volume Customer Query ranks customers based on their having
    placed a large quantity order. Large quantity orders are defined as those
    orders whose total quantity is above a certain level."""

    QUANTITY = 300

    subgq = lineitem.group_by([lineitem.l_orderkey])
    subq = subgq.aggregate(qty_sum=lineitem.l_quantity.sum())
    subq = subq.filter([subq.qty_sum > QUANTITY])

    q = customer
    q = q.join(orders, customer.c_custkey == orders.o_custkey)
    q = q.join(lineitem, orders.o_orderkey == lineitem.l_orderkey)
    q = q.filter([q.o_orderkey.isin(subq.l_orderkey)])

    gq = q.group_by(
        [q.c_name, q.c_custkey, q.o_orderkey, q.o_orderdate, q.o_totalprice]
    )
    q = gq.aggregate(sum_qty=q.l_quantity.sum())
    q = q.order_by([ibis.desc(q.o_totalprice), q.o_orderdate])
    return q.limit(100)
