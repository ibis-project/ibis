from __future__ import annotations

import ibis

from .conftest import add_date, tpch_test


@tpch_test
def test_tpc_h10(customer, orders, lineitem, nation):
    """Returned Item Reporting Query (Q10)"""
    DATE = "1993-10-01"

    q = customer
    q = q.join(orders, customer.c_custkey == orders.o_custkey)
    q = q.join(lineitem, lineitem.l_orderkey == orders.o_orderkey)
    q = q.join(nation, customer.c_nationkey == nation.n_nationkey)

    q = q.filter(
        [
            (q.o_orderdate >= ibis.date(DATE)) & (q.o_orderdate < add_date(DATE, dm=3)),
            q.l_returnflag == "R",
        ]
    )

    gq = q.group_by(
        [
            q.c_custkey,
            q.c_name,
            q.c_acctbal,
            q.n_name,
            q.c_address,
            q.c_phone,
            q.c_comment,
        ]
    )
    q = gq.aggregate(revenue=(q.l_extendedprice * (1 - q.l_discount)).sum()).relocate(
        "revenue", after="c_name"
    )

    q = q.order_by(ibis.desc(q.revenue))
    return q.limit(20)
