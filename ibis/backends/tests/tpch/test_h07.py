from __future__ import annotations

import ibis

from .conftest import add_date, tpch_test


@tpch_test
def test_tpc_h07(supplier, lineitem, orders, customer, nation):
    "Volume Shipping Query (Q7)"
    NATION1 = "FRANCE"
    NATION2 = "GERMANY"
    DATE = "1995-01-01"

    q = supplier
    q = q.join(lineitem, supplier.s_suppkey == lineitem.l_suppkey)
    q = q.join(orders, orders.o_orderkey == lineitem.l_orderkey)
    q = q.join(customer, customer.c_custkey == orders.o_custkey)
    n1 = nation
    n2 = nation.view()
    q = q.join(n1, supplier.s_nationkey == n1.n_nationkey)
    q = q.join(n2, customer.c_nationkey == n2.n_nationkey)

    q = q[
        n1.n_name.name("supp_nation"),
        n2.n_name.name("cust_nation"),
        lineitem.l_shipdate,
        lineitem.l_extendedprice,
        lineitem.l_discount,
        lineitem.l_shipdate.year().name("l_year"),
        (lineitem.l_extendedprice * (1 - lineitem.l_discount)).name("volume"),
    ]

    q = q.filter(
        [
            ((q.cust_nation == NATION1) & (q.supp_nation == NATION2))
            | ((q.cust_nation == NATION2) & (q.supp_nation == NATION1)),
            q.l_shipdate.between(ibis.date(DATE), add_date(DATE, dy=2, dd=-1)),
        ]
    )

    gq = q.group_by(["supp_nation", "cust_nation", "l_year"])
    q = gq.aggregate(revenue=q.volume.sum())
    q = q.order_by(["supp_nation", "cust_nation", "l_year"])

    return q
