from __future__ import annotations

import ibis

from .conftest import add_date, tpch_test


@tpch_test
def test_tpc_h15(lineitem, supplier):
    """Top Supplier Query (Q15)"""

    DATE = "1996-01-01"

    qrev = lineitem
    qrev = qrev.filter(
        [
            lineitem.l_shipdate >= ibis.date(DATE),
            lineitem.l_shipdate < add_date(DATE, dm=3),
        ]
    )

    gqrev = qrev.group_by([lineitem.l_suppkey])
    qrev = gqrev.aggregate(
        total_revenue=(qrev.l_extendedprice * (1 - qrev.l_discount)).sum()
    )

    q = supplier.join(qrev, supplier.s_suppkey == qrev.l_suppkey)
    q = q.filter([q.total_revenue == qrev.total_revenue.max()])
    q = q.order_by([q.s_suppkey])
    q = q[q.s_suppkey, q.s_name, q.s_address, q.s_phone, q.total_revenue]
    return q
