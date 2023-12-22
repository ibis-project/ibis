from __future__ import annotations

import pytest
import sqlalchemy as sa

import ibis

from .conftest import tpch_test


@tpch_test
@pytest.mark.broken(
    ["snowflake"],
    reason="ibis generates overlapping aliases",
    raises=sa.exc.CompileError,
)
def test_tpc_h21(supplier, lineitem, orders, nation):
    """Suppliers Who Kept Orders Waiting Query (Q21)

    This query identifies certain suppliers who were not able to ship required
    parts in a timely manner."""
    NATION = "SAUDI ARABIA"

    L2 = lineitem.view()
    L3 = lineitem.view()

    q = supplier
    q = q.join(lineitem, supplier.s_suppkey == lineitem.l_suppkey)
    q = q.join(orders, orders.o_orderkey == lineitem.l_orderkey)
    q = q.join(nation, supplier.s_nationkey == nation.n_nationkey)
    q = q[
        q.l_orderkey.name("l1_orderkey"),
        q.o_orderstatus,
        q.l_receiptdate,
        q.l_commitdate,
        q.l_suppkey.name("l1_suppkey"),
        q.s_name,
        q.n_name,
    ]
    q = q.filter(
        [
            q.o_orderstatus == "F",
            q.l_receiptdate > q.l_commitdate,
            q.n_name == NATION,
            ((L2.l_orderkey == q.l1_orderkey) & (L2.l_suppkey != q.l1_suppkey)).any(),
            ~(
                (
                    (L3.l_orderkey == q.l1_orderkey)
                    & (L3.l_suppkey != q.l1_suppkey)
                    & (L3.l_receiptdate > L3.l_commitdate)
                ).any()
            ),
        ]
    )

    gq = q.group_by([q.s_name])
    q = gq.aggregate(numwait=q.count())
    q = q.order_by([ibis.desc(q.numwait), q.s_name])
    return q.limit(100)
