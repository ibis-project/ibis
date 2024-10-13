from __future__ import annotations

import ibis


def test_rank_no_window_frame(con, snapshot):
    t = ibis.table(schema=dict(color=str, price=int), name="diamonds_sample")
    expr = t.mutate(ibis.rank().over(group_by="color", order_by="price"))
    sql = ibis.to_sql(expr, dialect="mssql")

    snapshot.assert_match(sql, "out.sql")
