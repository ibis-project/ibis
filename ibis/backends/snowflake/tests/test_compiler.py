from __future__ import annotations

import ibis
from ibis import _


def test_more_than_one_quantile(snapshot):
    tables = ibis.table(name="t", schema={"ROW_COUNT": "int"})

    expr = tables.aggregate(
        quantile_0_25=_.ROW_COUNT.quantile(0.25),
        quantile_0_75=_.ROW_COUNT.quantile(0.75),
    )

    sql = ibis.to_sql(expr, dialect="snowflake")
    snapshot.assert_match(sql, "two_quantiles.sql")
