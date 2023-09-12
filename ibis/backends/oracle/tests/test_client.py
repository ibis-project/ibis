from __future__ import annotations

from datetime import date  # noqa: TCH003

import pandas as pd
import pandas.testing as tm

import ibis
from ibis import udf


def test_ibis_is_not_defeated_by_statement_cache(con):
    con.execute(ibis.timestamp("2419-10-11 10:10:25").name("tmp"))
    con.execute(ibis.literal(0).name("tmp"))


def test_builtin_scalar_udf(con):
    @udf.scalar.builtin
    def to_date(a: str, fmt: str) -> date:
        """Convert a string to a date."""

    @udf.scalar.builtin
    def months_between(a: date, b: date) -> int:
        """Months between two dates."""

    date_fmt = "YYYY-MM-DD"
    expr = months_between(
        to_date("2019-12-11", date_fmt), to_date("2019-10-01", date_fmt)
    )
    result = con.execute(expr)
    assert result == 2


def test_builtin_agg_udf(con):
    @udf.agg.builtin
    def stats_one_way_anova(x, y, value: str) -> int:
        """Convert a string to a date."""

    ft = con.tables.functional_alltypes
    expr = (
        ft.group_by("string_col")
        .agg(df_w=stats_one_way_anova(ft.double_col, ft.int_col, "DF_WITHIN"))
        .order_by("string_col")
    )
    result = expr.execute()

    sql = """
    SELECT
      "string_col",
      stats_one_way_anova("double_col", "int_col", 'DF_WITHIN') as "df_w"
    FROM "functional_alltypes"
    GROUP BY "string_col"
    ORDER BY "string_col"
    """
    with con.begin() as c:
        expected = pd.DataFrame(
            c.exec_driver_sql(sql).fetchall(), columns=["string_col", "df_w"]
        )
    tm.assert_frame_equal(result, expected)
