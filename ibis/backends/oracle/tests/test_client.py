from __future__ import annotations

from datetime import date  # noqa: TC003

import oracledb
import pandas as pd
import pandas.testing as tm
import pytest

import ibis
from ibis import udf
from ibis.backends.oracle.tests.conftest import (
    ORACLE_HOST,
    ORACLE_PASS,
    ORACLE_USER,
)


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
            c.execute(sql).fetchall(), columns=["string_col", "df_w"]
        )
    tm.assert_frame_equal(result, expected, check_dtype=False)


def test_list_tables(con):
    assert con.list_tables()

    assert con.list_tables(database="SYS", like="EXU8OPT") == ["EXU8OPT"]


def test_from_url():
    new_con = ibis.connect("oracle://ibis:ibis@localhost:1521/IBIS_TESTING")

    result = new_con.sql('SELECT 1 AS "a"').to_pandas()
    assert result.a.iat[0] == 1


def test_invalid_port():
    port = 9999
    url = f"oracle://{ORACLE_USER}:{ORACLE_PASS}@{ORACLE_HOST}:{port}/IBIS_TESTING"
    with pytest.raises(
        oracledb.OperationalError,
        match="DPY-6005: cannot connect to database",
    ):
        ibis.connect(url)
