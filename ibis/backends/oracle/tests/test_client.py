from __future__ import annotations

from datetime import date  # noqa: TCH003

import ibis
from ibis import udf


def test_ibis_is_not_defeated_by_statement_cache(con):
    con.execute(ibis.timestamp("2419-10-11 10:10:25").name("tmp"))
    con.execute(ibis.literal(0).name("tmp"))


def test_builtin_udf(con):
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
