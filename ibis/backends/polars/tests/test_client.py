from __future__ import annotations

import pytest

from ibis.backends.tests.errors import PolarsSQLInterfaceError
from ibis.util import gen_name


def test_cannot_run_sql_after_drop(con):
    t = con.table("functional_alltypes")
    n = t.count().execute()

    name = gen_name("polars_dot_sql")
    con.create_table(name, t)

    sql = f"SELECT COUNT(*) FROM {name}"

    expr = con.sql(sql)
    result = expr.execute()
    assert result.iat[0, 0] == n

    con.drop_table(name)
    with pytest.raises(PolarsSQLInterfaceError):
        con.sql(sql)
