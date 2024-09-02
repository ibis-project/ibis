from __future__ import annotations

import pytest

import ibis
from ibis.backends.tests.errors import PolarsSQLInterfaceError
from ibis.util import gen_name

pd = pytest.importorskip("pandas")
tm = pytest.importorskip("pandas.testing")


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


def test_array_flatten(con):
    data = {"id": range(3), "happy": [[["abc"]], [["bcd"]], [["def"]]]}
    t = ibis.memtable(data)
    expr = t.select("id", flat=t.happy.flatten()).order_by("id")
    result = con.to_pyarrow(expr)
    expected = pd.DataFrame(
        {"id": data["id"], "flat": [row[0] for row in data["happy"]]}
    )
    tm.assert_frame_equal(result.to_pandas(), expected)
