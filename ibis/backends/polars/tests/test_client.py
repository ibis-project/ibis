from __future__ import annotations

import polars as pl
import polars.testing
import pytest

import ibis
from ibis.backends.tests.errors import PolarsSQLInterfaceError
from ibis.formats.polars import PolarsType
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


def test_memtable_polars_types(con):
    # Check that we can create a memtable with some polars-specific types,
    # and that those columns then work in downstream operations
    df = pl.DataFrame(
        {
            "x": ["a", "b", "a"],
            "y": ["c", "d", "c"],
            "z": ["e", "f", "e"],
        },
        schema={
            "x": pl.String,
            "y": pl.Categorical,
            "z": pl.Enum(["e", "f"]),
        },
    )
    t = ibis.memtable(df)
    res = con.to_polars((t.x + t.y + t.z).name("test"))
    sol = (df["x"] + df["y"] + df["z"]).rename("test")
    pl.testing.assert_series_equal(res, sol)


@pytest.mark.parametrize("to_method", ["to_pyarrow", "to_polars"])
def test_streaming(con, mocker, to_method):
    t = con.table("functional_alltypes")
    mocked_collect = mocker.patch("polars.LazyFrame.collect")
    getattr(con, to_method)(t, engine="streaming")
    mocked_collect.assert_called_once_with(engine="streaming")


@pytest.mark.parametrize("to_method", ["to_pyarrow", "to_polars"])
def test_engine(con, mocker, to_method):
    t = con.table("functional_alltypes")
    mocked_collect = mocker.patch("polars.LazyFrame.collect")
    getattr(con, to_method)(t, engine="gpu")
    mocked_collect.assert_called_once_with(engine="gpu")


def test_compile_with_memtable(con):
    t = ibis.memtable({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = con.compile(t)
    assert isinstance(result, pl.LazyFrame)


@pytest.mark.parametrize(
    ("dtype", "fn", "expected"),
    [
        (pl.Int8, lambda t: 100 * t.x, [2500, 3000, -1500]),
        (pl.Int8, lambda t: t.x + 120, [145, 150, 105]),
        (pl.Int8, lambda t: t.x - (-120), [145, 150, 105]),
        (pl.Int8, lambda t: t.x * t.y, [2500, 3000, -1500]),
        (pl.Int16, lambda t: t.x * 2000, [50_000, 60_000, -30_000]),
        (pl.Int16, lambda t: t.x + 32_760, [32_785, 32_790, 32_745]),
        (pl.Int16, lambda t: t.x - (-32_760), [32_785, 32_790, 32_745]),
    ],
)
def test_narrow_integer_arithmetic_is_widened(con, dtype, fn, expected):
    # https://github.com/ibis-project/ibis/issues/11749
    df = pl.DataFrame(
        {"x": [25, 30, -15], "y": [100, 100, 100]}, schema={"x": dtype, "y": dtype}
    )
    t = ibis.memtable(df)
    expr = fn(t).name("r")

    # check the polars expression itself, before ibis casts the result
    lf = con.compile(expr)
    assert lf.collect_schema()["r"] == PolarsType.from_ibis(expr.type())
    assert lf.collect()["r"].to_list() == expected
