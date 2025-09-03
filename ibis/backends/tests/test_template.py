from __future__ import annotations

import pytest

import ibis
from ibis.tests.tstring import t

tm = pytest.importorskip("pandas.testing")

five = ibis.literal(5)
world = ibis.literal("world")


@pytest.mark.notimpl(["polars"])
@pytest.mark.parametrize(
    ("template", "expected_result"),
    [
        (t("{five} + 3"), 8),
        (t("{five:.2f} + 3"), 8),  # format strings ignored
        (t("'hello ' || {world}"), "hello world"),
        (t("'hello ' || {world!r}"), "hello world"),  # conversion strings ignored
    ],
)
def test_scalar(con, template, expected_result):
    """Test that scalar template expressions execute correctly."""
    expr = ibis.sql_value(template)
    result = con.execute(expr)
    assert result == expected_result


@pytest.mark.notimpl(["polars"])
def test_column(con, alltypes):
    """Test template with column interpolation."""
    c = alltypes.int_col  # noqa: F841
    template = t("{c + 2} - 1")
    expr = ibis.sql_value(template)
    result = con.execute(expr)
    expected = con.execute(alltypes.int_col + 1)
    tm.assert_series_equal(result, expected, check_names=False)


def test_dialect():
    pa = pytest.importorskip("pyarrow")
    five = ibis.literal(5)  # noqa: F841
    template = t("CAST({five} AS REAL)")

    expr_sqlite = ibis.sql_value(template, dialect="sqlite")
    expr_default = ibis.sql_value(template)

    con_sqlite = ibis.sqlite.connect()
    result = con_sqlite.to_pyarrow(expr_default)
    assert result.type == pa.float32()
    assert result.as_py() == 5.0
    result = con_sqlite.to_pyarrow(expr_sqlite)
    assert result.type == pa.float64()
    assert result.as_py() == 5.0

    con_duckdb = ibis.duckdb.connect()
    result = con_duckdb.to_pyarrow(expr_default)
    assert result.type == pa.float32()
    assert result.as_py() == 5.0
    result = con_duckdb.to_pyarrow(expr_sqlite)
    assert result.type == pa.float64()
    assert result.as_py() == 5.0
