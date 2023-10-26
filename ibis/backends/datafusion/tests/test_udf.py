from __future__ import annotations

import pandas.testing as tm
import pytest

import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis import udf
from ibis.legacy.udf.vectorized import elementwise, reduction

pytest.importorskip("datafusion")
pc = pytest.importorskip("pyarrow.compute")


@elementwise(input_type=["string"], output_type="int64")
def my_string_length(arr, **kwargs):
    # arr is a pyarrow.StringArray
    return pc.cast(pc.multiply(pc.utf8_length(arr), 2), target_type="int64")


@elementwise(input_type=[dt.int64, dt.int64], output_type=dt.int64)
def my_add(arr1, arr2, **kwargs):
    return pc.add(arr1, arr2)


@reduction(input_type=[dt.float64], output_type=dt.float64)
def my_mean(arr):
    return pc.mean(arr)


def test_udf(alltypes):
    data_string_col = alltypes.date_string_col.execute()
    expected = data_string_col.str.len() * 2

    expr = my_string_length(alltypes.date_string_col)
    assert isinstance(expr, ir.Column)

    result = expr.execute()
    tm.assert_series_equal(result, expected, check_names=False)


def test_multiple_argument_udf(alltypes):
    expr = my_add(alltypes.smallint_col, alltypes.int_col).name("tmp")
    result = expr.execute()

    df = alltypes[["smallint_col", "int_col"]].execute()
    expected = (df.smallint_col + df.int_col).astype("int64")

    tm.assert_series_equal(result, expected.rename("tmp"))


def test_builtin_scalar_udf(con):
    @udf.scalar.builtin
    def to_hex(a: int) -> str:
        """Convert an integer to a hex string."""

    expr = to_hex(42)
    result = con.execute(expr)
    assert result == "2a"


def test_builtin_agg_udf(con):
    @udf.agg.builtin
    def median(a: float) -> float:
        """Median of a column."""

    expr = median(con.tables.batting.G)
    result = con.execute(expr)
    assert result == con.tables.batting.G.execute().median()


def test_builtin_agg_udf_filtered(con):
    @udf.agg.builtin
    def median(a: float, where: bool = True) -> float:
        """Median of a column."""

    median(con.tables.batting.G).execute()
