from __future__ import annotations

import pandas.testing as tm
import polars as pl
import pytest

import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis import udf
from ibis.legacy.udf.vectorized import elementwise, reduction

pytest.importorskip("polars")
pc = pytest.importorskip("pyarrow.compute")


@elementwise(input_type=["string"], output_type="int64")
def my_string_length(arr, **kwargs):
    return pl.from_arrow(
        pc.cast(pc.multiply(pc.utf8_length(arr.to_arrow()), 2), target_type="int64")
    )


@elementwise(input_type=[dt.int64, dt.int64], output_type=dt.int64)
def my_add(arr1, arr2, **kwargs):
    return pl.from_arrow(pc.add(arr1.to_arrow(), arr2.to_arrow()))


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
    expected = (df.smallint_col + df.int_col).astype("int32")

    tm.assert_series_equal(result, expected.rename("tmp"))


@pytest.mark.parametrize(
    ("value", "expected"), [(8, 2), (27, 3), (7, 7 ** (1.0 / 3.0))]
)
def test_builtin_scalar_udf(con, value, expected):
    @udf.scalar.builtin
    def cbrt(a: float) -> float:
        ...

    expr = cbrt(value)
    result = con.execute(expr)
    assert pytest.approx(result) == expected


@udf.scalar.pyarrow
def string_length(x: str) -> int:
    return pc.cast(pc.multiply(pc.utf8_length(x), 2), target_type="int64")


@udf.scalar.python
def string_length_python(x: str) -> int:
    return len(x) * 2


@udf.scalar.pyarrow
def add(x: int, y: int) -> int:
    return pc.add(x, y)


@udf.scalar.python
def add_python(x: int, y: int) -> int:
    return x + y


@pytest.mark.parametrize("func", [string_length, string_length_python])
def test_scalar_udf(alltypes, func):
    data_string_col = alltypes.date_string_col.execute()
    expected = data_string_col.str.len() * 2

    expr = func(alltypes.date_string_col)
    assert isinstance(expr, ir.Column)

    result = expr.execute()
    tm.assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize("func", [add, add_python])
def test_multiple_argument_scalar_udf(alltypes, func):
    expr = func(alltypes.smallint_col, alltypes.int_col).name("tmp")
    result = expr.execute()

    df = alltypes[["smallint_col", "int_col"]].execute()
    expected = (df.smallint_col + df.int_col).astype("int64")

    tm.assert_series_equal(result, expected.rename("tmp"))


def test_builtin_agg_udf(con):
    @udf.agg.builtin
    def approx_n_unique(a, where: bool = True) -> int:
        ...

    ft = con.tables.functional_alltypes
    expr = approx_n_unique(ft.string_col)
    result = con.execute(expr)
    assert result == 10

    expr = approx_n_unique(ft.string_col, where=ft.string_col == "1")
    result = con.execute(expr)
    assert result == 1
