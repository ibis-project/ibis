import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt

pytestmark = [
    pytest.mark.never(["mysql", "sqlite"], reason="No struct support"),
    pytest.mark.notyet(["impala"]),
    pytest.mark.notimpl(["datafusion", "pyspark"]),
]


fields = pytest.mark.parametrize("field", ["a", "b", "c"])


@pytest.mark.notimpl(["dask"])
@fields
def test_single_field(backend, struct, struct_df, field):
    expr = struct.abc[field]
    result = expr.execute()
    expected = struct_df.abc.map(
        lambda value: value[field] if isinstance(value, dict) else value
    ).rename(field)
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(["dask"])
def test_all_fields(struct, struct_df):
    result = struct.abc.execute()
    expected = struct_df.abc
    tm.assert_series_equal(result, expected)


_SIMPLE_DICT = dict(a=1, b="2", c=3.0)
_STRUCT_LITERAL = ibis.struct(
    _SIMPLE_DICT,
    type="struct<a: int64, b: string, c: float64>",
)
_NULL_STRUCT_LITERAL = ibis.NA.cast("struct<a: int64, b: string, c: float64>")


@pytest.mark.notimpl(["postgres"])
@pytest.mark.notyet(
    ["clickhouse"],
    reason="clickhouse doesn't support nullable nested types",
)
@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        param(
            _STRUCT_LITERAL.__getitem__,
            _SIMPLE_DICT.__getitem__,
            id="dict",
        ),
        param(
            _NULL_STRUCT_LITERAL.__getitem__,
            lambda _: None,
            id="null",
        ),
    ],
)
@fields
def test_literal(con, field, expr_fn, expected_fn):
    query = expr_fn(field)
    result = pd.Series([con.execute(query)]).replace(np.nan, None)
    expected = pd.Series([expected_fn(field)])
    tm.assert_series_equal(result, expected)


@pytest.mark.notimpl(["dask", "pandas", "postgres"])
def test_struct_column(con):
    t = con.table("functional_alltypes")
    df = t.execute()
    expr = ibis.struct(dict(a=t.string_col, b=1, c=t.int_col)).name("s")
    assert expr.type() == dt.Struct.from_dict(
        dict(a=dt.string, b=dt.int8, c=dt.int32)
    )
    result = expr.execute()
    expected = pd.Series(
        (dict(a=a, b=1, c=c) for a, c in zip(df.string_col, df.int_col)),
        name="s",
    )
    tm.assert_series_equal(result, expected)
