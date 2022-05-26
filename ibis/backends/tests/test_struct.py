import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
from pytest import param

import ibis

pytestmark = [
    pytest.mark.never(["mysql", "sqlite"], reason="No struct support"),
    pytest.mark.notyet(["impala"]),
    pytest.mark.notimpl(["datafusion", "pyspark"]),
]


fields = pytest.mark.parametrize("field", ["a", "b", "c"])


@pytest.mark.notimpl(["dask"])
@fields
def test_single_field(backend, struct, struct_df, field):
    result = struct.abc[field].execute()
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
