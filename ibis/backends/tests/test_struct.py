import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
from pytest import param

import ibis


@pytest.mark.never(["mysql", "sqlite"], reason="No struct support")
@pytest.mark.notyet(["impala"])
@pytest.mark.notimpl(["dask", "datafusion", "pyspark"])
@pytest.mark.parametrize("field", ["a", "b", "c"])
def test_single_field(backend, struct, struct_df, field):
    result = struct.abc[field].execute()
    expected = struct_df.abc.map(
        lambda value: value[field] if isinstance(value, dict) else value
    ).rename(field)
    backend.assert_series_equal(result, expected)


@pytest.mark.never(["mysql", "sqlite"], reason="No struct support")
@pytest.mark.notyet(["impala"])
@pytest.mark.notimpl(["dask", "datafusion", "pyspark"])
def test_all_fields(struct, struct_df):
    result = struct.abc.execute()
    expected = struct_df.abc
    tm.assert_series_equal(result, expected)


_SIMPLE_DICT = dict(a=1, b="2", c=3.0)


@pytest.mark.never(["mysql", "sqlite"], reason="No struct support")
@pytest.mark.notyet(["impala"])
@pytest.mark.notimpl(
    ["clickhouse", "datafusion", "pyspark", "postgres", "duckdb"]
)
@pytest.mark.parametrize("field", ["a", "b", "c"])
@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        param(
            lambda field: ibis.struct(
                _SIMPLE_DICT,
                type="struct<a: int64, b: string, c: float64>",
            )[field],
            _SIMPLE_DICT.__getitem__,
            id="dict",
            marks=[pytest.mark.notimpl(["postgres"])],
        ),
        param(
            lambda field: ibis.literal(
                None, type="struct<a: int64, b: string, c: float64>"
            )[field],
            lambda _: None,
            id="null",
            marks=[pytest.mark.notimpl(["duckdb"])],
        ),
    ],
)
def test_literal(con, field, expr_fn, expected_fn):
    query = expr_fn(field)
    result = pd.Series([con.execute(query)]).replace(np.nan, None)
    expected = pd.Series([expected_fn(field)])
    tm.assert_series_equal(result, expected)
