from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest

import ibis
import ibis.expr.datatypes as dt

pytestmark = [
    pytest.mark.never(["mysql", "sqlite", "mssql"], reason="No struct support"),
    pytest.mark.notyet(["impala"]),
    pytest.mark.notimpl(["datafusion", "druid", "oracle"]),
]


@pytest.mark.notimpl(["dask", "snowflake"])
@pytest.mark.parametrize("field", ["a", "b", "c"])
def test_single_field(backend, struct, struct_df, field):
    expr = struct.abc[field]
    result = expr.execute().sort_values().reset_index(drop=True)
    expected = (
        struct_df.abc.map(
            lambda value: value[field] if isinstance(value, dict) else value
        )
        .rename(field)
        .sort_values()
        .reset_index(drop=True)
    )
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(["dask"])
def test_all_fields(struct, struct_df):
    result = struct.abc.execute()
    expected = struct_df.abc

    assert {
        row if not isinstance(row, Mapping) else tuple(row.items()) for row in result
    } == {
        row if not isinstance(row, Mapping) else tuple(row.items()) for row in expected
    }


_SIMPLE_DICT = dict(a=1, b="2", c=3.0)
_STRUCT_LITERAL = ibis.struct(
    _SIMPLE_DICT,
    type="struct<a: int64, b: string, c: float64>",
)
_NULL_STRUCT_LITERAL = ibis.NA.cast("struct<a: int64, b: string, c: float64>")


@pytest.mark.notimpl(["postgres"])
@pytest.mark.parametrize("field", ["a", "b", "c"])
def test_literal(con, field):
    query = _STRUCT_LITERAL[field]
    dtype = query.type().to_pandas()
    result = pd.Series([con.execute(query)], dtype=dtype)
    result = result.replace({np.nan: None})
    expected = pd.Series([_SIMPLE_DICT[field]])
    tm.assert_series_equal(result, expected.astype(dtype))


@pytest.mark.notimpl(["postgres"])
@pytest.mark.parametrize("field", ["a", "b", "c"])
@pytest.mark.notyet(
    ["clickhouse"], reason="clickhouse doesn't support nullable nested types"
)
def test_null_literal(con, field):
    query = _NULL_STRUCT_LITERAL[field]
    result = pd.Series([con.execute(query)])
    result = result.replace({np.nan: None})
    expected = pd.Series([None], dtype="object")
    tm.assert_series_equal(result, expected)


@pytest.mark.notimpl(["dask", "pandas", "postgres"])
def test_struct_column(alltypes, df):
    t = alltypes
    expr = ibis.struct(dict(a=t.string_col, b=1, c=t.bigint_col)).name("s")
    assert expr.type() == dt.Struct(dict(a=dt.string, b=dt.int8, c=dt.int64))
    result = expr.execute()
    expected = pd.Series(
        (dict(a=a, b=1, c=c) for a, c in zip(df.string_col, df.bigint_col)),
        name="s",
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.notimpl(["dask", "pandas", "postgres", "polars"])
def test_collect_into_struct(alltypes):
    from ibis import _

    t = alltypes
    expr = (
        t[_.string_col.isin(("0", "1"))]
        .group_by(group="string_col")
        .agg(
            val=lambda t: ibis.struct(
                dict(key=t.bigint_col.collect().cast("!array<int64>"))
            )
        )
    )
    result = expr.execute()
    assert result.shape == (2, 2)
    assert set(result.group) == {"0", "1"}
    val = result.val
    assert len(val.loc[result.group == "0"].iat[0]["key"]) == 730
    assert len(val.loc[result.group == "1"].iat[0]["key"]) == 730
