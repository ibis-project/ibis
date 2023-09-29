from __future__ import annotations

import pandas as pd
import pandas.testing as tm
import pytest
from packaging.version import parse as vparse
from pytest import param

import ibis
from ibis.backends.tests.test_vectorized_udf import calc_mean, create_demean_struct_udf

pytestmark = pytest.mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "datafusion",
        "impala",
        "mysql",
        "postgres",
        "sqlite",
        "snowflake",
        "polars",
        "mssql",
        "trino",
        "druid",
        "oracle",
    ]
)

GROUP_BY_COL = "month"
ORDER_BY_COL = "timestamp_col"
TARGET_COL = "float_col"


@pytest.fixture
def context():
    # These need to be tz-naive because the timestamp_col in
    # the test data is tz-naive
    return pd.Timestamp("20090105"), pd.Timestamp("20090111")


def filter_by_time_context(df, context):
    begin, end = context
    return df[(df.timestamp_col >= begin) & (df.timestamp_col < end)]


broken_pandas_grouped_rolling = pytest.mark.xfail(
    condition=vparse("1.4") <= vparse(pd.__version__) < vparse("1.4.2"),
    raises=ValueError,
    reason="https://github.com/pandas-dev/pandas/pull/44068",
)


@pytest.mark.notimpl(["dask", "duckdb"])
@pytest.mark.xfail_version(
    pyspark=["pyspark<3.1"], pandas=["pyarrow>=13", "pandas>=2.1"]
)
@pytest.mark.parametrize(
    "window",
    [
        param(
            ibis.trailing_window(ibis.interval(days=3), order_by=ORDER_BY_COL),
            id="order_by",
        ),
        param(
            ibis.trailing_window(
                ibis.interval(days=3),
                order_by=ORDER_BY_COL,
                group_by=GROUP_BY_COL,
            ),
            id="order_by_group_by",
            marks=[broken_pandas_grouped_rolling],
        ),
    ],
)
def test_context_adjustment_window_udf(alltypes, context, window, monkeypatch):
    """Test context adjustment of udfs in window methods."""
    monkeypatch.setattr(ibis.options.context_adjustment, "time_col", "timestamp_col")

    expr = alltypes.mutate(v1=calc_mean(alltypes[TARGET_COL]).over(window))
    result = expr.execute(timecontext=context)

    expected = expr.execute()
    expected = filter_by_time_context(expected, context).reset_index(drop=True)

    tm.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["dask", "duckdb"])
@pytest.mark.xfail_version(pandas=["pyarrow>=13", "pandas>=2.1"])
def test_context_adjustment_filter_before_window(alltypes, context, monkeypatch):
    monkeypatch.setattr(ibis.options.context_adjustment, "time_col", "timestamp_col")

    window = ibis.trailing_window(ibis.interval(days=3), order_by=ORDER_BY_COL)

    expr = alltypes[alltypes["bool_col"]]
    expr = expr.mutate(v1=expr[TARGET_COL].count().over(window))

    result = expr.execute(timecontext=context)

    expected = expr.execute()
    expected = filter_by_time_context(expected, context)
    expected = expected.reset_index(drop=True)

    tm.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["duckdb", "pyspark"])
def test_context_adjustment_multi_col_udf_non_grouped(alltypes, context, monkeypatch):
    monkeypatch.setattr(ibis.options.context_adjustment, "time_col", "timestamp_col")

    w = ibis.window(preceding=None, following=None)

    demean_struct_udf = create_demean_struct_udf(
        result_formatter=lambda v1, v2: (v1, v2)
    )

    result = alltypes.mutate(
        demean_struct_udf(alltypes["double_col"], alltypes["int_col"])
        .over(w)
        .destructure()
    ).execute(timecontext=context)

    expected = alltypes.mutate(
        demean=lambda t: t.double_col - t.double_col.mean().over(w),
        demean_weight=lambda t: t.int_col - t.int_col.mean().over(w),
    ).execute(timecontext=context)
    tm.assert_frame_equal(result, expected)
