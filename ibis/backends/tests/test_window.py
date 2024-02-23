from __future__ import annotations

from functools import partial
from operator import methodcaller

import numpy as np
import pandas as pd
import pytest
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis.backends.tests.errors import (
    ClickHouseDatabaseError,
    ExaQueryError,
    GoogleBadRequest,
    ImpalaHiveServer2Error,
    MySQLOperationalError,
    OracleDatabaseError,
    PsycoPg2InternalError,
    Py4JJavaError,
    PyDruidProgrammingError,
    PyODBCProgrammingError,
    SnowflakeProgrammingError,
)
from ibis.legacy.udf.vectorized import analytic, reduction

pytestmark = [
    pytest.mark.notimpl(
        ["druid"], raises=(com.OperationNotDefinedError, PyDruidProgrammingError)
    )
]


# adapted from https://gist.github.com/xmnlab/2c1f93df1a6c6bde4e32c8579117e9cc
def pandas_ntile(x, bucket: int):
    """Divide values into a number of buckets.

    It divides an ordered and grouped data set into a number of buckets
    and assigns the appropriate bucket number to each row.

    Return an integer ranging from 0 to `bucket - 1`, dividing the
    partition as equally as possible.
    """

    # internal ntile function
    def _ntile(x: pd.Series, bucket: int) -> pd.Series:
        n = x.shape[0]
        sub_n = n // bucket
        diff = n - (sub_n * bucket)

        result = []
        for i in range(bucket):
            sub_result = [i] * (sub_n + (1 if diff else 0))
            result.extend(sub_result)
            if diff > 0:
                diff -= 1
        return pd.Series(result, index=x.index)

    if isinstance(x, pd.core.groupby.generic.SeriesGroupBy):
        return x.apply(partial(_ntile, bucket=bucket))
    elif isinstance(x, pd.Series):
        return _ntile(x, bucket)

    raise TypeError(
        "`x` should be `pandas.Series` or `pandas.DataFrame` or "
        "`pd.core.groupby.generic.SeriesGroupBy` or "
        "`pd.core.groupby.generic.DataFrameGroupBy`, "
        f"not {type(x)}."
    )


@reduction(input_type=[dt.double], output_type=dt.double)
def mean_udf(s):
    return s.mean()


@analytic(input_type=[dt.double], output_type=dt.double)
def calc_zscore(s):
    return (s - s.mean()) / s.std()


@pytest.mark.parametrize(
    ("result_fn", "expected_fn"),
    [
        param(
            lambda t, win: t.float_col.lag().over(win),
            lambda t: t.float_col.shift(1),
            id="lag",
        ),
        param(
            lambda t, win: t.float_col.lead().over(win),
            lambda t: t.float_col.shift(-1),
            id="lead",
            marks=[
                pytest.mark.broken(
                    ["clickhouse"],
                    reason="upstream is broken; returns all nulls",
                    raises=AssertionError,
                ),
            ],
        ),
        param(
            lambda t, win: t.id.rank().over(win),
            lambda t: t.id.rank(method="min").astype("int64") - 1,
            id="rank",
        ),
        param(
            lambda t, win: t.id.dense_rank().over(win),
            lambda t: t.id.rank(method="dense").astype("int64") - 1,
            id="dense_rank",
        ),
        param(
            lambda t, win: t.id.percent_rank().over(win),
            lambda t: t.apply(
                lambda df: (
                    df.sort_values("id").id.rank(method="min").sub(1).div(len(df) - 1)
                )
            ).reset_index(drop=True, level=[0]),
            id="percent_rank",
            marks=[
                pytest.mark.notyet(
                    ["clickhouse"],
                    reason="clickhouse doesn't implement percent_rank",
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="Feature is not yet implemented: Unrecognized window function: percent_rank",
                ),
            ],
        ),
        param(
            lambda t, win: t.id.cume_dist().over(win),
            lambda t: t.id.rank(method="min") / t.id.transform(len),
            id="cume_dist",
            marks=[
                pytest.mark.notyet(
                    ["clickhouse", "exasol"], raises=com.OperationNotDefinedError
                ),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="Feature is not yet implemented: Unrecognized window function: cume_dist",
                ),
            ],
        ),
        param(
            lambda t, win: t.float_col.ntile(buckets=7).over(win),
            lambda t: pandas_ntile(t.float_col, 7),
            id="ntile",
            marks=[
                pytest.mark.notimpl(
                    ["dask", "pandas", "polars"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.broken(
                    ["impala"],
                    raises=AssertionError,
                    reason="Results don't match; possibly due to ordering",
                ),
                pytest.mark.broken(
                    ["datafusion"],
                    raises=AssertionError,
                    reason="Results are shifted + 1",
                ),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="Feature is not yet implemented: Unrecognized window function: ntile",
                ),
            ],
        ),
        param(
            lambda t, win: t.float_col.first().over(win),
            lambda t: t.float_col.transform("first"),
            id="first",
            marks=[
                pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError),
            ],
        ),
        param(
            lambda t, win: t.float_col.last().over(win),
            lambda t: t.float_col.transform("last"),
            id="last",
            marks=[
                pytest.mark.notimpl(["exasol"], raises=com.OperationNotDefinedError),
            ],
        ),
        param(
            lambda t, win: t.double_col.nth(3).over(win),
            lambda t: t.double_col.apply(
                lambda s: pd.concat(
                    [
                        pd.Series(np.nan, index=s.index[:3], dtype="float64"),
                        pd.Series(s.iloc[3], index=s.index[3:], dtype="float64"),
                    ]
                )
            ),
            id="nth",
            marks=[
                pytest.mark.notimpl(["pandas"], raises=com.OperationNotDefinedError),
                pytest.mark.notyet(
                    ["impala", "mssql"], raises=com.OperationNotDefinedError
                ),
                pytest.mark.notimpl(["dask"], raises=com.OperationNotDefinedError),
                pytest.mark.notimpl(["flink"], raises=com.OperationNotDefinedError),
                pytest.mark.notimpl(["risingwave"], raises=PsycoPg2InternalError),
                pytest.mark.broken(
                    ["snowflake"],
                    raises=AssertionError,
                    reason="behavior of windows with nth_value has changed; unclear whether that is in ibis or in snowflake itself",
                ),
            ],
        ),
        param(
            lambda _, win: ibis.row_number().over(win),
            lambda t: t.cumcount(),
            id="row_number",
        ),
        param(
            lambda t, win: t.double_col.cumsum().over(win),
            lambda t: t.double_col.cumsum(),
            id="cumsum",
        ),
        param(
            lambda t, win: t.double_col.cummean().over(win),
            lambda t: (t.double_col.expanding().mean().reset_index(drop=True, level=0)),
            id="cummean",
        ),
        param(
            lambda t, win: t.float_col.cummin().over(win),
            lambda t: t.float_col.cummin(),
            id="cummin",
        ),
        param(
            lambda t, win: t.float_col.cummax().over(win),
            lambda t: t.float_col.cummax(),
            id="cummax",
        ),
        param(
            lambda t, win: (t.double_col == 0).any().over(win),
            lambda t: (
                t.double_col.expanding()
                .agg(lambda s: s.eq(0).any())
                .reset_index(drop=True, level=0)
                .astype(bool)
            ),
            id="cumany",
        ),
        param(
            lambda t, win: (t.double_col == 0).notany().over(win),
            lambda t: (
                t.double_col.expanding()
                .agg(lambda s: ~s.eq(0).any())
                .reset_index(drop=True, level=0)
                .astype(bool)
            ),
            id="cumnotany",
            marks=[
                pytest.mark.broken(["oracle"], raises=OracleDatabaseError),
            ],
        ),
        param(
            lambda t, win: (t.double_col == 0).all().over(win),
            lambda t: (
                t.double_col.expanding()
                .agg(lambda s: s.eq(0).all())
                .reset_index(drop=True, level=0)
                .astype(bool)
            ),
            id="cumall",
        ),
        param(
            lambda t, win: (t.double_col == 0).notall().over(win),
            lambda t: (
                t.double_col.expanding()
                .agg(lambda s: ~s.eq(0).all())
                .reset_index(drop=True, level=0)
                .astype(bool)
            ),
            id="cumnotall",
            marks=[
                pytest.mark.broken(["oracle"], raises=OracleDatabaseError),
            ],
        ),
        param(
            lambda t, win: t.double_col.sum().over(win),
            lambda gb: gb.double_col.cumsum(),
            id="sum",
        ),
        param(
            lambda t, win: t.double_col.mean().over(win),
            lambda gb: (
                gb.double_col.expanding().mean().reset_index(drop=True, level=0)
            ),
            id="mean",
        ),
        param(
            lambda t, win: t.float_col.min().over(win),
            lambda gb: gb.float_col.cummin(),
            id="min",
        ),
        param(
            lambda t, win: t.float_col.max().over(win),
            lambda gb: gb.float_col.cummax(),
            id="max",
        ),
        param(
            lambda t, win: t.double_col.count().over(win),
            # pandas doesn't including the current row, but following=0 implies
            # that we must, so we add one to the pandas result
            lambda gb: gb.double_col.cumcount() + 1,
            id="count",
        ),
    ],
)
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
def test_grouped_bounded_expanding_window(
    backend, alltypes, df, result_fn, expected_fn
):
    expr = alltypes.mutate(
        val=result_fn(
            alltypes,
            win=ibis.window(
                following=0,
                group_by=[alltypes.string_col],
                order_by=[alltypes.id],
            ),
        )
    )

    result = expr.execute().set_index("id").sort_index()
    column = expected_fn(df.sort_values("id").groupby("string_col", group_keys=True))
    if column.index.nlevels > 1:
        column = column.droplevel(0)
    expected = df.assign(val=column).set_index("id").sort_index()

    left, right = result.val, expected.val

    backend.assert_series_equal(left, right)


@pytest.mark.parametrize(
    ("result_fn", "expected_fn"),
    [
        param(
            lambda t, win: t.double_col.mean().over(win),
            lambda df: (df.double_col.expanding().mean()),
            id="mean",
            marks=[
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="Feature is not yet implemented: Window function with empty PARTITION BY is not supported yet",
                ),
            ],
        ),
        param(
            # Disabled on PySpark and Spark backends because in pyspark<3.0.0,
            # Pandas UDFs are only supported on unbounded windows
            lambda t, win: mean_udf(t.double_col).over(win),
            lambda df: (df.double_col.expanding().mean()),
            id="mean_udf",
            marks=[
                pytest.mark.notimpl(
                    [
                        "bigquery",
                        "clickhouse",
                        "duckdb",
                        "flink",
                        "impala",
                        "mssql",
                        "mysql",
                        "oracle",
                        "postgres",
                        "risingwave",
                        "sqlite",
                        "snowflake",
                        "datafusion",
                        "trino",
                        "exasol",
                    ],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
    ],
)
# Some backends do not support non-grouped window specs
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
def test_ungrouped_bounded_expanding_window(
    backend, alltypes, df, result_fn, expected_fn
):
    expr = alltypes.mutate(
        val=result_fn(
            alltypes,
            win=ibis.window(following=0, order_by=[alltypes.id]),
        )
    )
    result = expr.execute().set_index("id").sort_index()

    column = expected_fn(df.sort_values("id"))
    expected = df.assign(val=column).set_index("id").sort_index()

    left, right = result.val, expected.val

    backend.assert_series_equal(left, right)


@pytest.mark.parametrize(
    "preceding, following",
    [
        (0, 2),
        (None, (0, 2)),
    ],
)
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
def test_grouped_bounded_following_window(backend, alltypes, df, preceding, following):
    window = ibis.window(
        preceding=preceding,
        following=following,
        group_by=[alltypes.string_col],
        order_by=[alltypes.id],
    )

    expr = alltypes.mutate(val=alltypes.id.mean().over(window))

    result = expr.execute().set_index("id").sort_index()

    # shift id column before applying Pandas rolling window summarizer to
    # simulate forward looking window aggregation
    gdf = df.sort_values("id").groupby("string_col")
    gdf.id = gdf.apply(lambda t: t.id.shift(-2))
    expected = (
        df.assign(
            val=gdf.id.rolling(3, min_periods=1)
            .mean()
            .sort_index(level=1)
            .reset_index(drop=True)
        )
        .set_index("id")
        .sort_index()
    )

    # discard first 2 rows of each group to account for the shift
    n = len(gdf) * 2
    left, right = result.val.shift(-n), expected.val.shift(-n)

    backend.assert_series_equal(left, right)


@pytest.mark.parametrize(
    "window_fn, window_size",
    [
        param(
            lambda t: ibis.window(
                preceding=2,
                following=0,
                group_by=[t.string_col],
                order_by=[t.id],
            ),
            3,
            id="preceding-2-following-0",
        ),
        param(
            lambda t: ibis.window(
                preceding=(2, 0),
                following=None,
                group_by=[t.string_col],
                order_by=[t.id],
            ),
            3,
            id="preceding-2-following-0-tuple",
        ),
        param(
            lambda t: ibis.trailing_window(
                preceding=2, group_by=[t.string_col], order_by=[t.id]
            ),
            3,
            id="trailing-2",
        ),
        param(
            lambda t: ibis.window(
                # snowflake doesn't allow windows larger than 1000
                preceding=999,
                following=0,
                group_by=[t.string_col],
                order_by=[t.id],
            ),
            1000,
            id="large-preceding-999-following-0",
        ),
        param(
            lambda t: ibis.window(
                preceding=1000, following=0, group_by=[t.string_col], order_by=[t.id]
            ),
            1001,
            marks=[
                pytest.mark.notyet(
                    ["snowflake"],
                    raises=SnowflakeProgrammingError,
                    reason="Windows larger than 1000 are not allowed",
                )
            ],
            id="large-preceding-1000-following-0",
        ),
    ],
)
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
def test_grouped_bounded_preceding_window(
    backend, alltypes, df, window_fn, window_size
):
    window = window_fn(alltypes)
    expr = alltypes.mutate(val=alltypes.double_col.sum().over(window))

    result = expr.execute().set_index("id").sort_index()
    gdf = df.sort_values("id").groupby("string_col")
    expected = (
        df.assign(
            val=gdf.double_col.rolling(window_size, min_periods=1)
            .sum()
            .sort_index(level=1)
            .reset_index(drop=True)
        )
        .set_index("id")
        .sort_index()
    )

    left, right = result.val, expected.val

    backend.assert_series_equal(left, right)


@pytest.mark.parametrize(
    ("result_fn", "expected_fn"),
    [
        param(
            lambda t, win: t.double_col.mean().over(win),
            lambda gb: (gb.double_col.transform("mean")),
            id="mean",
        ),
        param(
            lambda t, win: mean_udf(t.double_col).over(win),
            lambda gb: (gb.double_col.transform("mean")),
            id="mean_udf",
            marks=[
                pytest.mark.notimpl(
                    [
                        "bigquery",
                        "clickhouse",
                        "duckdb",
                        "flink",
                        "impala",
                        "mssql",
                        "mysql",
                        "oracle",
                        "postgres",
                        "risingwave",
                        "sqlite",
                        "snowflake",
                        "trino",
                        "datafusion",
                        "exasol",
                    ],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    ("ordered"),
    [
        param(False, id="unordered"),
    ],
)
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
def test_grouped_unbounded_window(
    backend, alltypes, df, result_fn, expected_fn, ordered
):
    # Define a window that is
    # 1) Grouped
    # 2) Ordered if `ordered` is True
    # 3) Unbounded
    order_by = [alltypes.id] if ordered else None
    window = ibis.window(group_by=[alltypes.string_col], order_by=order_by)
    expr = alltypes.mutate(
        val=result_fn(
            alltypes,
            win=window,
        )
    )
    result = expr.execute()
    result = result.set_index("id").sort_index()

    # Apply `expected_fn` onto a DataFrame that is
    # 1) Grouped
    # 2) Ordered if `ordered` is True
    df = df.sort_values("id") if ordered else df
    expected = df.assign(val=expected_fn(df.groupby("string_col")))
    expected = expected.set_index("id").sort_index()

    left, right = result.val, expected.val
    backend.assert_series_equal(left, right)


@pytest.mark.parametrize(
    ("ibis_method", "pandas_fn"),
    [
        param(methodcaller("sum"), lambda s: s.cumsum(), id="sum"),
        param(methodcaller("min"), lambda s: s.cummin(), id="min"),
        param(methodcaller("mean"), lambda s: s.expanding().mean(), id="mean"),
    ],
)
@pytest.mark.broken(["snowflake"], raises=AssertionError)
@pytest.mark.broken(["dask"], raises=AssertionError)
@pytest.mark.notyet(["mssql"], raises=PyODBCProgrammingError)
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="Feature is not yet implemented: Window function with empty PARTITION BY is not supported yet",
)
def test_simple_ungrouped_unbound_following_window(
    backend, alltypes, ibis_method, pandas_fn
):
    t = alltypes[alltypes.double_col < 50].order_by("id")
    df = t.execute()

    w = ibis.window(rows=(0, None), order_by=t.id)
    expr = ibis_method(t.double_col).over(w).name("double_col")
    result = expr.execute()
    expected = pandas_fn(df.double_col[::-1])[::-1]
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(
    ["flink"],
    raises=Py4JJavaError,
    reason="flink doesn't allow order by NULL without casting the null to a specific type",
)
@pytest.mark.never(
    ["mssql"], raises=Exception, reason="order by constant is not supported"
)
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="Feature is not yet implemented: Window function with empty PARTITION BY is not supported yet",
)
@pytest.mark.xfail_version(datafusion=["datafusion==35"])
def test_simple_ungrouped_window_with_scalar_order_by(alltypes):
    t = alltypes[alltypes.double_col < 50].order_by("id")
    w = ibis.window(rows=(0, None), order_by=ibis.NA)
    expr = t.double_col.sum().over(w).name("double_col")
    # hard to reproduce this in pandas, so just test that it actually executes
    expr.execute()


@pytest.mark.parametrize(
    ("result_fn", "expected_fn", "ordered"),
    [
        # Reduction ops
        param(
            lambda t, win: t.double_col.mean().over(win),
            lambda df: pd.Series([df.double_col.mean()] * len(df.double_col)),
            True,
            id="ordered-mean",
            marks=[
                pytest.mark.broken(
                    ["impala"],
                    reason="default window semantics are different",
                    raises=AssertionError,
                ),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="Feature is not yet implemented: Window function with empty PARTITION BY is not supported yet",
                ),
            ],
        ),
        param(
            lambda t, win: t.double_col.mean().over(win),
            lambda df: pd.Series([df.double_col.mean()] * len(df.double_col)),
            False,
            id="unordered-mean",
        ),
        param(
            lambda _, win: ibis.ntile(7).over(win),
            lambda df: pandas_ntile(df.id, 7),
            True,
            id="unordered-ntile",
            marks=[
                pytest.mark.notimpl(
                    ["pandas", "dask"], raises=com.OperationNotDefinedError
                ),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="Feature is not yet implemented: Unrecognized window function: ntile",
                ),
            ],
        ),
        param(
            lambda t, win: mean_udf(t.double_col).over(win),
            lambda df: pd.Series([df.double_col.mean()] * len(df.double_col)),
            True,
            id="ordered-mean_udf",
            marks=[
                pytest.mark.notimpl(
                    [
                        "bigquery",
                        "clickhouse",
                        "duckdb",
                        "flink",
                        "impala",
                        "mssql",
                        "mysql",
                        "oracle",
                        "postgres",
                        "risingwave",
                        "sqlite",
                        "snowflake",
                        "trino",
                        "datafusion",
                        "exasol",
                    ],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        param(
            lambda t, win: mean_udf(t.double_col).over(win),
            lambda df: pd.Series([df.double_col.mean()] * len(df.double_col)),
            False,
            id="unordered-mean_udf",
            marks=[
                pytest.mark.notimpl(
                    [
                        "bigquery",
                        "clickhouse",
                        "duckdb",
                        "impala",
                        "mssql",
                        "mysql",
                        "oracle",
                        "postgres",
                        "risingwave",
                        "sqlite",
                        "snowflake",
                        "trino",
                        "datafusion",
                        "exasol",
                        "flink",
                    ],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        # Analytic ops
        param(
            lambda t, win: t.float_col.lag().over(win),
            lambda df: df.float_col.shift(1),
            True,
            id="ordered-lag",
            marks=[
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="Feature is not yet implemented: Window function with empty PARTITION BY is not supported yet",
                ),
            ],
        ),
        param(
            lambda t, win: t.float_col.lag().over(win),
            lambda df: df.float_col.shift(1),
            False,
            id="unordered-lag",
            marks=[
                pytest.mark.broken(
                    ["trino", "exasol"],
                    reason="this isn't actually broken: the backend result is equal up to ordering",
                    raises=AssertionError,
                    strict=False,  # sometimes it passes
                ),
                pytest.mark.notyet(["flink"], raises=Py4JJavaError),
                pytest.mark.broken(["mssql"], raises=PyODBCProgrammingError),
                pytest.mark.notyet(
                    ["snowflake"],
                    reason="backend supports this functionality but requires meaningful ordering",
                    raises=AssertionError,
                ),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="Feature is not yet implemented: Window function with empty PARTITION BY is not supported yet",
                ),
            ],
        ),
        param(
            lambda t, win: t.float_col.lead().over(win),
            lambda df: df.float_col.shift(-1),
            True,
            id="ordered-lead",
            marks=[
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="Feature is not yet implemented: Window function with empty PARTITION BY is not supported yet",
                ),
            ],
        ),
        param(
            lambda t, win: t.float_col.lead().over(win),
            lambda df: df.float_col.shift(-1),
            False,
            id="unordered-lead",
            marks=[
                pytest.mark.broken(
                    ["trino"],
                    reason=(
                        "this isn't actually broken: the trino backend "
                        "result is equal up to ordering"
                    ),
                    raises=AssertionError,
                    strict=False,  # sometimes it passes
                ),
                pytest.mark.notyet(["flink"], raises=Py4JJavaError),
                pytest.mark.broken(["mssql"], raises=PyODBCProgrammingError),
                pytest.mark.notyet(
                    ["snowflake"],
                    reason="backend supports this functionality but requires meaningful ordering",
                    raises=AssertionError,
                ),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="Feature is not yet implemented: Window function with empty PARTITION BY is not supported yet",
                ),
            ],
        ),
        param(
            lambda t, win: calc_zscore(t.double_col).over(win),
            lambda df: df.double_col.transform(calc_zscore.func),
            True,
            id="ordered-zscore_udf",
            marks=[
                pytest.mark.notimpl(
                    [
                        "bigquery",
                        "clickhouse",
                        "duckdb",
                        "flink",
                        "impala",
                        "mssql",
                        "mysql",
                        "oracle",
                        "postgres",
                        "risingwave",
                        "pyspark",
                        "sqlite",
                        "snowflake",
                        "trino",
                        "datafusion",
                        "exasol",
                    ],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        param(
            lambda t, win: calc_zscore(t.double_col).over(win),
            lambda df: df.double_col.transform(calc_zscore.func),
            False,
            id="unordered-zscore_udf",
            marks=[
                pytest.mark.notimpl(
                    [
                        "bigquery",
                        "clickhouse",
                        "duckdb",
                        "impala",
                        "mssql",
                        "mysql",
                        "oracle",
                        "postgres",
                        "risingwave",
                        "pyspark",
                        "sqlite",
                        "snowflake",
                        "trino",
                        "datafusion",
                        "exasol",
                        "flink",
                    ],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
    ],
)
# Some backends do not support non-grouped window specs
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
def test_ungrouped_unbounded_window(
    backend, alltypes, df, result_fn, expected_fn, ordered
):
    # Define a window that is
    # 1) Ungrouped
    # 2) Ordered if `ordered` is True
    # 3) Unbounded
    order_by = [alltypes.id] if ordered else None
    window = ibis.window(order_by=order_by)
    expr = alltypes.mutate(val=result_fn(alltypes, win=window))
    result = expr.execute()
    result = result.set_index("id").sort_index()

    # Apply `expected_fn` onto a DataFrame that is
    # 1) Ungrouped
    # 2) Ordered if `ordered` is True
    df = df.sort_values("id") if ordered else df
    expected = df.assign(val=expected_fn(df))
    expected = expected.set_index("id").sort_index()

    left, right = result.val, expected.val

    backend.assert_series_equal(left, right)


@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(["snowflake"], raises=SnowflakeProgrammingError)
@pytest.mark.notimpl(
    ["impala"], raises=ImpalaHiveServer2Error, reason="limited RANGE support"
)
@pytest.mark.notyet(
    ["clickhouse"],
    reason="RANGE OFFSET frame for 'DB::ColumnNullable' ORDER BY column is not implemented",
    raises=ClickHouseDatabaseError,
)
@pytest.mark.notyet(["mssql"], raises=PyODBCProgrammingError)
@pytest.mark.broken(
    ["mysql"],
    raises=MySQLOperationalError,
    reason="https://github.com/tobymao/sqlglot/issues/2779",
)
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="Feature is not yet implemented: window frame in `RANGE` mode is not supported yet",
)
def test_grouped_bounded_range_window(backend, alltypes, df):
    # Explanation of the range window spec below:
    #
    # `preceding=10, following=0, order_by='id'``:
    #     The window at a particular row (call its `id` value x) will contain
    #     some other row (call its `id` value y) if x-10 <= y <= x.
    # `group_by='string_col'`:
    #     The window at a particular row will only contain other rows that
    #     have the same 'string_col' value.
    preceding = 10
    window = ibis.range_window(
        preceding=preceding,
        following=0,
        order_by="id",
        group_by="string_col",
    )
    expr = alltypes.mutate(val=alltypes.double_col.sum().over(window))
    result = expr.execute().set_index("id").sort_index()

    def gb_fn(df):
        indices = np.searchsorted(df.id, [df["prec"], df["foll"]], side="left")
        double_col = df.double_col.values
        return pd.Series(
            [double_col[start:stop].sum() for start, stop in indices.T],
            index=df.index,
        )

    res = (
        # add 1 to get the upper bound without having to make two
        # searchsorted calls
        df.assign(prec=lambda t: t.id - preceding, foll=lambda t: t.id + 1)
        .sort_values("id")
        .groupby("string_col")
        .apply(gb_fn)
        .droplevel(0)
    )
    expected = (
        df.assign(
            # Mimic our range window spec using .apply()
            val=res
        )
        .set_index("id")
        .sort_index()
    )

    backend.assert_series_equal(result.val, expected.val)


@pytest.mark.notimpl(["clickhouse", "polars"], raises=com.OperationNotDefinedError)
@pytest.mark.notyet(
    ["clickhouse"],
    reason="clickhouse doesn't implement percent_rank",
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="Feature is not yet implemented: Unrecognized window function: percent_rank",
)
def test_percent_rank_whole_table_no_order_by(backend, alltypes, df):
    expr = alltypes.mutate(val=lambda t: t.id.percent_rank())

    result = expr.execute().set_index("id").sort_index()
    column = df.id.rank(method="min").sub(1).div(len(df) - 1)
    expected = df.assign(val=column).set_index("id").sort_index()

    backend.assert_series_equal(result.val, expected.val)


@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
def test_grouped_ordered_window_coalesce(backend, alltypes, df):
    t = alltypes
    expr = (
        t.group_by("month")
        .order_by("id")
        .mutate(lagged_value=ibis.coalesce(t.bigint_col.lag(), 0))[
            ["id", "lagged_value"]
        ]
    )
    result = (
        expr.execute()
        .sort_values(["id"])
        .lagged_value.reset_index(drop=True)
        .astype("int64")
    )

    def agg(df):
        df = df.sort_values(["id"])
        df = df.assign(bigint_col=lambda df: df.bigint_col.shift())
        return df

    expected = (
        df.groupby("month", group_keys=False)
        .apply(agg)
        .sort_values(["id"])
        .reset_index(drop=True)
        .bigint_col.fillna(0.0)
        .astype("int64")
        .rename("lagged_value")
    )
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="Feature is not yet implemented: Window function with empty PARTITION BY is not supported yet",
)
def test_mutate_window_filter(backend, alltypes):
    t = alltypes
    win = ibis.window(order_by=[t.id])
    expr = (
        t.mutate(next_int=t.int_col.lead().over(win))
        .filter(lambda t: t.int_col == 1)
        .select("int_col", "next_int")
        .limit(3)
    )
    res = expr.execute()
    sol = pd.DataFrame({"int_col": [1, 1, 1], "next_int": [2, 2, 2]})
    backend.assert_frame_equal(res, sol, check_dtype=False)


@pytest.mark.notimpl(["polars", "exasol"], raises=com.OperationNotDefinedError)
def test_first_last(backend):
    t = backend.win
    w = ibis.window(group_by=t.g, order_by=[t.x, t.y], preceding=1, following=0)
    expr = t.mutate(
        x_first=t.x.first().over(w),
        x_last=t.x.last().over(w),
        y_first=t.y.first().over(w),
        y_last=t.y.last().over(w),
    )
    result = expr.execute()
    expected = pd.DataFrame(
        {
            "g": ["a"] * 5,
            "x": range(5),
            "y": [3, 2, 0, 1, 1],
            "x_first": [0, 0, 1, 2, 3],
            "x_last": range(5),
            "y_first": [3, 3, 2, 0, 1],
            "y_last": [3, 2, 0, 1, 1],
        }
    )
    backend.assert_frame_equal(result, expected)


@pytest.mark.notyet(
    ["bigquery"], raises=GoogleBadRequest, reason="not supported by BigQuery"
)
@pytest.mark.notyet(
    ["clickhouse"], raises=ClickHouseDatabaseError, reason="not supported by ClickHouse"
)
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="not supported by DataFusion"
)
@pytest.mark.notyet(
    ["impala"], raises=ImpalaHiveServer2Error, reason="not supported by Impala"
)
@pytest.mark.notyet(
    ["mysql"], raises=MySQLOperationalError, reason="not supported by MySQL"
)
@pytest.mark.notyet(
    ["polars", "snowflake", "sqlite"],
    raises=com.OperationNotDefinedError,
    reason="not support by the backend",
)
@pytest.mark.notyet(
    ["mssql"], raises=PyODBCProgrammingError, reason="not support by the backend"
)
@pytest.mark.broken(["flink"], raises=Py4JJavaError, reason="bug in Flink")
@pytest.mark.broken(
    ["exasol"],
    raises=ExaQueryError,
    reason="database can't handle UTC timestamps in DataFrames",
)
@pytest.mark.broken(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="sql parser error: Expected literal int, found: INTERVAL at line:1, column:99",
)
def test_range_expression_bounds(backend):
    t = ibis.memtable(
        {
            "time": pd.to_datetime(
                [
                    "2016-05-25 13:30:00.023",
                    "2016-05-25 13:30:00.023",
                    "2016-05-25 13:30:00.030",
                    "2016-05-25 13:30:00.041",
                    "2016-05-25 13:30:00.048",
                    "2016-05-25 13:30:00.049",
                    "2016-05-25 13:30:00.072",
                    "2016-05-25 13:30:00.075",
                ],
                utc=True,
            ),
            "ticker": ["GOOG", "MSFT", "MSFT", "MSFT", "GOOG", "AAPL", "GOOG", "MSFT"],
            "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
            "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03],
        }
    )
    expr = t.select(
        "ticker",
        total_bid_amount=lambda t: t.bid.sum().over(
            range=(-ibis.interval(seconds=10), 0), order_by=t.time
        ),
    )

    con = backend.connection
    result = con.execute(expr)

    assert not result.empty
    assert len(result) == con.execute(t.count())


@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
@pytest.mark.notyet(
    ["clickhouse"],
    reason="clickhouse doesn't implement percent_rank",
    raises=com.OperationNotDefinedError,
)
@pytest.mark.broken(
    ["mssql"], reason="lack of support for booleans", raises=PyODBCProgrammingError
)
@pytest.mark.broken(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="Feature is not yet implemented: Unrecognized window function: percent_rank",
)
@pytest.mark.broken(["dask"], reason="different result ordering", raises=AssertionError)
def test_rank_followed_by_over_call_merge_frames(backend, alltypes, df):
    # GH #7631
    t = alltypes
    expr = t.int_col.percent_rank().over(ibis.window(group_by=t.int_col.notnull()))
    result = expr.execute()

    expected = (
        df.sort_values("int_col")
        .groupby(df["int_col"].notnull())
        .apply(lambda df: (df.int_col.rank(method="min").sub(1).div(len(df) - 1)))
        .T.reset_index(drop=True)
        .iloc[:, 0]
        .rename(expr.get_name())
    )

    backend.assert_series_equal(result, expected)


@pytest.mark.notyet(
    ["mssql"],
    reason="IS NULL not valid syntax for mssql",
    raises=PyODBCProgrammingError,
)
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="Feature is not yet implemented: Window function with empty PARTITION BY is not supported yet",
)
def test_windowed_order_by_sequence_is_preserved(con):
    table = ibis.memtable({"bool_col": [True, False, False, None, True]})
    window = ibis.window(
        order_by=[
            ibis.asc(table["bool_col"].isnull()),
            ibis.asc(table["bool_col"]),
        ],
    )
    expr = table.select(
        rank=table["bool_col"].rank().over(window),
        bool_col=table["bool_col"],
    )

    result = con.execute(expr)
    value = result.bool_col.loc[result["rank"] == 4].item()
    assert pd.isna(value)
