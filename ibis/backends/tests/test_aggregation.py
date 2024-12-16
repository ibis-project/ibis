from __future__ import annotations

import itertools
from datetime import date
from operator import methodcaller

import pytest
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis import _
from ibis import literal as L
from ibis.backends.tests.errors import (
    ClickHouseDatabaseError,
    DatabricksServerOperationError,
    ExaQueryError,
    GoogleBadRequest,
    ImpalaHiveServer2Error,
    MySQLNotSupportedError,
    OracleDatabaseError,
    PolarsInvalidOperationError,
    PsycoPg2InternalError,
    Py4JError,
    Py4JJavaError,
    PyDruidProgrammingError,
    PyODBCProgrammingError,
    PySparkAnalysisException,
    PySparkPythonException,
    SnowflakeProgrammingError,
    TrinoUserError,
)
from ibis.conftest import IS_SPARK_REMOTE
from ibis.legacy.udf.vectorized import reduction

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

with pytest.warns(FutureWarning, match="v9.0"):

    @reduction(input_type=[dt.double], output_type=dt.double)
    def mean_udf(s):
        return s.mean()


aggregate_test_params = [
    param(lambda t: t.double_col.mean(), lambda t: t.double_col.mean(), id="mean"),
    param(
        lambda t: mean_udf(t.double_col),
        lambda t: t.double_col.mean(),
        id="mean_udf",
        marks=[
            pytest.mark.notimpl(
                [
                    "bigquery",
                    "datafusion",
                    "postgres",
                    "risingwave",
                    "clickhouse",
                    "impala",
                    "duckdb",
                    "polars",
                    "snowflake",
                    "mssql",
                    "trino",
                    "druid",
                    "oracle",
                    "flink",
                    "exasol",
                    "databricks",
                ],
                raises=com.OperationNotDefinedError,
            ),
            pytest.mark.never(
                ["sqlite", "mysql"],
                reason="no udf support",
                raises=com.OperationNotDefinedError,
            ),
            pytest.mark.notyet(
                ["pyspark"],
                condition=IS_SPARK_REMOTE,
                raises=PySparkPythonException,
                reason="remote udfs not yet tested due to environment complexities",
            ),
        ],
    ),
    param(lambda t: t.double_col.min(), lambda t: t.double_col.min(), id="min"),
    param(lambda t: t.double_col.max(), lambda t: t.double_col.max(), id="max"),
    param(
        # int_col % 3 so there are no ties for most common value
        lambda t: (t.int_col % 3).mode(),
        lambda t: (t.int_col % 3).mode().iloc[0],
        id="mode",
        marks=[
            pytest.mark.notyet(
                [
                    "bigquery",
                    "clickhouse",
                    "datafusion",
                    "impala",
                    "mysql",
                    "mssql",
                    "pyspark",
                    "trino",
                    "druid",
                    "flink",
                    "risingwave",
                    "exasol",
                ],
                raises=com.OperationNotDefinedError,
            ),
        ],
    ),
    param(
        lambda t: (t.double_col + 5).sum(),
        lambda t: (t.double_col + 5).sum(),
        id="complex_sum",
    ),
    param(
        lambda t: t.timestamp_col.max(),
        lambda t: t.timestamp_col.max(),
        id="timestamp_max",
    ),
]

argidx_not_grouped_marks = [
    "impala",
    "mysql",
    "mssql",
    "druid",
    "oracle",
    "flink",
    "exasol",
]


def make_argidx_params(marks):
    marks = [pytest.mark.notyet(marks, raises=com.OperationNotDefinedError)]
    return [
        param(
            lambda t: t.timestamp_col.argmin(t.id),
            lambda s: s.timestamp_col.iloc[s.id.argmin()],
            id="argmin",
            marks=marks,
        ),
        param(
            lambda t: t.double_col.argmax(t.id),
            lambda s: s.double_col.iloc[s.id.argmax()],
            id="argmax",
            marks=marks,
        ),
    ]


@pytest.mark.parametrize(
    ("result_fn", "expected_fn"),
    aggregate_test_params + make_argidx_params(argidx_not_grouped_marks),
)
def test_aggregate(backend, alltypes, df, result_fn, expected_fn):
    expr = alltypes.aggregate(tmp=result_fn)
    result = expr.execute()

    # Create a single-row single-column dataframe with the Pandas `agg` result
    # (to match the output format of Ibis `aggregate`)
    expected = pd.DataFrame({"tmp": [expected_fn(df)]})

    backend.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize(
    ("result_fn", "expected_fn"),
    aggregate_test_params + make_argidx_params(argidx_not_grouped_marks),
)
def test_aggregate_grouped(backend, alltypes, df, result_fn, expected_fn):
    grouping_key_col = "bigint_col"

    # Two (equivalent) variations:
    #  1) `group_by` then `aggregate`
    #  2) `aggregate` with `by`
    expr1 = alltypes.group_by(grouping_key_col).aggregate(tmp=result_fn)
    expr2 = alltypes.aggregate(tmp=result_fn, by=grouping_key_col)
    result1 = expr1.execute()
    result2 = expr2.execute()

    # Note: Using `reset_index` to get the grouping key as a column
    expected = (
        df.groupby(grouping_key_col).apply(expected_fn).rename("tmp").reset_index()
    )

    # Row ordering may differ depending on backend, so sort on the
    # grouping key
    result1 = result1.sort_values(by=grouping_key_col).reset_index(drop=True)
    result2 = result2.sort_values(by=grouping_key_col).reset_index(drop=True)
    expected = expected.sort_values(by=grouping_key_col).reset_index(drop=True)

    backend.assert_frame_equal(result1, expected, check_dtype=False)
    backend.assert_frame_equal(result2, expected, check_dtype=False)


@pytest.mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "datafusion",
        "duckdb",
        "impala",
        "mysql",
        "postgres",
        "risingwave",
        "sqlite",
        "snowflake",
        "polars",
        "mssql",
        "trino",
        "druid",
        "oracle",
        "flink",
        "exasol",
        "databricks",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notyet(
    ["pyspark"],
    raises=(PySparkPythonException, NotImplementedError),
)
def test_aggregate_multikey_group_reduction_udf(backend, alltypes, df):
    """Tests .aggregate() on a multi-key group_by with a reduction
    operation."""
    with pytest.warns(FutureWarning, match="v9.0"):

        @reduction(
            input_type=[dt.double],
            output_type=dt.Struct({"mean": dt.double, "std": dt.double}),
        )
        def mean_and_std(v):
            return v.mean(), v.std()

    grouping_key_cols = ["bigint_col", "int_col"]

    expr1 = alltypes.group_by(grouping_key_cols).aggregate(
        mean_and_std(alltypes["double_col"]).destructure()
    )

    result1 = expr1.execute()

    # Note: Using `reset_index` to get the grouping key as a column
    expected = (
        df.groupby(grouping_key_cols)["double_col"].agg(["mean", "std"]).reset_index()
    )

    # Row ordering may differ depending on backend, so sort on the
    # grouping key
    result1 = result1.sort_values(by=grouping_key_cols).reset_index(drop=True)
    expected = (
        expected.sort_values(by=grouping_key_cols)
        .reset_index(drop=True)
        .assign(int_col=lambda df: df.int_col.astype("int32"))
    )

    backend.assert_frame_equal(result1, expected)


@pytest.mark.parametrize(
    ("result_fn", "expected_fn"),
    [
        param(
            lambda t, where: t.bool_col.count(where=where),
            lambda t, where: len(t.bool_col[where].dropna()),
            id="count",
        ),
        param(
            lambda t, where: t.bool_col.nunique(where=where),
            lambda t, where: t.bool_col[where].dropna().nunique(),
            id="nunique",
        ),
        param(
            lambda t, where: t.bool_col.any(where=where),
            lambda t, where: t.bool_col[where].any(),
            id="any",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=AttributeError,
                    reason="'IntegerColumn' object has no attribute 'any'",
                ),
            ],
        ),
        param(
            lambda t, where: t.bool_col.notany(where=where),
            lambda t, where: ~t.bool_col[where].any(),
            id="notany",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=AttributeError,
                    reason="'IntegerColumn' object has no attribute 'notany'",
                ),
            ],
        ),
        param(
            lambda t, where: ~t.bool_col.any(where=where),
            lambda t, where: ~t.bool_col[where].any(),
            id="any_negate",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=AttributeError,
                    reason="'IntegerColumn' object has no attribute 'any'",
                ),
            ],
        ),
        param(
            lambda t, where: t.bool_col.all(where=where),
            lambda t, where: t.bool_col[where].all(),
            id="all",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=AttributeError,
                    reason="'IntegerColumn' object has no attribute 'all'",
                ),
            ],
        ),
        param(
            lambda t, where: t.bool_col.notall(where=where),
            lambda t, where: ~t.bool_col[where].all(),
            id="notall",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=AttributeError,
                    reason="'IntegerColumn' object has no attribute 'notall'",
                ),
            ],
        ),
        param(
            lambda t, where: ~t.bool_col.all(where=where),
            lambda t, where: ~t.bool_col[where].all(),
            id="all_negate",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=AttributeError,
                    reason="'IntegerColumn' object has no attribute 'all'",
                ),
            ],
        ),
        param(
            lambda t, where: t.double_col.sum(where=where),
            lambda t, where: t.double_col[where].sum(),
            id="sum",
        ),
        param(
            lambda t, where: (t.int_col > 0).sum(where=where),
            lambda t, where: (t.int_col > 0)[where].sum(),
            id="bool_sum",
            marks=[
                pytest.mark.notimpl(
                    ["oracle"],
                    raises=OracleDatabaseError,
                    reason="ORA-02000: missing AS keyword",
                ),
            ],
        ),
        param(
            lambda t, where: t.double_col.mean(where=where),
            lambda t, where: t.double_col[where].mean(),
            id="mean",
        ),
        param(
            lambda t, where: t.double_col.min(where=where),
            lambda t, where: t.double_col[where].min(),
            id="min",
        ),
        param(
            lambda t, where: t.double_col.max(where=where),
            lambda t, where: t.double_col[where].max(),
            id="max",
        ),
        param(
            # int_col % 3 so there are no ties for most common value
            lambda t, where: (t.int_col % 3).mode(where=where),
            lambda t, where: (t.int_col % 3)[where].mode().iloc[0],
            id="mode",
            marks=[
                pytest.mark.notyet(
                    [
                        "bigquery",
                        "clickhouse",
                        "datafusion",
                        "impala",
                        "mysql",
                        "pyspark",
                        "mssql",
                        "trino",
                        "druid",
                        "exasol",
                        "flink",
                        "risingwave",
                    ],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        param(
            lambda t, where: t.double_col.argmin(t.int_col, where=where),
            lambda t, where: t.double_col[where].iloc[t.int_col[where].argmin()],
            id="argmin",
            marks=[
                pytest.mark.notyet(
                    [
                        "impala",
                        "mysql",
                        "mssql",
                        "druid",
                        "oracle",
                        "exasol",
                        "flink",
                    ],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        param(
            lambda t, where: t.double_col.argmax(t.int_col, where=where),
            lambda t, where: t.double_col[where].iloc[t.int_col[where].argmax()],
            id="argmax",
            marks=[
                pytest.mark.notyet(
                    [
                        "impala",
                        "mysql",
                        "mssql",
                        "druid",
                        "oracle",
                        "exasol",
                        "flink",
                    ],
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        param(
            lambda t, where: t.double_col.std(how="sample", where=where),
            lambda t, where: t.double_col[where].std(ddof=1),
            id="std",
            marks=[pytest.mark.notimpl(["druid"], raises=com.OperationNotDefinedError)],
        ),
        param(
            lambda t, where: t.double_col.var(how="sample", where=where),
            lambda t, where: t.double_col[where].var(ddof=1),
            id="var",
            marks=[pytest.mark.notimpl(["druid"], raises=com.OperationNotDefinedError)],
        ),
        param(
            lambda t, where: t.double_col.std(how="pop", where=where),
            lambda t, where: t.double_col[where].std(ddof=0),
            id="std_pop",
            marks=[pytest.mark.notimpl(["druid"], raises=com.OperationNotDefinedError)],
        ),
        param(
            lambda t, where: t.double_col.var(how="pop", where=where),
            lambda t, where: t.double_col[where].var(ddof=0),
            id="var_pop",
            marks=[pytest.mark.notimpl(["druid"], raises=com.OperationNotDefinedError)],
        ),
        param(
            lambda t, where: t.string_col.approx_nunique(where=where),
            lambda t, where: t.string_col[where].nunique(),
            id="approx_nunique",
            marks=[
                pytest.mark.xfail_version(
                    duckdb=["duckdb>=1.1"],
                    raises=AssertionError,
                    reason="not exact, even at this tiny scale",
                    strict=False,
                ),
                pytest.mark.notimpl(
                    ["datafusion"],
                    reason="data type is not supported",
                    raises=Exception,
                ),
            ],
        ),
        param(
            lambda t, where: t.bigint_col.bit_and(where=where),
            lambda t, where: np.bitwise_and.reduce(t.bigint_col[where].values),
            id="bit_and",
            marks=[
                pytest.mark.notimpl(
                    ["polars", "mssql", "exasol"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notimpl(["druid"], strict=False, raises=AssertionError),
                pytest.mark.notyet(
                    ["impala", "pyspark", "flink"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda t, where: t.bigint_col.bit_or(where=where),
            lambda t, where: np.bitwise_or.reduce(t.bigint_col[where].values),
            id="bit_or",
            marks=[
                pytest.mark.notimpl(
                    ["polars", "mssql", "exasol"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notyet(
                    ["impala", "pyspark", "flink"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda t, where: t.bigint_col.bit_xor(where=where),
            lambda t, where: np.bitwise_xor.reduce(t.bigint_col[where].values),
            id="bit_xor",
            marks=[
                pytest.mark.notimpl(
                    ["polars", "mssql", "exasol"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notyet(
                    ["impala", "pyspark", "flink"], raises=com.OperationNotDefinedError
                ),
            ],
        ),
        param(
            lambda t, where: t.count(where=where),
            lambda t, where: len(t[where]),
            id="count_star",
        ),
    ],
)
@pytest.mark.parametrize(
    ("ibis_cond", "pandas_cond"),
    [
        param(lambda _: None, lambda _: slice(None), id="no_cond"),
        param(
            lambda t: t.string_col.isin(["1", "7"]),
            lambda t: t.string_col.isin(["1", "7"]),
            id="is_in",
        ),
    ],
)
def test_reduction_ops(
    backend,
    alltypes,
    df,
    result_fn,
    expected_fn,
    ibis_cond,
    pandas_cond,
):
    # Operate on a subset of the data, since aggregations like var/std with
    # sample/population can be too numerically similar for a larger number of
    # rows.
    alltypes = alltypes.filter(alltypes.id < 1550)
    df = df[df.id < 1550]

    expr = alltypes.agg(tmp=result_fn(alltypes, ibis_cond(alltypes))).tmp
    result = expr.execute().squeeze()
    expected = expected_fn(df, pandas_cond(df))

    try:
        np.testing.assert_allclose(result, expected, rtol=backend.reduction_tolerance)
    except TypeError:  # assert_allclose only handles numerics
        # if we're not testing numerics, then the arrays should be exactly equal
        np.testing.assert_array_equal(result, expected)


@pytest.mark.notimpl(
    ["druid", "impala", "mssql", "mysql", "oracle"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["risingwave"],
    raises=com.UnsupportedOperationError,
    reason="risingwave requires an `order_by` for these aggregations",
)
@pytest.mark.parametrize("method", ["first", "last"])
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize(
    "include_null",
    [
        False,
        param(
            True,
            marks=[
                pytest.mark.notimpl(
                    [
                        "clickhouse",
                        "exasol",
                        "flink",
                        "postgres",
                        "risingwave",
                        "snowflake",
                    ],
                    raises=com.UnsupportedOperationError,
                    reason="`include_null=True` is not supported",
                ),
                pytest.mark.notimpl(
                    ["bigquery", "pyspark", "databricks"],
                    raises=com.UnsupportedOperationError,
                    reason="Can't mix `where` and `include_null=True`",
                    strict=False,
                ),
            ],
        ),
    ],
)
def test_first_last(alltypes, method, filtered, include_null):
    # `first` and `last` effectively choose an arbitrary value when no
    # additional order is specified. *Most* backends will result in the
    # first/last element in a column being selected (at least when operating on
    # a leaf table), but that's really not guaranteed. These operations need an
    # order to be meaningful.
    #
    # To sanely test this we create a column that is a mix of nulls and a
    # single value (or a single value after filtering is applied).
    if filtered:
        new = alltypes.int_col.cases((3, 30), (4, 40))
        where = _.int_col == 3
    else:
        new = (alltypes.int_col == 3).ifelse(30, None)
        where = None

    t = alltypes.mutate(new=new)

    expr = getattr(t.new, method)(where=where, include_null=include_null)
    res = expr.execute()
    if include_null:
        # no ordering, so technically could be either 30 or NULL
        assert res == 30 or pd.isna(res)
    else:
        assert res == 30


@pytest.mark.notimpl(
    ["clickhouse", "exasol", "flink", "pyspark", "sqlite", "databricks"],
    raises=com.UnsupportedOperationError,
)
@pytest.mark.notimpl(
    ["druid", "impala", "mssql", "mysql", "oracle"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.parametrize("method", ["first", "last"])
@pytest.mark.parametrize("filtered", [False, True], ids=["not-filtered", "filtered"])
@pytest.mark.parametrize(
    "include_null",
    [
        param(False, id="exclude-null"),
        param(
            True,
            marks=[
                pytest.mark.notimpl(
                    ["clickhouse", "exasol", "flink", "postgres", "snowflake"],
                    raises=com.UnsupportedOperationError,
                    reason="`include_null=True` is not supported",
                ),
                pytest.mark.notimpl(
                    ["bigquery"],
                    raises=com.UnsupportedOperationError,
                    reason="Can't mix `where` and `include_null=True`",
                    strict=False,
                ),
            ],
            id="include-null",
        ),
    ],
)
def test_first_last_ordered(alltypes, method, filtered, include_null):
    t = alltypes.mutate(new=alltypes.int_col.nullif(0).nullif(9))
    if filtered:
        where = _.int_col != (1 if method == "last" else 8)
        sol = 2 if method == "last" else 7
    else:
        where = None
        sol = 1 if method == "last" else 8

    expr = getattr(t.new, method)(
        where=where, order_by=t.int_col.desc(), include_null=include_null
    )
    res = expr.execute()
    if include_null:
        assert pd.isna(res)
    else:
        assert res == sol


@pytest.mark.notimpl(
    [
        "druid",
        "exasol",
        "flink",
        "impala",
        "mssql",
        "mysql",
        "oracle",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.parametrize("method", ["argmin", "argmax"])
@pytest.mark.parametrize("filtered", [True, False], ids=["filtered", "unfiltered"])
@pytest.mark.parametrize("null_result", [True, False], ids=["null", "non-null"])
def test_argmin_argmax(alltypes, method, filtered, null_result):
    t = alltypes.mutate(by_col=_.int_col.nullif(0).nullif(9), val_col=10 * _.int_col)

    if filtered:
        where = _.int_col != (1 if method == "argmin" else 8)
        sol = 20 if method == "argmin" else 70
    else:
        where = None
        sol = 10 if method == "argmin" else 80

    if null_result:
        t = t.mutate(val_col=_.val_col.nullif(sol))

    expr = getattr(t.val_col, method)("by_col", where=where)
    res = expr.execute()
    assert pd.isna(res) if null_result else res == sol


@pytest.mark.notimpl(
    [
        "impala",
        "mysql",
        "mssql",
        "druid",
        "oracle",
        "exasol",
        "flink",
        "risingwave",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.parametrize("filtered", [False, True])
def test_arbitrary(alltypes, filtered):
    # Arbitrary chooses a non-null arbitrary value. To ensure we can test for
    # _something_ we create a column that is a mix of nulls and a single value
    # (or a single value after filtering is applied).
    if filtered:
        new = alltypes.int_col.cases((3, 30), (4, 40))
        where = _.int_col == 3
    else:
        new = (alltypes.int_col == 3).ifelse(30, None)
        where = None

    t = alltypes.mutate(new=new)

    expr = t.new.arbitrary(where=where)
    res = expr.execute()
    assert res == 30


@pytest.mark.parametrize(
    ("ibis_cond", "pandas_cond"),
    [
        param(lambda _: None, lambda _: slice(None), id="no_cond"),
        param(
            lambda t: t.string_col.isin(["1", "7"]),
            lambda t: t.string_col.isin(["1", "7"]),
            id="cond",
            marks=[
                pytest.mark.notyet(
                    ["mysql"],
                    raises=com.UnsupportedOperationError,
                    reason="backend does not support filtered count distinct with more than one column",
                ),
            ],
        ),
    ],
)
@pytest.mark.notyet(
    ["druid", "mssql", "oracle", "sqlite", "flink"],
    raises=(
        OracleDatabaseError,
        com.UnsupportedOperationError,
        com.OperationNotDefinedError,
    ),
    reason="backend doesn't support count distinct with multiple columns",
)
@pytest.mark.notyet(
    ["datafusion"],
    raises=com.OperationNotDefinedError,
    reason="no one has attempted implementation yet",
)
def test_count_distinct_star(alltypes, df, ibis_cond, pandas_cond):
    table = alltypes[["int_col", "double_col", "string_col"]]
    expr = table.nunique(where=ibis_cond(table))
    result = expr.execute()
    df = df[["int_col", "double_col", "string_col"]]
    expected = len(df.loc[pandas_cond(df)].drop_duplicates())
    assert int(result) == int(expected)


@pytest.mark.parametrize(
    ("result_fn", "expected_fn"),
    [
        param(
            lambda t, where: t.double_col.quantile(0.5, where=where),
            lambda t, where: t.double_col[where].quantile(0.5),
            id="quantile",
            marks=[
                pytest.mark.notimpl(
                    [
                        "bigquery",
                        "datafusion",
                        "impala",
                        "mssql",
                        "mysql",
                        "sqlite",
                        "druid",
                    ],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.never(
                    ["trino"],
                    reason="backend implements approximate quantiles",
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.never(
                    ["flink"],
                    reason="backend doesn't implement approximate quantiles yet",
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
        param(
            lambda t, where: t.double_col.quantile([0.5], where=where),
            lambda t, where: t.double_col[where].quantile([0.5]),
            id="multi-quantile",
            marks=[
                pytest.mark.notimpl(
                    [
                        "bigquery",
                        "datafusion",
                        "polars",
                        "druid",
                        "oracle",
                    ],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notyet(
                    ["mysql", "mssql", "impala", "exasol", "sqlite"],
                    raises=com.UnsupportedBackendType,
                ),
                pytest.mark.notyet(
                    ["snowflake", "risingwave"],
                    reason="backend doesn't implement array of quantiles as input",
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.never(
                    ["trino"],
                    reason="backend implements approximate quantiles",
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.never(
                    ["flink"],
                    reason="backend doesn't implement approximate quantiles yet",
                    raises=com.OperationNotDefinedError,
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    ("ibis_cond", "pandas_cond"),
    [
        param(lambda _: None, lambda _: slice(None), id="no_cond"),
        param(
            lambda t: t.string_col.isin(["1", "7"]),
            lambda t: t.string_col.isin(["1", "7"]),
            id="is_in",
            marks=[
                pytest.mark.notimpl(["datafusion"], raises=com.OperationNotDefinedError)
            ],
        ),
    ],
)
def test_quantile(
    alltypes,
    df,
    result_fn,
    expected_fn,
    ibis_cond,
    pandas_cond,
):
    expr = alltypes.agg(tmp=result_fn(alltypes, ibis_cond(alltypes))).tmp
    result = expr.execute().squeeze()
    expected = expected_fn(df, pandas_cond(df))
    assert pytest.approx(result) == expected


@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize(
    "multi",
    [
        False,
        param(
            True,
            marks=[
                pytest.mark.notimpl(
                    ["datafusion", "oracle", "snowflake", "polars", "risingwave"],
                    raises=com.OperationNotDefinedError,
                    reason="multi-quantile not yet implemented",
                ),
                pytest.mark.notyet(
                    ["mssql", "exasol"],
                    raises=com.UnsupportedBackendType,
                    reason="array types not supported",
                ),
            ],
        ),
    ],
)
@pytest.mark.notyet(
    ["druid", "flink", "impala", "mysql", "sqlite"],
    raises=(com.OperationNotDefinedError, com.UnsupportedBackendType),
    reason="quantiles (approximate or otherwise) not supported",
)
def test_approx_quantile(con, filtered, multi):
    t = ibis.memtable({"x": [0, 25, 25, 50, 75, 75, 100, 125, 125, 150, 175, 175, 200]})
    where = t.x <= 100 if filtered else None
    q = [0.25, 0.75] if multi else 0.25
    res = con.execute(t.x.approx_quantile(q, where=where))
    if multi:
        assert isinstance(res, list)
        assert all(pd.api.types.is_float(r) for r in res)
        sol = [25, 75] if filtered else [50, 150]
    else:
        assert pd.api.types.is_float(res)
        sol = 25 if filtered else 50

    # Give pretty wide bounds for approximation - we're mostly testing that
    # the call is valid and the filtering logic is applied properly.
    assert res == pytest.approx(sol, abs=10)


@pytest.mark.parametrize(
    ("result_fn", "expected_fn"),
    [
        param(
            lambda t, where: t.G.cov(t.RBI, where=where, how="pop"),
            lambda t, where: t.G[where].cov(t.RBI[where], ddof=0),
            id="covar_pop",
            marks=[
                pytest.mark.notimpl(
                    ["polars", "druid"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notyet(
                    ["mysql", "impala", "sqlite", "flink"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="function covar_pop(integer, integer) does not exist",
                ),
            ],
        ),
        param(
            lambda t, where: t.G.cov(t.RBI, where=where, how="sample"),
            lambda t, where: t.G[where].cov(t.RBI[where], ddof=1),
            id="covar_samp",
            marks=[
                pytest.mark.notimpl(
                    ["polars", "druid"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notyet(
                    ["mysql", "impala", "sqlite", "flink"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="function covar_pop(integer, integer) does not exist",
                ),
            ],
        ),
        param(
            lambda t, where: t.G.corr(t.RBI, where=where, how="pop"),
            lambda t, where: t.G[where].corr(t.RBI[where]),
            id="corr_pop",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notyet(
                    ["impala", "mysql", "sqlite", "flink"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notyet(
                    ["clickhouse"],
                    raises=(ValueError, AttributeError),
                    reason="ClickHouse only implements `sample` correlation coefficient",
                ),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="function covar_pop(integer, integer) does not exist",
                ),
            ],
        ),
        param(
            lambda t, where: t.G.corr(t.RBI, where=where, how="sample"),
            lambda t, where: t.G[where].corr(t.RBI[where]),
            id="corr_samp",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notyet(
                    ["postgres", "duckdb", "snowflake", "risingwave", "exasol"],
                    raises=com.UnsupportedOperationError,
                    reason="backend only implements population correlation coefficient",
                ),
                pytest.mark.notyet(
                    ["impala", "mysql", "sqlite", "flink"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notyet(
                    ["bigquery"],
                    raises=ValueError,
                    reason="Correlation with how='sample' is not supported.",
                ),
                pytest.mark.notyet(
                    [
                        "risingwave",
                        "snowflake",
                        "oracle",
                    ],
                    raises=ValueError,
                    reason="XXXXSQLExprTranslator only implements population correlation coefficient",
                ),
                pytest.mark.notyet(["trino"], raises=com.UnsupportedOperationError),
            ],
        ),
        param(
            lambda t, where: (t.G > 34.0).cov(
                t.G <= 34.0,
                where=where,
                how="pop",
            ),
            lambda t, where: (t.G[where] > 34.0).cov(t.G[where] <= 34.0, ddof=0),
            id="covar_pop_bool",
            marks=[
                pytest.mark.notimpl(
                    ["polars", "druid"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notyet(
                    ["mysql", "impala", "sqlite", "flink"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="function covar_pop(integer, integer) does not exist",
                ),
            ],
        ),
        param(
            lambda t, where: (t.G > 34.0).corr(
                t.G <= 34.0,
                where=where,
                how="pop",
            ),
            lambda t, where: (t.G[where] > 34.0).corr(t.G[where] <= 34.0),
            id="corr_pop_bool",
            marks=[
                pytest.mark.notimpl(
                    ["druid"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notyet(
                    ["impala", "mysql", "sqlite", "flink"],
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.notyet(
                    ["clickhouse"],
                    raises=ValueError,
                    reason="ClickHouse only implements `sample` correlation coefficient",
                ),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="function covar_pop(integer, integer) does not exist",
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    ("ibis_cond", "pandas_cond"),
    [
        param(lambda _: None, lambda _: slice(None), id="no_cond"),
        param(
            lambda t: t.yearID.isin([2009, 2015]),
            lambda t: t.yearID.isin([2009, 2015]),
            id="cond",
        ),
    ],
)
@pytest.mark.notimpl(["mssql"], raises=com.OperationNotDefinedError)
def test_corr_cov(
    con,
    batting,
    batting_df,
    result_fn,
    expected_fn,
    ibis_cond,
    pandas_cond,
):
    expr = result_fn(batting, ibis_cond(batting)).name("tmp")
    result = expr.execute()

    expected = expected_fn(batting_df, pandas_cond(batting_df))

    # Backends use different algorithms for computing covariance each with
    # different amounts of numerical stability.
    #
    # This makes a generic, precise and accurate comparison function incredibly
    # fragile and tedious to write.
    assert (
        # polars seems to have a numerically-imprecise-compared-to-pandas
        # (though perhaps fast!) implementation of correlation and covariance
        pytest.approx(result, abs=1e-2 if con.name == "polars" else None) == expected
    )


@pytest.mark.notimpl(
    ["mysql", "sqlite", "mssql", "druid"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notyet(["flink"], raises=com.OperationNotDefinedError)
def test_approx_median(alltypes):
    expr = alltypes.double_col.approx_median()
    result = expr.execute()
    assert isinstance(result, float)


@pytest.mark.notimpl(
    ["bigquery", "druid", "sqlite"], raises=com.OperationNotDefinedError
)
@pytest.mark.notyet(
    ["impala", "mysql", "mssql", "druid", "trino"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.never(
    ["flink"],
    reason="backend doesn't implement approximate quantiles yet",
    raises=com.OperationNotDefinedError,
)
def test_median(alltypes, df):
    expr = alltypes.double_col.median()
    result = expr.execute()
    expected = df.double_col.median()
    assert result == expected


@pytest.mark.notimpl(
    ["bigquery", "druid", "sqlite"], raises=com.OperationNotDefinedError
)
@pytest.mark.notyet(
    ["impala", "mysql", "mssql", "trino", "flink"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notyet(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason="doesn't support median of strings",
)
@pytest.mark.notyet(
    ["pyspark"], raises=AssertionError, reason="pyspark returns null for string median"
)
@pytest.mark.notyet(
    ["databricks"],
    raises=DatabricksServerOperationError,
    reason="percentile of string is not allowed",
)
@pytest.mark.notyet(
    ["snowflake"],
    raises=SnowflakeProgrammingError,
    reason="doesn't support median of strings",
)
@pytest.mark.notyet(
    ["exasol"],
    raises=ExaQueryError,
    reason="doesn't support quantile on strings",
)
@pytest.mark.notyet(["polars"], raises=PolarsInvalidOperationError)
@pytest.mark.notyet(["datafusion"], raises=Exception, reason="not supported upstream")
@pytest.mark.parametrize(
    "func",
    [
        param(methodcaller("quantile", 0.5), id="quantile"),
        param(
            methodcaller("median"),
            id="median",
            marks=[
                pytest.mark.notyet(
                    ["oracle"],
                    raises=OracleDatabaseError,
                    reason="doesn't support median of strings",
                )
            ],
        ),
    ],
)
def test_string_quantile(alltypes, func):
    expr = func(alltypes.select(col=ibis.literal("a")).limit(5).col)
    result = expr.execute()
    assert result == "a"


@pytest.mark.notimpl(
    ["bigquery", "sqlite", "druid"], raises=com.OperationNotDefinedError
)
@pytest.mark.notyet(
    ["impala", "mysql", "mssql", "trino", "exasol", "flink"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notyet(
    ["snowflake"],
    raises=SnowflakeProgrammingError,
    reason="doesn't support median of dates",
)
@pytest.mark.notyet(
    ["pyspark"],
    raises=PySparkAnalysisException,
    reason="doesn't support quantile on dates",
)
@pytest.mark.notyet(
    ["exasol"],
    raises=ExaQueryError,
    reason="doesn't support quantile on dates",
)
@pytest.mark.notyet(["datafusion"], raises=Exception, reason="not supported upstream")
@pytest.mark.notyet(
    ["polars"], raises=PolarsInvalidOperationError, reason="not supported upstream"
)
@pytest.mark.notyet(
    ["databricks"],
    raises=DatabricksServerOperationError,
    reason="percentile of string is not allowed",
)
def test_date_quantile(alltypes):
    expr = alltypes.timestamp_col.date().quantile(0.5)
    result = expr.execute()
    assert result == date(2009, 12, 31)


@pytest.mark.parametrize(
    ("ibis_sep", "pandas_sep"),
    [
        param(":", ":", id="const"),
        param(
            L(":") + ":",
            "::",
            id="expr",
            marks=[
                pytest.mark.notyet(["mysql"], raises=com.UnsupportedOperationError),
                pytest.mark.notyet(
                    ["bigquery"],
                    raises=GoogleBadRequest,
                    reason="Argument 2 to STRING_AGG must be a literal or query parameter",
                ),
                pytest.mark.notimpl(
                    ["polars"],
                    raises=com.UnsupportedArgumentError,
                    reason="polars doesn't support expression separators",
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    ("ibis_cond", "pandas_cond"),
    [
        param(lambda _: None, lambda _: slice(None), id="no_cond"),
        param(
            lambda t: t.string_col.isin(["1", "7"]),
            lambda t: t.string_col.isin(["1", "7"]),
            id="is_in",
        ),
        param(
            lambda t: t.string_col.notin(["1", "7"]),
            lambda t: ~t.string_col.isin(["1", "7"]),
            id="not_in",
        ),
    ],
)
@pytest.mark.notyet(["flink"], raises=Py4JJavaError)
def test_group_concat(
    backend, alltypes, df, ibis_cond, pandas_cond, ibis_sep, pandas_sep
):
    expr = (
        alltypes.group_by("bigint_col")
        .aggregate(
            tmp=lambda t: t.string_col.group_concat(ibis_sep, where=ibis_cond(t))
        )
        .order_by("bigint_col")
    )
    result = expr.execute()
    expected = (
        (
            df
            if isinstance(pandas_cond(df), slice)
            else df.assign(string_col=df.string_col.where(pandas_cond(df)))
        )
        .groupby("bigint_col")
        .string_col.agg(
            lambda s: (np.nan if pd.isna(s).all() else pandas_sep.join(s.values))
        )
        .rename("tmp")
        .sort_index()
        .reset_index()
    )

    backend.assert_frame_equal(
        result.replace(np.nan, None), expected.replace(np.nan, None)
    )


@pytest.mark.notimpl(
    [
        "clickhouse",
        "datafusion",
        "druid",
        "flink",
        "impala",
        "pyspark",
        "sqlite",
        "databricks",
    ],
    raises=com.UnsupportedOperationError,
)
@pytest.mark.parametrize("filtered", [False, True])
def test_group_concat_ordered(alltypes, df, filtered):
    ibis_cond = (_.id % 13 == 0) if filtered else None
    pd_cond = (df.id % 13 == 0) if filtered else True
    expr = (
        alltypes.filter(_.bigint_col == 10)
        .id.cast("str")
        .group_concat(":", where=ibis_cond, order_by=_.id.desc())
    )
    result = expr.execute()
    expected = ":".join(
        df.id[(df.bigint_col == 10) & pd_cond].sort_values(ascending=False).astype(str)
    )
    assert result == expected


def gen_test_collect_marks(distinct, filtered, ordered, include_null):
    """The marks for this test fail for different combinations of parameters.
    Rather than set `strict=False` (which can let bugs sneak through), we split
    the mark generation into a function"""
    if distinct:
        yield pytest.mark.notimpl(["datafusion"], raises=com.UnsupportedOperationError)
    if ordered:
        yield pytest.mark.notimpl(
            ["clickhouse", "pyspark", "flink", "databricks"],
            raises=com.UnsupportedOperationError,
        )
    if include_null:
        yield pytest.mark.notimpl(
            ["clickhouse", "pyspark", "snowflake", "databricks"],
            raises=com.UnsupportedOperationError,
        )

    # Handle special cases
    if filtered and distinct:
        yield pytest.mark.notimpl(
            ["bigquery", "snowflake"],
            raises=com.UnsupportedOperationError,
            reason="Can't combine where and distinct",
        )
    elif filtered and include_null:
        yield pytest.mark.notimpl(
            ["bigquery"],
            raises=com.UnsupportedOperationError,
            reason="Can't combine where and include_null",
        )
    elif include_null:
        yield pytest.mark.notimpl(
            ["bigquery"],
            raises=GoogleBadRequest,
            reason="BigQuery can't retrieve arrays with null values",
        )


@pytest.mark.notimpl(
    ["druid", "exasol", "impala", "mssql", "mysql", "oracle", "sqlite"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.parametrize(
    "distinct, filtered, ordered, include_null",
    [
        param(*ps, marks=list(gen_test_collect_marks(*ps)))
        for ps in itertools.product(*([[True, False]] * 4))
    ],
)
def test_collect(alltypes, df, distinct, filtered, ordered, include_null):
    expr = alltypes.mutate(x=_.string_col.nullif("3")).x.collect(
        where=((_.id % 13 == 0) if filtered else None),
        include_null=include_null,
        distinct=distinct,
        order_by=(_.x.desc() if ordered else ()),
    )
    res = expr.execute()

    x = df.string_col.where(df.string_col != "3", None)
    if filtered:
        x = x[df.id % 13 == 0]
    if not include_null:
        x = x.dropna()
    if distinct:
        x = x.drop_duplicates()
    sol = sorted(x, key=lambda x: (x is not None, x), reverse=True)

    if not ordered:
        # If unordered, order afterwards so we can compare
        res = sorted(res, key=lambda x: (x is not None, x), reverse=True)

    assert res == sol


@pytest.mark.notimpl(["mssql"], raises=PyODBCProgrammingError)
def test_topk_op(alltypes, df):
    # TopK expression will order rows by "count" but each backend
    # can have different result for that.
    # Note: Maybe would be good if TopK could order by "count"
    # and the field used by TopK
    t = alltypes.order_by(alltypes.string_col)
    df = df.sort_values("string_col")
    expr = t.string_col.topk(3)
    result = expr.execute()
    expected = df.groupby("string_col")["string_col"].count().head(3)
    assert all(result.iloc[:, 1].values == expected.values)


@pytest.mark.parametrize(
    ("result_fn", "expected_fn"),
    [
        param(
            lambda t: t.semi_join(t.string_col.topk(3), "string_col"),
            lambda t: t[
                t.string_col.isin(
                    t.groupby("string_col")["string_col"].count().head(3).index
                )
            ],
            id="string_col_filter_top3",
            marks=pytest.mark.notimpl(["mssql"], raises=PyODBCProgrammingError),
        )
    ],
)
@pytest.mark.notyet(
    ["druid"], raises=PyDruidProgrammingError, reason="Java NullPointerException"
)
@pytest.mark.notyet(
    ["flink"], raises=Py4JError, reason="Flink doesn't support semi joins"
)
def test_topk_filter_op(con, alltypes, df, result_fn, expected_fn):
    # TopK expression will order rows by "count" but each backend
    # can have different result for that.
    # Note: Maybe would be good if TopK could order by "count"
    # and the field used by TopK
    if con.name == "sqlite":
        # TODO: remove after CTE extraction is reimplemented
        pytest.skip("topk -> semi-join performance has increased post SQLGlot refactor")
    t = alltypes.order_by(alltypes.string_col)
    df = df.sort_values("string_col")
    expr = result_fn(t)
    result = expr.execute()
    expected = expected_fn(df)
    assert result.shape[0] == expected.shape[0]


@pytest.mark.parametrize(
    "agg_fn", [lambda s: list(s), lambda s: np.array(s)], ids=["list", "ndarray"]
)
@pytest.mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "datafusion",
        "duckdb",
        "impala",
        "mysql",
        "postgres",
        "risingwave",
        "sqlite",
        "snowflake",
        "polars",
        "mssql",
        "trino",
        "druid",
        "oracle",
        "exasol",
        "flink",
        "databricks",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notyet(
    ["pyspark"],
    condition=IS_SPARK_REMOTE,
    raises=PySparkPythonException,
    reason="remote udfs not yet tested due to environment complexities",
)
def test_aggregate_list_like(backend, alltypes, df, agg_fn):
    """Tests .aggregate() where the result of an aggregation is a list-like.

    We expect the list / np.array to be treated as a scalar (in other
    words, the resulting table expression should have one element, which
    is the list / np.array).
    """
    with pytest.warns(FutureWarning, match="v9.0"):
        udf = reduction(input_type=[dt.double], output_type=dt.Array(dt.double))(agg_fn)

    expr = alltypes.aggregate(result_col=udf(alltypes.double_col))
    result = expr.execute()

    # Expecting a 1-row DataFrame
    expected = pd.DataFrame({"result_col": [agg_fn(df.double_col)]})

    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "datafusion",
        "duckdb",
        "impala",
        "mysql",
        "postgres",
        "risingwave",
        "sqlite",
        "snowflake",
        "polars",
        "mssql",
        "trino",
        "druid",
        "oracle",
        "flink",
        "exasol",
        "flink",
        "databricks",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notyet(
    ["pyspark"],
    condition=IS_SPARK_REMOTE,
    raises=PySparkPythonException,
    reason="remote udfs not yet tested due to environment complexities",
)
def test_aggregate_mixed_udf(backend, alltypes, df):
    """Tests .aggregate() with multiple aggregations with mixed result types.

    (In particular, one aggregation that results in an array, and other
    aggregation(s) that result in a non-array)
    """
    with pytest.warns(FutureWarning, match="v9.0"):

        @reduction(input_type=[dt.double], output_type=dt.double)
        def sum_udf(v):
            return np.sum(v)

        @reduction(input_type=[dt.double], output_type=dt.Array(dt.double))
        def collect_udf(v):
            return np.array(v)

    expr = alltypes.aggregate(
        sum_col=sum_udf(alltypes.double_col),
        collect_udf=collect_udf(alltypes.double_col),
    )
    result = expr.execute()

    expected = pd.DataFrame(
        {
            "sum_col": [sum_udf.func(df.double_col)],
            "collect_udf": [collect_udf.func(df.double_col)],
        }
    )

    backend.assert_frame_equal(result, expected, check_like=True)


def test_binds_are_cast(alltypes):
    expr = alltypes.aggregate(
        high_line_count=alltypes.string_col.cases(("1-URGENT", 1), else_=0).sum()
    )

    expr.execute()


def test_agg_sort(alltypes):
    query = alltypes.aggregate(count=alltypes.count())
    with pytest.raises(com.IntegrityError):
        query.order_by(alltypes.year)


def test_filter(backend, alltypes, df):
    expr = (
        alltypes.filter(_.string_col == "1")
        .mutate(x=L(1, "int64"))
        .group_by(_.x)
        .aggregate(sum=_.double_col.sum())
    )

    # TODO: The pyspark backend doesn't apply schemas to outputs
    result = expr.execute().astype({"x": "int64"})
    expected = (
        df.loc[df.string_col == "1", :]
        .assign(x=1)
        .groupby("x")
        .double_col.sum()
        .rename("sum")
        .reset_index()
    )
    backend.assert_frame_equal(result, expected, check_like=True)


def test_agg_name_in_output_column(alltypes):
    query = alltypes.aggregate([alltypes.int_col.min(), alltypes.int_col.max()])
    df = query.execute()
    assert "min" in df.columns[0].lower()
    assert "max" in df.columns[1].lower()


def test_grouped_case(backend, con):
    table = ibis.memtable({"key": [1, 1, 2, 2], "value": [10, 30, 20, 40]})

    case_expr = ibis.cases((table.value < 25, table.value), else_=ibis.null())

    expr = (
        table.group_by(k="key")
        .aggregate(mx=case_expr.max())
        .drop_null("k")
        .order_by("k")
    )
    result = con.execute(expr)
    expected = pd.DataFrame({"k": [1, 2], "mx": [10, 20]})
    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(
    ["datafusion"],
    raises=BaseException,
    reason="because pyo3 panic exception is raised",
)
@pytest.mark.notyet(["flink"], raises=Py4JJavaError)
@pytest.mark.notyet(["impala"], raises=ImpalaHiveServer2Error)
@pytest.mark.notyet(["clickhouse"], raises=ClickHouseDatabaseError)
@pytest.mark.notyet(["druid"], raises=PyDruidProgrammingError)
@pytest.mark.notyet(["snowflake"], raises=SnowflakeProgrammingError)
@pytest.mark.notyet(["trino"], raises=TrinoUserError)
@pytest.mark.notyet(["mysql"], raises=MySQLNotSupportedError)
@pytest.mark.notyet(["oracle"], raises=OracleDatabaseError)
@pytest.mark.notyet(["pyspark"], raises=PySparkAnalysisException)
@pytest.mark.notyet(["mssql"], raises=PyODBCProgrammingError)
@pytest.mark.notyet(["risingwave"], raises=AssertionError, strict=False)
@pytest.mark.notyet(["databricks"], raises=DatabricksServerOperationError)
def test_group_concat_over_window(backend, con):
    # TODO: this test is flaky on risingwave and I DO NOT LIKE IT
    input_df = pd.DataFrame(
        {
            "s": ["a|b|c", "b|a|c", "b|b|b|c|a"],
            "token": ["a", "b", "c"],
            "pk": [1, 1, 2],
            "id": [1, 2, 3],
        }
    )
    expected = input_df.assign(test=["a|b|c|b|a|c", "b|a|c", "b|b|b|c|a"])

    table = ibis.memtable(input_df)
    expr = table.mutate(
        test=table.s.group_concat(sep="|").over(
            group_by="pk", order_by="id", rows=(0, None)
        )
    ).order_by("id")

    result = con.execute(expr)
    backend.assert_frame_equal(result, expected)


def test_value_counts_on_expr(backend, alltypes, df):
    expr = alltypes.bigint_col.add(1).value_counts()
    columns = list(expr.columns)
    expr = expr.order_by(columns)
    result = expr.execute().sort_values(columns).reset_index(drop=True)
    expected = df.bigint_col.add(1).value_counts().reset_index()
    expected.columns = columns
    expected = expected.sort_values(by=columns).reset_index(drop=True)
    backend.assert_frame_equal(result, expected)


@pytest.mark.xfail_version(datafusion=["datafusion==42"])
def test_group_by_expr(backend, con):
    expr = (
        ibis.memtable(
            {"project_name": ["duckdb", "ibis-framework", "ibis", "numpy", "pandas"]}
        )
        .group_by(n=_.project_name.length())
        .aggregate(c=_.project_name.nunique())
        .order_by(_.n.desc())
    )
    result = con.execute(expr)
    expected = pd.DataFrame(dict(n=[14, 6, 5, 4], c=[1, 2, 1, 1])).astype(
        dict(n="int32", c="int64")
    )
    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "value", [ibis.literal("a"), ibis.null("str")], ids=["string", "null"]
)
@pytest.mark.notyet(
    ["mssql"], raises=PyODBCProgrammingError, reason="not supported by the database"
)
def test_group_by_scalar(alltypes, df, value):
    expr = alltypes.group_by(key=value).agg(n=lambda t: t.count())
    result = expr.execute()
    n = result["n"].values[0].item()
    assert n == len(df)
