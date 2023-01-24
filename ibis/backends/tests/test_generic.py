import decimal
import io
from contextlib import redirect_stdout
from operator import invert, methodcaller, neg

import numpy as np
import pandas as pd
import pytest
import sqlalchemy
import toolz
from pytest import param
from sqlalchemy.exc import ProgrammingError

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
from ibis import _
from ibis import literal as L

try:
    import duckdb

    DuckDBConversionException = duckdb.ConversionException
except ImportError:
    DuckDBConversionException = None

try:
    import clickhouse_driver

    ClickhouseDriverOperationalError = clickhouse_driver.dbapi.errors.OperationalError
except ImportError:
    ClickhouseDriverOperationalError = None


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        param(
            ibis.NA.fillna(5),
            5,
            marks=pytest.mark.notimpl(["mssql"]),
            id="na_fillna",
        ),
        param(
            L(5).fillna(10),
            5,
            marks=pytest.mark.notimpl(["mssql"]),
            id="non_na_fillna",
        ),
        param(L(5).nullif(5), None, id="nullif_null"),
        param(L(10).nullif(5), 10, id="nullif_not_null"),
    ],
)
@pytest.mark.notimpl(["datafusion"])
def test_scalar_fillna_nullif(con, expr, expected):
    if expected is None:
        # The exact kind of null value used differs per backend (and version).
        # Example 1: Pandas returns np.nan while BigQuery returns None.
        # Example 2: PySpark returns np.nan if pyspark==3.0.0, but returns None
        # if pyspark <=3.0.0.
        # TODO: Make this behavior consistent (#2365)
        assert pd.isna(con.execute(expr))
    else:
        assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("col", "filt"),
    [
        param(
            "nan_col",
            _.nan_col.isnan(),
            marks=pytest.mark.notimpl(["datafusion", "mysql", "sqlite"]),
            id="nan_col",
        ),
        param(
            "none_col",
            _.none_col.isnull(),
            marks=[pytest.mark.notimpl(["datafusion", "mysql"])],
            id="none_col",
        ),
    ],
)
@pytest.mark.notimpl(["mssql"])
def test_isna(backend, alltypes, col, filt):
    table = alltypes.select(
        nan_col=ibis.literal(np.nan), none_col=ibis.NA.cast("float64")
    )
    df = table.execute()

    result = table[filt].execute().reset_index(drop=True)
    expected = df[df[col].isna()].reset_index(drop=True)

    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "value",
    [
        None,
        param(
            np.nan,
            marks=pytest.mark.notimpl(
                [
                    "bigquery",
                    "clickhouse",
                    "duckdb",
                    "impala",
                    "postgres",
                    "mysql",
                    "snowflake",
                    "polars",
                    "trino",
                ],
                reason="NaN != NULL for these backends",
            ),
            id="nan_col",
        ),
    ],
)
@pytest.mark.notimpl(["datafusion", "mssql"])
def test_column_fillna(backend, alltypes, value):
    table = alltypes.mutate(missing=ibis.literal(value).cast("float64"))
    pd_table = table.execute()

    res = table.mutate(missing=table.missing.fillna(0.0)).execute()
    sol = pd_table.assign(missing=pd_table.missing.fillna(0.0))
    backend.assert_frame_equal(res.reset_index(drop=True), sol.reset_index(drop=True))


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        param(
            ibis.coalesce(5, None, 4),
            5,
            id="generic",
            marks=[
                pytest.mark.broken(
                    "polars",
                    reason="implementation error, cannot get ref Int8 from Boolean",
                ),
            ],
        ),
        param(ibis.coalesce(ibis.NA, 4, ibis.NA), 4, id="null_start_end"),
        param(
            ibis.coalesce(ibis.NA, ibis.NA, 3.14),
            3.14,
            id="non_null_last",
            marks=pytest.mark.broken(
                "polars",
                reason="implementation error, cannot get ref Int8 from Float64",
            ),
        ),
    ],
)
@pytest.mark.notimpl(["datafusion"])
def test_coalesce(con, expr, expected):
    result = con.execute(expr.name("tmp"))

    if isinstance(result, decimal.Decimal):
        # in case of Impala the result is decimal
        # >>> decimal.Decimal('5.56') == 5.56
        # False
        assert result == decimal.Decimal(str(expected))
    else:
        assert result == pytest.approx(expected)


# TODO(dask) - identicalTo - #2553
@pytest.mark.notimpl(["clickhouse", "datafusion", "polars", "dask", "pyspark", "mssql"])
def test_identical_to(backend, alltypes, sorted_df):
    sorted_alltypes = alltypes.order_by('id')
    df = sorted_df
    dt = df[['tinyint_col', 'double_col']]

    ident = sorted_alltypes.tinyint_col.identical_to(sorted_alltypes.double_col)
    expr = sorted_alltypes['id', ident.name('tmp')].order_by('id')
    result = expr.execute().tmp

    expected = (dt.tinyint_col.isnull() & dt.double_col.isnull()) | (
        dt.tinyint_col == dt.double_col
    )

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('column', 'elements'),
    [
        ('int_col', [1, 2, 3]),
        ('int_col', (1, 2, 3)),
        ('string_col', ['1', '2', '3']),
        ('string_col', ('1', '2', '3')),
        ('int_col', {1}),
        ('int_col', frozenset({1})),
    ],
)
@pytest.mark.notimpl(["mssql"])
def test_isin(backend, alltypes, sorted_df, column, elements):
    sorted_alltypes = alltypes.order_by('id')
    expr = sorted_alltypes[
        'id', sorted_alltypes[column].isin(elements).name('tmp')
    ].order_by('id')
    result = expr.execute().tmp

    expected = sorted_df[column].isin(elements)
    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('column', 'elements'),
    [
        ('int_col', [1, 2, 3]),
        ('int_col', (1, 2, 3)),
        ('string_col', ['1', '2', '3']),
        ('string_col', ('1', '2', '3')),
        ('int_col', {1}),
        ('int_col', frozenset({1})),
    ],
)
@pytest.mark.notimpl(["mssql"])
def test_notin(backend, alltypes, sorted_df, column, elements):
    sorted_alltypes = alltypes.order_by('id')
    expr = sorted_alltypes[
        'id', sorted_alltypes[column].notin(elements).name('tmp')
    ].order_by('id')
    result = expr.execute().tmp

    expected = ~sorted_df[column].isin(elements)
    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('predicate_fn', 'expected_fn'),
    [
        param(
            lambda t: t['bool_col'],
            lambda df: df['bool_col'],
            id="no_op",
            marks=pytest.mark.min_version(datafusion="0.5.0"),
        ),
        param(lambda t: ~t['bool_col'], lambda df: ~df['bool_col'], id="negate"),
        param(
            lambda t: t.bool_col & t.bool_col,
            lambda df: df.bool_col & df.bool_col,
            id="and",
        ),
        param(
            lambda t: t.bool_col | t.bool_col,
            lambda df: df.bool_col | df.bool_col,
            id="or",
        ),
        param(
            lambda t: t.bool_col ^ t.bool_col,
            lambda df: df.bool_col ^ df.bool_col,
            id="xor",
            marks=pytest.mark.notimpl(["datafusion"]),
        ),
    ],
)
def test_filter(backend, alltypes, sorted_df, predicate_fn, expected_fn):
    sorted_alltypes = alltypes.order_by('id')
    table = sorted_alltypes[predicate_fn(sorted_alltypes)].order_by('id')
    result = table.execute()
    expected = sorted_df[expected_fn(sorted_df)]

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
        "sqlite",
        "snowflake",
        "polars",
        "mssql",
        "trino",
    ]
)
def test_filter_with_window_op(backend, alltypes, sorted_df):
    sorted_alltypes = alltypes.order_by('id')
    table = sorted_alltypes
    window = ibis.window(group_by=table.id)
    table = table.filter(lambda t: t['id'].mean().over(window) > 3).order_by('id')
    result = table.execute()
    expected = (
        sorted_df.groupby(['id'])
        .filter(lambda t: t['id'].mean() > 3)
        .reset_index(drop=True)
    )
    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["datafusion"])
def test_case_where(backend, alltypes, df):
    table = alltypes
    table = table.mutate(
        new_col=(
            ibis.case()
            .when(table['int_col'] == 1, 20)
            .when(table['int_col'] == 0, 10)
            .else_(0)
            .end()
            .cast('int64')
        )
    )

    result = table.execute()

    expected = df.copy()
    mask_0 = expected['int_col'] == 1
    mask_1 = expected['int_col'] == 0

    expected['new_col'] = 0
    expected.loc[mask_0, 'new_col'] = 20
    expected.loc[mask_1, 'new_col'] = 10

    backend.assert_frame_equal(result, expected)


# TODO: some of these are notimpl (datafusion) others are probably never
@pytest.mark.notimpl(["datafusion", "mysql", "sqlite", "mssql"])
@pytest.mark.min_version(duckdb="0.3.3", reason="isnan/isinf unsupported")
def test_select_filter_mutate(backend, alltypes, df):
    """Test that select, filter and mutate are executed in right order.

    Before PR #2635, try_fusion in analysis.py would fuse these
    operations together in a way that the order of the operations were
    wrong. (mutate was executed before filter).
    """
    t = alltypes

    # Prepare the float_col so that filter must execute
    # before the cast to get the correct result.
    t = t.mutate(
        float_col=ibis.case().when(t['bool_col'], t['float_col']).else_(np.nan).end()
    )

    # Actual test
    t = t[t.columns]
    t = t[~t['float_col'].isnan()]
    t = t.mutate(float_col=t['float_col'].cast('float64'))
    result = t.execute()

    expected = df.copy()
    expected.loc[~df['bool_col'], 'float_col'] = None
    expected = expected[~expected['float_col'].isna()].reset_index(drop=True)
    expected = expected.assign(float_col=expected['float_col'].astype('float64'))

    backend.assert_series_equal(result.float_col, expected.float_col)


def test_table_fillna_invalid(alltypes):
    with pytest.raises(
        com.IbisTypeError, match=r"Column 'invalid_col' is not found in table"
    ):
        alltypes.fillna({'invalid_col': 0.0})

    with pytest.raises(
        com.IbisTypeError, match="Cannot fillna on column 'string_col' of type.*"
    ):
        alltypes[["int_col", "string_col"]].fillna(0)

    with pytest.raises(
        com.IbisTypeError, match="Cannot fillna on column 'int_col' of type.*"
    ):
        alltypes.fillna({"int_col": "oops"})


@pytest.mark.parametrize(
    "replacements",
    [
        {"int_col": 20},
        {"double_col": -1, "string_col": "missing"},
        {"double_col": -1.5, "string_col": "missing"},
    ],
)
@pytest.mark.notimpl(["datafusion", "mssql", "clickhouse"])
def test_table_fillna_mapping(backend, alltypes, replacements):
    table = alltypes.mutate(
        int_col=alltypes.int_col.nullif(1),
        double_col=alltypes.double_col.nullif(3.0),
        string_col=alltypes.string_col.nullif("2"),
    ).select("id", "int_col", "double_col", "string_col")
    pd_table = table.execute()

    result = table.fillna(replacements).execute().reset_index(drop=True)
    expected = pd_table.fillna(replacements).reset_index(drop=True)

    backend.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.notimpl(["datafusion", "mssql", "clickhouse"])
def test_table_fillna_scalar(backend, alltypes):
    table = alltypes.mutate(
        int_col=alltypes.int_col.nullif(1),
        double_col=alltypes.double_col.nullif(3.0),
        string_col=alltypes.string_col.nullif("2"),
    ).select("id", "int_col", "double_col", "string_col")
    pd_table = table.execute()

    res = table[["int_col", "double_col"]].fillna(0).execute().reset_index(drop=True)
    sol = pd_table[["int_col", "double_col"]].fillna(0).reset_index(drop=True)
    backend.assert_frame_equal(res, sol, check_dtype=False)

    res = table[["string_col"]].fillna("missing").execute().reset_index(drop=True)
    sol = pd_table[["string_col"]].fillna("missing").reset_index(drop=True)
    backend.assert_frame_equal(res, sol, check_dtype=False)


def test_mutate_rename(alltypes):
    table = alltypes.select(["bool_col", "string_col"])
    table = table.mutate(dupe_col=table["bool_col"])
    result = table.execute()
    # check_dtype is False here because there are dtype diffs between
    # Pyspark and Pandas on Java 8 - filling the 'none_col' with an int
    # results in float in Pyspark, and int in Pandas. This diff does
    # not exist in Java 11.
    assert list(result.columns) == ["bool_col", "string_col", "dupe_col"]


def test_dropna_invalid(alltypes):
    with pytest.raises(
        com.IbisTypeError, match=r"Column 'invalid_col' is not found in table"
    ):
        alltypes.dropna(subset=['invalid_col'])

    with pytest.raises(ValueError, match=r".*is not in.*"):
        alltypes.dropna(how='invalid')


@pytest.mark.parametrize(
    'how', ['any', pytest.param('all', marks=pytest.mark.notyet("polars"))]
)
@pytest.mark.parametrize(
    'subset', [None, [], 'col_1', ['col_1', 'col_2'], ['col_1', 'col_3']]
)
@pytest.mark.notimpl(["datafusion"])
def test_dropna_table(backend, alltypes, how, subset):
    is_two = alltypes.int_col == 2
    is_four = alltypes.int_col == 4

    table = alltypes.mutate(
        col_1=is_two.ifelse(ibis.NA, alltypes.float_col),
        col_2=is_four.ifelse(ibis.NA, alltypes.float_col),
        col_3=(is_two | is_four).ifelse(ibis.NA, alltypes.float_col),
    ).select("col_1", "col_2", "col_3")

    table_pandas = table.execute()

    result = table.dropna(subset, how).execute().reset_index(drop=True)
    expected = table_pandas.dropna(how=how, subset=subset).reset_index(drop=True)

    backend.assert_frame_equal(result, expected)


def test_select_sort_sort(alltypes):
    query = alltypes[alltypes.year, alltypes.bool_col]
    query = query.order_by(query.year).order_by(query.bool_col)


@pytest.mark.parametrize(
    "key, df_kwargs",
    [
        param("id", {"by": "id"}),
        param(_.id, {"by": "id"}),
        param(lambda _: _.id, {"by": "id"}),
        param(
            ("id", False),
            {"by": "id", "ascending": False},
            marks=pytest.mark.notimpl(["dask"]),
        ),
        param(
            ibis.desc("id"),
            {"by": "id", "ascending": False},
            marks=pytest.mark.notimpl(["dask"]),
        ),
        param(
            ["id", "int_col"],
            {"by": ["id", "int_col"]},
            marks=pytest.mark.notimpl(["dask"]),
        ),
        param(
            ["id", ("int_col", False)],
            {"by": ["id", "int_col"], "ascending": [True, False]},
            marks=pytest.mark.notimpl(["dask"]),
        ),
    ],
)
def test_order_by(backend, alltypes, df, key, df_kwargs):
    result = alltypes.filter(_.id < 100).order_by(key).execute()
    expected = df.loc[df.id < 100].sort_values(**df_kwargs)
    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["dask", "datafusion", "impala", "pandas", "polars", "mssql"])
@pytest.mark.notyet(
    ["clickhouse"],
    reason="clickhouse doesn't have a [0.0, 1.0) random implementation",
)
def test_order_by_random(alltypes):
    expr = alltypes.filter(_.id < 100).order_by(ibis.random()).limit(5)
    r1 = expr.execute()
    r2 = expr.execute()
    assert len(r1) == 5
    assert len(r2) == 5
    # Ensure that multiple executions returns different results
    assert not r1.equals(r2)


def check_table_info(buf, schema):
    info_str = buf.getvalue()

    assert "Null" in info_str
    assert all(type.__class__.__name__ in info_str for type in schema.types)
    assert all(name in info_str for name in schema.names)


def test_table_info_buf(alltypes):
    buf = io.StringIO()
    alltypes.info(buf=buf)
    check_table_info(buf, alltypes.schema())


def test_table_info_no_buf(alltypes):
    buf = io.StringIO()
    with redirect_stdout(buf):
        alltypes.info()
    check_table_info(buf, alltypes.schema())


@pytest.mark.parametrize(
    ("ibis_op", "pandas_op"),
    [
        param(
            _.string_col.isin([]),
            lambda df: df.string_col.isin([]),
            id="isin",
        ),
        param(
            _.string_col.notin([]),
            lambda df: ~df.string_col.isin([]),
            id="notin",
        ),
        param(
            (_.string_col.length() * 1).isin([1]),
            lambda df: (df.string_col.str.len() * 1).isin([1]),
            id="isin_non_empty",
        ),
        param(
            (_.string_col.length() * 1).notin([1]),
            lambda df: ~(df.string_col.str.len() * 1).isin([1]),
            id="notin_non_empty",
        ),
    ],
)
def test_isin_notin(backend, alltypes, df, ibis_op, pandas_op):
    expr = alltypes[ibis_op]
    expected = df.loc[pandas_op(df)].sort_values(["id"]).reset_index(drop=True)
    result = expr.execute().sort_values(["id"]).reset_index(drop=True)
    backend.assert_frame_equal(result, expected)


@pytest.mark.notyet(
    ["dask"],
    reason="dask doesn't support Series as isin/notin argument",
    raises=NotImplementedError,
)
@pytest.mark.notimpl(["datafusion"])
@pytest.mark.parametrize(
    ("ibis_op", "pandas_op"),
    [
        param(
            _.string_col.isin(_.string_col),
            lambda df: df.string_col.isin(df.string_col),
            id="isin_col",
        ),
        param(
            (_.bigint_col + 1).isin(_.string_col.length() + 1),
            lambda df: df.bigint_col.add(1).isin(df.string_col.str.len().add(1)),
            id="isin_expr",
        ),
        param(
            _.string_col.notin(_.string_col),
            lambda df: ~df.string_col.isin(df.string_col),
            id="notin_col",
        ),
        param(
            (_.bigint_col + 1).notin(_.string_col.length() + 1),
            lambda df: ~(df.bigint_col.add(1)).isin(df.string_col.str.len().add(1)),
            id="notin_expr",
        ),
    ],
)
def test_isin_notin_column_expr(backend, alltypes, df, ibis_op, pandas_op):
    expr = alltypes[ibis_op].order_by("id")
    expected = df[pandas_op(df)].sort_values(["id"]).reset_index(drop=True)
    result = expr.execute()
    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected", "op"),
    [
        param(True, True, toolz.identity, id="true_noop"),
        param(False, False, toolz.identity, id="false_noop"),
        param(
            True, False, invert, id="true_invert", marks=pytest.mark.notimpl(["mssql"])
        ),
        param(
            False, True, invert, id="false_invert", marks=pytest.mark.notimpl(["mssql"])
        ),
        param(True, False, neg, id="true_negate", marks=pytest.mark.notimpl(["mssql"])),
        param(
            False, True, neg, id="false_negate", marks=pytest.mark.notimpl(["mssql"])
        ),
    ],
)
def test_logical_negation_literal(con, expr, expected, op):
    assert con.execute(op(ibis.literal(expr)).name("tmp")) == expected


@pytest.mark.parametrize(
    "op",
    [
        toolz.identity,
        param(invert, marks=pytest.mark.notimpl(["mssql"])),
        param(neg, marks=pytest.mark.notimpl(["mssql"])),
    ],
)
def test_logical_negation_column(backend, alltypes, df, op):
    result = op(alltypes["bool_col"]).name("tmp").execute()
    expected = op(df["bool_col"])
    backend.assert_series_equal(result, expected, check_names=False)


@pytest.mark.notimpl(["datafusion"])
@pytest.mark.parametrize(
    ("dtype", "zero", "expected"),
    [("int64", 0, 1), ("float64", 0.0, 1.0)],
)
def test_zeroifnull_literals(con, dtype, zero, expected):
    assert con.execute(ibis.NA.cast(dtype).zeroifnull()) == zero
    assert con.execute(ibis.literal(expected, type=dtype).zeroifnull()) == expected


@pytest.mark.notimpl(["datafusion"])
@pytest.mark.min_version(
    dask="2022.01.1",
    reason="unsupported operation with later versions of pandas",
)
def test_zeroifnull_column(backend, alltypes, df):
    expr = alltypes.int_col.nullif(1).zeroifnull().name('tmp')
    result = expr.execute().astype("int32")
    expected = df.int_col.replace(1, 0).rename("tmp").astype("int32")
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(["datafusion"])
def test_where_select(backend, alltypes, df):
    table = alltypes
    table = table.select(
        [
            "int_col",
            (ibis.where(table["int_col"] == 0, 42, -1).cast("int64").name("where_col")),
        ]
    )

    result = table.execute()

    expected = df.loc[:, ["int_col"]].copy()

    expected['where_col'] = -1
    expected.loc[expected['int_col'] == 0, 'where_col'] = 42

    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["datafusion"])
def test_where_column(backend, alltypes, df):
    expr = ibis.where(alltypes["int_col"] == 0, 42, -1).cast("int64").name("where_col")
    result = expr.execute()

    expected = pd.Series(
        np.where(df.int_col == 0, 42, -1),
        name="where_col",
        dtype="int64",
    )

    backend.assert_series_equal(result, expected)


def test_select_filter(backend, alltypes, df):
    t = alltypes

    expr = t.select("int_col").filter(t.string_col == "4")
    result = expr.execute()

    expected = df.loc[df.string_col == "4", ["int_col"]].reset_index(drop=True)
    backend.assert_frame_equal(result, expected)


def test_select_filter_select(backend, alltypes, df):
    t = alltypes
    expr = t.select("int_col").filter(t.string_col == "4").int_col
    result = expr.execute().rename("int_col")

    expected = df.loc[df.string_col == "4", "int_col"].reset_index(drop=True)
    backend.assert_series_equal(result, expected)


def test_interactive(alltypes):
    expr = alltypes.mutate(
        str_col=_.string_col.replace("1", "").nullif("2"),
        date_col=_.timestamp_col.date(),
        delta_col=lambda t: ibis.now() - t.timestamp_col,
    )

    orig = ibis.options.interactive
    ibis.options.interactive = True
    try:
        repr(expr)
    finally:
        ibis.options.interactive = orig


def test_correlated_subquery(alltypes):
    expr = alltypes[_.double_col > _.view().double_col]
    assert expr.compile() is not None


@pytest.mark.notimpl(["dask", "polars", "pyspark", "datafusion"])
def test_uncorrelated_subquery(backend, batting, batting_df):
    subset_batting = batting[batting.yearID <= 2000]
    expr = batting[_.yearID == subset_batting.yearID.max()]['playerID', 'yearID']
    result = expr.execute()

    expected = batting_df[batting_df.yearID == 2000][['playerID', 'yearID']]
    backend.assert_frame_equal(result, expected)


def test_int_column(alltypes):
    expr = alltypes.mutate(x=1).x
    result = expr.execute()
    assert expr.type() == dt.int8
    assert result.dtype == np.int8


@pytest.mark.notimpl(["datafusion"])
@pytest.mark.never(
    ["bigquery", "sqlite", "snowflake"], reason="backend only implements int64"
)
def test_int_scalar(alltypes):
    expr = alltypes.smallint_col.min()
    result = expr.execute()
    assert expr.type() == dt.int16
    assert result.dtype == np.int16


@pytest.mark.notimpl(["dask", "datafusion", "pandas", "polars"])
@pytest.mark.notyet(
    ["clickhouse"], reason="https://github.com/ClickHouse/ClickHouse/issues/6697"
)
@pytest.mark.notyet(["pyspark"])
@pytest.mark.parametrize("method_name", ["any", "notany"])
def test_exists(batting, awards_players, method_name):
    method = methodcaller(method_name)
    expr = batting[method(batting.yearID == awards_players.yearID)]
    result = expr.execute()
    assert not result.empty


@pytest.mark.parametrize(
    ("ibis_op", "expected_values"),
    [
        param(
            ibis.null(),
            {
                'bigquery': "NULL",
                'clickhouse': 'Nullable(Nothing)',
                'duckdb': "NULL",
                'impala': 'BOOLEAN',
                'snowflake': None,
                'sqlite': "null",
                'trino': 'unknown',
                "postgres": "null",
            },
            id="null",
        ),
        param(
            ibis.literal(False, type=dt.boolean),
            {
                'bigquery': "BOOL",
                'clickhouse': 'UInt8',
                'impala': 'BOOLEAN',
                'snowflake': "BOOLEAN",
                'sqlite': "integer",
                'trino': 'boolean',
                "duckdb": "BOOLEAN",
                "postgres": "boolean",
            },
            id="boolean",
        ),
        param(
            ibis.literal(1, type=dt.int8),
            {
                'bigquery': "INT64",
                'clickhouse': 'UInt8',
                'impala': 'TINYINT',
                'snowflake': "INTEGER",
                'sqlite': "integer",
                'trino': 'integer',
                "duckdb": "SMALLINT",
                "postgres": "integer",
            },
            id="int8",
        ),
        param(
            ibis.literal(1, type=dt.int16),
            {
                'bigquery': "INT64",
                'clickhouse': 'UInt8',
                'impala': 'TINYINT',
                'snowflake': "INTEGER",
                'sqlite': "integer",
                'trino': 'integer',
                "duckdb": "SMALLINT",
                "postgres": "integer",
            },
            id="int16",
        ),
        param(
            ibis.literal(1, type=dt.int32),
            {
                'bigquery': "INT64",
                'clickhouse': 'UInt8',
                'impala': 'TINYINT',
                'snowflake': "INTEGER",
                'sqlite': "integer",
                'trino': 'integer',
                "duckdb": "INTEGER",
                "postgres": "integer",
            },
            id="int32",
        ),
        param(
            ibis.literal(1, type=dt.int64),
            {
                'bigquery': "INT64",
                'clickhouse': 'UInt8',
                'impala': 'TINYINT',
                'snowflake': "INTEGER",
                'sqlite': "integer",
                'trino': 'integer',
                "duckdb": "BIGINT",
                "postgres": "integer",
            },
            id="int64",
        ),
        param(
            ibis.literal(1, type=dt.uint8),
            {
                'bigquery': "INT64",
                'clickhouse': 'UInt8',
                'impala': 'TINYINT',
                'snowflake': "INTEGER",
                'sqlite': "integer",
                'trino': 'integer',
                "duckdb": "UTINYINT",
                "postgres": "integer",
            },
            id="uint8",
        ),
        param(
            ibis.literal(1, type=dt.uint16),
            {
                'bigquery': "INT64",
                'clickhouse': 'UInt8',
                'impala': 'TINYINT',
                'snowflake': "INTEGER",
                'sqlite': "integer",
                'trino': 'integer',
                "duckdb": "USMALLINT",
                "postgres": "integer",
            },
            id="uint16",
        ),
        param(
            ibis.literal(1, type=dt.uint32),
            {
                'bigquery': "INT64",
                'clickhouse': 'UInt8',
                'impala': 'TINYINT',
                'snowflake': "INTEGER",
                'sqlite': "integer",
                'trino': 'integer',
                "duckdb": "UINTEGER",
                "postgres": "integer",
            },
            id="uint32",
        ),
        param(
            ibis.literal(1, type=dt.uint64),
            {
                'bigquery': "INT64",
                'clickhouse': 'UInt8',
                'impala': 'TINYINT',
                'snowflake': "INTEGER",
                'sqlite': "integer",
                'trino': 'integer',
                "duckdb": "UBIGINT",
                "postgres": "integer",
            },
            id="uint64",
        ),
        param(
            ibis.literal(1, type=dt.float16),
            {
                'bigquery': "FLOAT64",
                'clickhouse': 'Float64',
                'impala': 'DECIMAL(2,1)',
                'snowflake': "INTEGER",
                'sqlite': "real",
                'trino': 'double',
                "duckdb": "FLOAT",
                "postgres": "numeric",
            },
            marks=[
                pytest.mark.broken(
                    ['polars'],
                    "<class 'ibis.expr.datatypes.core.Float16'>",
                    raises=KeyError,
                ),
            ],
            id="float16",
        ),
        param(
            ibis.literal(1, type=dt.float32),
            {
                'bigquery': "FLOAT64",
                'clickhouse': 'Float64',
                'impala': 'DECIMAL(2,1)',
                'snowflake': "INTEGER",
                'sqlite': "real",
                'trino': 'double',
                "duckdb": "FLOAT",
                "postgres": "numeric",
            },
            id="float32",
        ),
        param(
            ibis.literal(1, type=dt.float64),
            {
                'bigquery': "FLOAT64",
                'clickhouse': 'Float64',
                'impala': 'DECIMAL(2,1)',
                'snowflake': "INTEGER",
                'sqlite': "real",
                'trino': 'double',
                "duckdb": "DOUBLE",
                "postgres": "numeric",
            },
            id="float64",
        ),
        param(
            ibis.literal("STRING", type=dt.string),
            {
                'bigquery': "STRING",
                'clickhouse': 'String',
                'snowflake': "VARCHAR",
                'sqlite': 'text',
                'trino': 'varchar(6)',
                "duckdb": "VARCHAR",
                "impala": "STRING",
                "postgres": "text",
            },
            id="string",
        ),
        param(
            ibis.literal("STRI'NG", type=dt.string),
            {
                'bigquery': "STRING",
                'clickhouse': 'String',
                'snowflake': "VARCHAR",
                'sqlite': 'text',
                'trino': 'varchar(7)',
                "duckdb": "VARCHAR",
                "impala": "STRING",
                "postgres": "text",
            },
            id="string-quote1",
        ),
        param(
            ibis.literal("STRI\"NG", type=dt.string),
            {
                'bigquery': "STRING",
                'clickhouse': 'String',
                'snowflake': "VARCHAR",
                'sqlite': 'text',
                'trino': 'varchar(7)',
                "duckdb": "VARCHAR",
                "impala": "STRING",
                "postgres": "text",
            },
            id="string-quote2",
        ),
        param(
            ibis.literal(b"A", type=dt.binary),
            {
                'bigquery': "BYTES",
                'clickhouse': 'String',
                'snowflake': "BINARY",
                'sqlite': "blob",
                'trino': 'STRING',
                "postgres": "bytea",
            },
            marks=[
                pytest.mark.broken(
                    ['duckdb'],
                    "Conversion Error: Unimplemented type for cast (TINYINT[] -> BLOB)",
                    raises=DuckDBConversionException,
                ),
                pytest.mark.broken(
                    ['polars'],
                    "ValueError: could not convert value \"b'A'\" as a Literal",
                ),
                pytest.mark.broken(
                    ['trino'],
                    "(builtins.AttributeError) 'bytes' object has no attribute 'encode'",
                    raises=sqlalchemy.exc.StatementError,
                ),
                pytest.mark.broken(
                    ['clickhouse'],
                    "<class 'ibis.expr.operations.generic.Literal'>",
                    raises=NotImplementedError,
                ),
                pytest.mark.broken(
                    ['impala'],
                    "Unsupported type: Binary(nullable=True)",
                    raises=NotImplementedError,
                ),
            ],
            id="binary",
        ),
        param(
            ibis.date(12, 12, 12),
            {
                'bigquery': "DATE",
                'snowflake': 'DATE',
                'sqlite': "text",
                'trino': 'date',
                "duckdb": "DATE",
                "postgres": "date",
            },
            marks=[
                pytest.mark.broken(
                    ['impala'],
                    "No translation rule for <class 'ibis.expr.operations.temporal.DateFromYMD'>",
                    raises=com.OperationNotDefinedError,
                ),
                pytest.mark.broken(
                    ['clickhouse'],
                    "<class 'ibis.expr.operations.temporal.DateFromYMD'>",
                    raises=NotImplementedError,
                ),
            ],
            id="date",
        ),
        param(
            ibis.time(12, 12, 12),
            {
                'bigquery': "TIME",
                'snowflake': "TIME",
                'sqlite': "text",
                'trino': 'time(3)',
                "duckdb": "TIME",
                "postgres": "time without time zone",
            },
            marks=[
                pytest.mark.broken(
                    ['clickhouse'],
                    "<class 'ibis.expr.operations.temporal.TimeFromHMS'>",
                    raises=NotImplementedError,
                ),
                pytest.mark.broken(
                    ['impala'],
                    "No translation rule for <class 'ibis.expr.operations.temporal.TimeFromHMS'>",
                    raises=com.OperationNotDefinedError,
                ),
            ],
            id="time",
        ),
        param(
            ibis.now().cast(dt.timestamp),
            {
                'bigquery': "TIMESTAMP",
                'clickhouse': 'DateTime',
                'impala': 'TIMESTAMP',
                'snowflake': 'TIMESTAMP_LTZ',
                'sqlite': "text",
                'trino': 'timestamp(3) with time zone',
                "duckdb": "TIMESTAMP",
                "postgres": "timestamp with time zone",
            },
            id="timestamp",
        ),
        param(
            ibis.interval(1, unit="s"),
            {
                'bigquery': "INTERVAL",
                'clickhouse': 'IntervalSecond',
                'sqlite': "integer",
                'trino': 'integer',
                "duckdb": "INTERVAL",
                "postgres": "interval",
            },
            marks=[
                pytest.mark.broken(
                    ['snowflake'],
                    '(snowflake.connector.errors.ProgrammingError) 001007 (22023): SQL compilation error:'
                    "invalid type [CAST(INTERVAL_LITERAL('second', '1') AS VARIANT)] for parameter 'TO_VARIANT'",
                    raises=ProgrammingError,
                ),
                pytest.mark.broken(
                    ['impala'],
                    'AnalysisException: Syntax error in line 1: SELECT typeof(INTERVAL 1 SECOND) AS `TypeOf(1)` '
                    'Encountered: ) Expected: +',
                ),
            ],
            id="interval",
        ),
        param(
            ibis.literal("08f48812-7948-4718-96c7-27fa6a398db6", type=dt.uuid),
            {
                'sqlite': "text",
                'trino': 'uuid',
                "postgres": "uuid",
            },
            marks=[
                pytest.mark.broken(
                    ['bigquery'],
                    'Cannot create literal for UUID(nullable=True) type.',
                    raises=NotImplementedError,
                ),
                pytest.mark.broken(
                    ['duckdb', 'polars'],
                    "<class 'ibis.expr.datatypes.core.UUID'>",
                    raises=KeyError,
                ),
                pytest.mark.broken(
                    ['snowflake'],
                    '(snowflake.connector.errors.ProgrammingError) 252004: Failed processing pyformat-parameters: 255001: Binding data in type (uuid) is not supported.',
                    raises=ProgrammingError,
                ),
                pytest.mark.broken(
                    ['sqlite'],
                    "(sqlite3.InterfaceError) Error binding parameter 0 - probably unsupported type.",
                    raises=sqlalchemy.exc.InterfaceError,
                ),
                pytest.mark.broken(
                    ['clickhouse'],
                    "<class 'ibis.expr.operations.generic.Literal'>",
                    raises=NotImplementedError,
                ),
                pytest.mark.broken(
                    ['impala'],
                    "Unsupported type: UUID(nullable=True)",
                    raises=NotImplementedError,
                ),
            ],
            id="uuid",
        ),
        param(
            ibis.literal("00:00:0A:BB:28:FC", type=dt.macaddr),
            {
                'bigquery': "STRING",
                'clickhouse': "String",
                'snowflake': "VARCHAR",
                'sqlite': "text",
                'trino': 'varchar(17)',
                "impala": 'STRING',
                "postgres": "text",
            },
            marks=[
                pytest.mark.broken(
                    ['duckdb', 'polars'],
                    "<class 'ibis.expr.datatypes.core.MACADDR'>",
                    raises=KeyError,
                ),
            ],
            id="macaddr",
        ),
        param(
            ibis.literal("127.0.0.1", type=dt.inet),
            {
                'bigquery': "STRING",
                'clickhouse': "IPv4",
                'impala': 'STRING',
                'snowflake': "VARCHAR",
                'sqlite': "text",
                'trino': 'varchar(9)',
                "postgres": "text",
            },
            marks=[
                pytest.mark.broken(
                    ['duckdb', 'polars'],
                    "<class 'ibis.expr.datatypes.core.INET'>",
                    raises=KeyError,
                ),
            ],
            id="inet",
        ),
        param(
            ibis.literal(decimal.Decimal("1.2"), type=dt.decimal),
            {
                'bigquery': "INT64",
                'snowflake': "VARCHAR",
                'sqlite': "real",
                'trino': 'decimal(2,1)',
                "duckdb": "DECIMAL(18,3)",
                "postgres": "numeric",
            },
            marks=[
                pytest.mark.broken(
                    ['bigquery'],
                    "ufunc 'isfinite' not supported for the input types, and the inputs could not be "
                    "safely coerced to any supported types according to the casting rule ''safe''",
                    raises=TypeError,
                ),
                pytest.mark.broken(
                    ['clickhouse'],
                    "Code: 46. DB::Exception: Unknown function Decimal: "
                    "While processing toTypeName(Decimal('1.2')).",
                    raises=ClickhouseDriverOperationalError,
                ),
                pytest.mark.broken(
                    ['polars'],
                    "could not convert value '1.2' as a Literal",
                    raises=ValueError,
                ),
                pytest.mark.broken(
                    ['impala'],
                    "impala.error.HiveServer2Error: AnalysisException: Syntax error in line 1:"
                    "SELECT typeof(Decimal('1.2')) AS `TypeOf(Decimal('1.2'))"
                    "Encountered: DECIMAL"
                    "Expected: ALL, CASE, CAST, DEFAULT, DISTINCT, EXISTS, FALSE, IF, "
                    "INTERVAL, LEFT, NOT, NULL, REPLACE, RIGHT, TRUNCATE, TRUE, IDENTIFIER"
                    "CAUSED BY: Exception: Syntax error",
                ),
            ],
            id="decimal",
        ),
        param(
            ibis.array([1.0, 2.0, 3.0]),
            {
                'clickhouse': "Array(Float64)",
                'snowflake': "ARRAY",
                'trino': 'array(double)',
                "bigquery": "ARRAY",
                "duckdb": "DOUBLE[]",
                "postgres": "numeric[]",
            },
            marks=[
                pytest.mark.broken(
                    ['sqlite'],
                    "(sqlite3.InterfaceError) Error binding parameter 0 - probably unsupported type.",
                    raises=sqlalchemy.exc.InterfaceError,
                ),
                pytest.mark.broken(
                    ['impala'],
                    "Unsupported type: Array(value_type=Float64(nullable=True), nullable=True)",
                    raises=NotImplementedError,
                ),
            ],
            id="array<float>",
        ),
        param(
            ibis.array(["A", "B", "C"]),
            {
                'clickhouse': "Array(String)",
                'snowflake': "ARRAY",
                'trino': 'array(varchar(1))',
                "bigquery": "ARRAY",
                "duckdb": "VARCHAR[]",
                "postgres": "text[]",
            },
            marks=[
                pytest.mark.broken(
                    ['sqlite'],
                    "(sqlite3.InterfaceError) Error binding parameter 0 - probably unsupported type.",
                    raises=sqlalchemy.exc.InterfaceError,
                ),
                pytest.mark.broken(
                    ['impala'],
                    "Unsupported type: Array(value_type=String(nullable=True), nullable=True)",
                    raises=NotImplementedError,
                ),
            ],
            id="array<string>",
        ),
        # TODO: We should add test cases for other types including:
        #   dt.geometry, dt.geography, dt.point, dt.linestring, dt.polygon, dt.multilinestring,
        #   dt.multipoint, dt.multipolygon, dt.json, dt.struct
    ],
)
@pytest.mark.notimpl(
    [
        "dask",
        "datafusion",
        "mssql",
        "mysql",
        "pandas",
        "polars",
        "pyspark",
    ],
    raises=(NotImplementedError, com.OperationNotDefinedError),
)
def test_typeof(backend, con, ibis_op, expected_values):
    expr = ibis_op.typeof()
    result = con.execute(expr)

    backend_name = backend.name()
    assert result == expected_values[backend_name]
