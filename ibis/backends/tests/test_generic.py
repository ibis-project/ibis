from __future__ import annotations

import contextlib
import datetime
import decimal
from collections import Counter
from itertools import permutations
from operator import invert, methodcaller

import pytest
import toolz
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.selectors as s
from ibis import _
from ibis.backends.tests.errors import (
    ClickHouseDatabaseError,
    ClickHouseInternalError,
    ExaQueryError,
    GoogleBadRequest,
    ImpalaHiveServer2Error,
    MySQLProgrammingError,
    OracleDatabaseError,
    PolarsInvalidOperationError,
    PsycoPg2InternalError,
    PsycoPgSyntaxError,
    Py4JJavaError,
    PyAthenaDatabaseError,
    PyAthenaOperationalError,
    PyDruidProgrammingError,
    PyODBCDataError,
    PyODBCProgrammingError,
    SnowflakeProgrammingError,
    TrinoUserError,
)
from ibis.common.annotations import ValidationError

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
tm = pytest.importorskip("pandas.testing")
pa = pytest.importorskip("pyarrow")

NULL_BACKEND_TYPES = {
    "bigquery": "NULL",
    "clickhouse": "Nullable(Nothing)",
    "datafusion": "NULL",
    "duckdb": "NULL",
    "impala": "BOOLEAN",
    "snowflake": None,
    "sqlite": "null",
    "trino": "unknown",
    "postgres": "null",
    "risingwave": "null",
    "databricks": "void",
}


@pytest.mark.notyet(
    ["flink", "athena"], "The runtime does not support untyped `NULL` values."
)
def test_null_literal(con, backend):
    expr = ibis.null()
    assert pd.isna(con.execute(expr))

    with contextlib.suppress(com.OperationNotDefinedError):
        backend_name = backend.name()
        assert con.execute(expr.typeof()) == NULL_BACKEND_TYPES[backend_name]

    assert expr.type() == dt.null
    assert pd.isna(con.execute(expr.cast(str).upper()))


@pytest.mark.notimpl(
    "mssql",
    reason="https://github.com/ibis-project/ibis/issues/9109",
    raises=AssertionError,
)
def test_null_literal_typed(con):
    expr = ibis.null(bool)
    assert expr.type() == dt.boolean
    assert pd.isna(con.execute(expr))
    assert pd.isna(con.execute(~expr))
    assert pd.isna(con.execute(expr.cast(str).upper()))


BOOLEAN_BACKEND_TYPE = {
    "bigquery": "BOOL",
    "clickhouse": "Bool",
    "impala": "BOOLEAN",
    "snowflake": "BOOLEAN",
    "sqlite": "integer",
    "trino": "boolean",
    "duckdb": "BOOLEAN",
    "postgres": "boolean",
    "risingwave": "boolean",
    "flink": "BOOLEAN NOT NULL",
    "databricks": "boolean",
    "athena": "boolean",
}


def test_null_literal_typed_typeof(con, backend):
    expr = ibis.null(bool)
    TYPES = {
        **BOOLEAN_BACKEND_TYPE,
        "clickhouse": "Nullable(Bool)",
        "flink": "BOOLEAN",
        "sqlite": "null",  # in sqlite, typeof(x) is determined by the VALUE of x at runtime, not it's static type
        "snowflake": None,
        "bigquery": "NULL",
    }

    with contextlib.suppress(com.OperationNotDefinedError):
        assert con.execute(expr.typeof()) == TYPES[backend.name()]


def test_boolean_literal(con, backend):
    expr = ibis.literal(False, type=dt.boolean)
    result = con.execute(expr)
    assert not result
    assert type(result) in (np.bool_, bool)

    with contextlib.suppress(com.OperationNotDefinedError):
        assert con.execute(expr.typeof()) == BOOLEAN_BACKEND_TYPE[backend.name()]


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(ibis.null().fill_null(5), 5, id="na_fill_null"),
        param(ibis.literal(5).fill_null(10), 5, id="non_na_fill_null"),
        param(ibis.literal(5).nullif(5), None, id="nullif_null"),
        param(ibis.literal(10).nullif(5), 10, id="nullif_not_null"),
    ],
)
def test_scalar_fill_null_nullif(con, expr, expected):
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
    ("col", "value", "filt"),
    [
        param(
            "nan_col",
            ibis.literal(np.nan),
            methodcaller("isnan"),
            marks=[
                pytest.mark.notimpl(["mysql", "mssql", "sqlite", "druid"]),
                pytest.mark.notyet(
                    ["exasol"],
                    raises=ExaQueryError,
                    reason="no way to test for nan-ness",
                ),
                pytest.mark.notyet(
                    ["flink"],
                    "NaN is not supported in Flink SQL",
                    raises=NotImplementedError,
                ),
            ],
            id="nan_col",
        ),
        param(
            "none_col",
            ibis.null().cast("float64"),
            methodcaller("isnull"),
            id="none_col",
        ),
    ],
)
def test_isna(backend, alltypes, col, value, filt):
    table = alltypes.select(**{col: value})
    df = table.execute()

    result = table.filter(filt(table[col])).execute().reset_index(drop=True)
    expected = df[df[col].isna()].reset_index(drop=True)

    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "value",
    [
        None,
        param(
            np.nan,
            marks=[
                pytest.mark.notimpl(
                    [
                        "bigquery",
                        "clickhouse",
                        "datafusion",
                        "duckdb",
                        "impala",
                        "postgres",
                        "risingwave",
                        "mysql",
                        "snowflake",
                        "polars",
                        "trino",
                        "mssql",
                        "druid",
                        "oracle",
                        "exasol",
                        "pyspark",
                        "databricks",
                        "athena",
                    ],
                    reason="NaN != NULL for these backends",
                ),
                pytest.mark.notyet(
                    ["flink"],
                    "NaN is not supported in Flink SQL",
                    raises=NotImplementedError,
                ),
            ],
            id="nan_col",
        ),
    ],
)
def test_column_fill_null(backend, alltypes, value):
    table = alltypes.mutate(missing=ibis.literal(value).cast("float64"))
    pd_table = table.execute()

    res = table.mutate(missing=table.missing.fill_null(0.0)).execute()
    sol = pd_table.assign(missing=pd_table.missing.fillna(0.0))
    backend.assert_frame_equal(res.reset_index(drop=True), sol.reset_index(drop=True))


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(ibis.coalesce(5, None, 4), 5, id="generic"),
        param(ibis.coalesce(ibis.null(), 4, ibis.null()), 4, id="null_start_end"),
        param(ibis.coalesce(ibis.null(), ibis.null(), 3.14), 3.14, id="non_null_last"),
    ],
)
def test_coalesce(con, expr, expected):
    result = con.execute(expr.name("tmp"))

    if isinstance(result, decimal.Decimal):
        # in case of Impala the result is decimal
        # >>> decimal.Decimal('5.56') == 5.56
        # False
        assert result == decimal.Decimal(str(expected))
    else:
        assert result == pytest.approx(expected)


@pytest.mark.notimpl(["clickhouse", "druid", "exasol"])
def test_identical_to(backend, alltypes, sorted_df):
    sorted_alltypes = alltypes.order_by("id")
    df = sorted_df
    dt = df[["tinyint_col", "double_col"]]

    ident = sorted_alltypes.tinyint_col.identical_to(sorted_alltypes.double_col)
    expr = sorted_alltypes.select("id", ident.name("tmp")).order_by("id")
    result = expr.execute().tmp

    expected = (dt.tinyint_col.isnull() & dt.double_col.isnull()) | (
        dt.tinyint_col == dt.double_col
    )

    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("column", "elements"),
    [
        ("int_col", [1, 2, 3]),
        ("int_col", (1, 2, 3)),
        ("string_col", ["1", "2", "3"]),
        ("string_col", ("1", "2", "3")),
        ("int_col", {1}),
        ("int_col", frozenset({1})),
    ],
)
@pytest.mark.notimpl(["druid"])
def test_isin(backend, alltypes, sorted_df, column, elements):
    sorted_alltypes = alltypes.order_by("id")
    expr = sorted_alltypes.select(
        "id", sorted_alltypes[column].isin(elements).name("tmp")
    ).order_by("id")
    result = expr.execute().tmp

    expected = sorted_df[column].isin(elements)
    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("column", "elements"),
    [
        ("int_col", [1, 2, 3]),
        ("int_col", (1, 2, 3)),
        ("string_col", ["1", "2", "3"]),
        ("string_col", ("1", "2", "3")),
        ("int_col", {1}),
        ("int_col", frozenset({1})),
    ],
)
@pytest.mark.notimpl(["druid"])
def test_notin(backend, alltypes, sorted_df, column, elements):
    sorted_alltypes = alltypes.order_by("id")
    expr = sorted_alltypes.select(
        "id", sorted_alltypes[column].notin(elements).name("tmp")
    ).order_by("id")
    result = expr.execute().tmp

    expected = ~sorted_df[column].isin(elements)
    expected = backend.default_series_rename(expected)
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("predicate_fn", "expected_fn"),
    [
        param(lambda t: t["bool_col"], lambda df: df["bool_col"], id="no_op"),
        param(lambda t: ~t["bool_col"], lambda df: ~df["bool_col"], id="negate"),
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
        ),
    ],
)
@pytest.mark.notimpl(["druid"])
def test_filter(backend, alltypes, sorted_df, predicate_fn, expected_fn):
    sorted_alltypes = alltypes.order_by("id")
    table = sorted_alltypes.filter(predicate_fn(sorted_alltypes)).order_by("id")
    result = table.execute()
    expected = sorted_df[expected_fn(sorted_df)]

    backend.assert_frame_equal(result, expected)


@pytest.mark.notyet(
    ["exasol"],
    raises=ExaQueryError,
    reason="sqlglot `eliminate_qualify` transform produces underscores in aliases, which is not allowed by exasol",
)
@pytest.mark.notimpl(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason="requires enabling window functions",
)
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
@pytest.mark.notyet(
    ["oracle"],
    raises=OracleDatabaseError,
    reason="sqlglot `eliminate_qualify` transform produces underscores in aliases, which is not allowed by oracle",
)
@pytest.mark.notyet(
    ["flink"],
    reason="Flink engine does not support generic window clause with no order by",
)
# TODO(kszucs): this is not supported at the expression level
def test_filter_with_window_op(backend, alltypes, sorted_df):
    table = alltypes
    window = ibis.window(group_by=table.id)
    table = table.filter(lambda t: t["id"].mean().over(window) > 3).order_by("id")
    result = table.execute()
    expected = (
        sorted_df.groupby(["id"])
        .filter(lambda t: t["id"].mean() > 3)
        .reset_index(drop=True)
    )
    backend.assert_frame_equal(result, expected)


def test_case_where(backend, alltypes, df):
    table = alltypes
    table = table.mutate(
        new_col=(
            ibis.cases(
                (table["int_col"] == 1, 20),
                (table["int_col"] == 0, 10),
                else_=0,
            ).cast("int64")
        )
    )

    result = table.execute()

    expected = df.copy()
    mask_0 = expected["int_col"] == 1
    mask_1 = expected["int_col"] == 0

    expected["new_col"] = 0
    expected.loc[mask_0, "new_col"] = 20
    expected.loc[mask_1, "new_col"] = 10

    backend.assert_frame_equal(result, expected)


# TODO: some of these are notimpl (datafusion) others are probably never
@pytest.mark.notimpl(["mysql", "sqlite", "mssql", "druid", "exasol"])
@pytest.mark.notyet(
    ["flink"], "NaN is not supported in Flink SQL", raises=NotImplementedError
)
def test_select_filter_mutate(backend, alltypes, df):
    """Test that select, filter and mutate are executed in right order.

    Before PR #2635, try_fusion in analysis.py would fuse these
    operations together in a way that the order of the operations were
    wrong. (mutate was executed before filter).
    """
    t = alltypes

    # Prepare the float_col so that filter must execute
    # before the cast to get the correct result.
    t = t.mutate(float_col=ibis.cases((t["bool_col"], t["float_col"]), else_=np.nan))

    # Actual test
    t = t.select(t.columns)
    t = t.filter(~t["float_col"].isnan())
    t = t.mutate(float_col=t["float_col"].cast("float64"))
    result = t.execute()

    expected = df.copy()
    expected.loc[~df["bool_col"], "float_col"] = None
    expected = expected[~expected["float_col"].isna()].reset_index(drop=True)
    expected = expected.assign(float_col=expected["float_col"].astype("float64"))

    backend.assert_series_equal(result.float_col, expected.float_col)


def test_table_fill_null_invalid(alltypes):
    with pytest.raises(
        com.IbisTypeError, match=r"Column 'invalid_col' is not found in table"
    ):
        alltypes.fill_null({"invalid_col": 0.0})

    with pytest.raises(
        com.IbisTypeError, match="Cannot fill_null on column 'string_col' of type.*"
    ):
        alltypes[["int_col", "string_col"]].fill_null(0)

    with pytest.raises(
        com.IbisTypeError, match="Cannot fill_null on column 'int_col' of type.*"
    ):
        alltypes.fill_null({"int_col": "oops"})


@pytest.mark.parametrize(
    "replacements",
    [
        param({"int_col": 20}, id="int"),
        param({"double_col": -1, "string_col": "missing"}, id="double-int-str"),
        param({"double_col": -1.5, "string_col": "missing"}, id="double-str"),
        param({}, id="empty"),
    ],
)
def test_table_fill_null_mapping(backend, alltypes, replacements):
    table = alltypes.mutate(
        int_col=alltypes.int_col.nullif(1),
        double_col=alltypes.double_col.nullif(3.0),
        string_col=alltypes.string_col.nullif("2"),
    ).select("id", "int_col", "double_col", "string_col")
    pd_table = table.execute()

    result = table.fill_null(replacements).execute().reset_index(drop=True)
    expected = pd_table.fillna(replacements).reset_index(drop=True)

    backend.assert_frame_equal(result, expected, check_dtype=False)


def test_table_fill_null_scalar(backend, alltypes):
    table = alltypes.mutate(
        int_col=alltypes.int_col.nullif(1),
        double_col=alltypes.double_col.nullif(3.0),
        string_col=alltypes.string_col.nullif("2"),
    ).select("id", "int_col", "double_col", "string_col")
    pd_table = table.execute()

    res = table[["int_col", "double_col"]].fill_null(0).execute().reset_index(drop=True)
    sol = pd_table[["int_col", "double_col"]].fillna(0).reset_index(drop=True)
    backend.assert_frame_equal(res, sol, check_dtype=False)

    res = table[["string_col"]].fill_null("missing").execute().reset_index(drop=True)
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


def test_drop_null_invalid(alltypes):
    with pytest.raises(
        com.IbisTypeError, match=r"Column 'invalid_col' is not found in table"
    ):
        alltypes.drop_null(["invalid_col"])

    with pytest.raises(ValidationError):
        alltypes.drop_null(how="invalid")


@pytest.mark.parametrize("how", ["any", "all"])
@pytest.mark.parametrize(
    "subset",
    [
        param(None, id="none"),
        param(
            [],
            marks=pytest.mark.notimpl(["exasol"], raises=ExaQueryError, strict=False),
            id="empty",
        ),
        param("col_1", id="single"),
        param(["col_1", "col_2"], id="one-and-two"),
        param(["col_1", "col_3"], id="one-and-three"),
    ],
)
@pytest.mark.notimpl(["druid"], strict=False)
def test_drop_null_table(backend, alltypes, how, subset):
    is_two = alltypes.int_col == 2
    is_four = alltypes.int_col == 4

    table = alltypes.mutate(
        col_1=is_two.ifelse(ibis.null(), alltypes.float_col),
        col_2=is_four.ifelse(ibis.null(), alltypes.float_col),
        col_3=(is_two | is_four).ifelse(ibis.null(), alltypes.float_col),
    ).select("id", "col_1", "col_2", "col_3")

    table_pandas = table.execute()
    result = (
        table.drop_null(subset, how=how).order_by("id").execute().reset_index(drop=True)
    )
    expected = (
        table_pandas.dropna(how=how, subset=subset)
        .sort_values(["id"])
        .reset_index(drop=True)
    )

    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "key, df_kwargs",
    [
        param("id", {"by": "id"}),
        param(_.id, {"by": "id"}),
        param(lambda _: _.id, {"by": "id"}),
        param(ibis.desc("id"), {"by": "id", "ascending": False}),
        param(["id", "int_col"], {"by": ["id", "int_col"]}),
        param(
            ["id", ibis.desc("int_col")],
            {"by": ["id", "int_col"], "ascending": [True, False]},
        ),
    ],
)
@pytest.mark.notimpl(["druid"])
def test_order_by(backend, alltypes, df, key, df_kwargs):
    result = alltypes.filter(_.id < 100).order_by(key).execute()
    expected = df.loc[df.id < 100].sort_values(**df_kwargs)
    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(
    ["polars", "druid", "risingwave"],
    raises=com.OperationNotDefinedError,
    reason="random not supported",
)
def test_order_by_random(alltypes):
    expr = alltypes.filter(_.id < 100).order_by(ibis.random()).limit(5)
    r1 = expr.execute()
    r2 = expr.execute()
    assert len(r1) == 5
    assert len(r2) == 5
    # Ensure that multiple executions returns different results
    assert not r1.equals(r2)


@pytest.mark.notimpl(["druid"])
@pytest.mark.parametrize(
    "op, expected",
    [
        param("desc", {"a": [1, 2, 3], "b": ["foo", "baz", None]}),
        param("asc", {"a": [2, 1, 3], "b": ["baz", "foo", None]}),
    ],
    ids=["desc", "asc"],
)
def test_order_by_nulls_default(con, op, expected):
    # default nulls_first is False
    t = ibis.memtable([{"a": 1, "b": "foo"}, {"a": 2, "b": "baz"}, {"a": 3, "b": None}])
    expr = t.order_by(getattr(t["b"], op)())
    result = con.execute(expr).reset_index(drop=True)
    expected = pd.DataFrame(expected)

    tm.assert_frame_equal(
        result.replace({np.nan: None}), expected.replace({np.nan: None})
    )


@pytest.mark.notimpl(["druid"])
@pytest.mark.parametrize(
    "op, nulls_first, expected",
    [
        param("desc", True, {"a": [3, 1, 2], "b": [None, "foo", "baz"]}),
        param("asc", True, {"a": [3, 2, 1], "b": [None, "baz", "foo"]}),
    ],
    ids=["desc", "asc"],
)
def test_order_by_nulls(con, op, nulls_first, expected):
    t = ibis.memtable([{"a": 1, "b": "foo"}, {"a": 2, "b": "baz"}, {"a": 3, "b": None}])
    expr = t.order_by(getattr(t["b"], op)(nulls_first=nulls_first))
    result = con.execute(expr).reset_index(drop=True)
    expected = pd.DataFrame(expected)

    tm.assert_frame_equal(
        result.replace({np.nan: None}), expected.replace({np.nan: None})
    )


@pytest.mark.notimpl(["druid"])
@pytest.mark.never(
    ["mysql"],
    raises=AssertionError,
    reason="someone decided a long time ago that 'A' = 'a' is true in these systems",
)
@pytest.mark.parametrize(
    "op1, nf1, op2, nf2, expected",
    [
        param(
            "asc",
            False,
            "desc",
            False,
            {
                "col1": [1, 1, 1, 2, 3, 3, None],
                "col2": ["c", "a", None, "B", "a", "D", "a"],
            },
            id="asc-desc-ff",
        ),
        param(
            "asc",
            True,
            "desc",
            True,
            {
                "col1": [None, 1, 1, 1, 2, 3, 3],
                "col2": ["a", None, "c", "a", "B", "a", "D"],
            },
            id="asc-desc-tt",
        ),
        param(
            "asc",
            True,
            "desc",
            False,
            {
                "col1": [None, 1, 1, 1, 2, 3, 3],
                "col2": ["a", "c", "a", None, "B", "a", "D"],
            },
            id="asc-desc-tf",
        ),
        param(
            "asc",
            True,
            "asc",
            True,
            {
                "col1": [None, 1, 1, 1, 2, 3, 3],
                "col2": ["a", None, "a", "c", "B", "D", "a"],
            },
            id="asc-asc-tt",
        ),
        param(
            "asc",
            True,
            "asc",
            False,
            {
                "col1": [None, 1, 1, 1, 2, 3, 3],
                "col2": ["a", "a", "c", None, "B", "D", "a"],
            },
            id="asc-asc-tf",
        ),
    ],
)
def test_order_by_two_cols_nulls(con, op1, nf1, nf2, op2, expected):
    t = ibis.memtable(
        {
            # this is here because pandas converts None to nan, but of course
            # only for numeric types, because that's totally reasonable
            "col1": pd.Series([1, 3, 2, 1, 3, 1, None], dtype="object"),
            "col2": ["a", "a", "B", "c", "D", None, "a"],
        }
    )
    expr = t.order_by(
        getattr(t["col1"], op1)(nulls_first=nf1),
        getattr(t["col2"], op2)(nulls_first=nf2),
    )

    result = con.execute(expr).reset_index(drop=True)
    expected = pd.DataFrame(expected)

    tm.assert_frame_equal(
        result.replace({np.nan: None}), expected.replace({np.nan: None})
    )


@pytest.mark.notyet(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason="Druid only supports trivial unions",
)
def test_table_info(alltypes):
    expr = alltypes.info()
    df = expr.execute()
    assert alltypes.columns == tuple(df.name)
    assert expr.columns == (
        "name",
        "type",
        "nullable",
        "nulls",
        "non_nulls",
        "null_frac",
        "pos",
    )
    assert expr.columns == tuple(df.columns)


@pytest.mark.notyet(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason="Druid only supports trivial unions",
)
@pytest.mark.notyet(
    ["flink"], reason="IOException - Insufficient number of network buffers"
)
def test_table_info_large(con):
    num_cols = 129
    col_names = [f"col_{i}" for i in range(num_cols)]
    t = ibis.memtable({col: [0, 1] for col in col_names})
    result = con.execute(t.info())
    assert list(result.name) == col_names
    assert result.pos.dtype == np.int16


@pytest.mark.notimpl(
    ["datafusion", "bigquery", "impala", "mysql", "mssql", "trino", "flink", "athena"],
    raises=com.OperationNotDefinedError,
    reason="quantile and mode is not supported",
)
@pytest.mark.notimpl(["druid"], raises=com.OperationNotDefinedError)
@pytest.mark.notyet(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason="Druid only supports trivial unions",
)
@pytest.mark.parametrize(
    ("selector", "expected_columns"),
    [
        param(
            s.any_of(
                s.of_type("numeric"),
                s.of_type("string"),
                s.of_type("bool"),
                s.of_type("timestamp"),
            ),
            [
                "name",
                "pos",
                "type",
                "count",
                "nulls",
                "unique",
                "mode",
                "mean",
                "std",
                "min",
                "p25",
                "p50",
                "p75",
                "max",
            ],
            marks=[
                pytest.mark.notimpl(
                    ["sqlite"],
                    raises=com.OperationNotDefinedError,
                    reason="quantile is not supported",
                ),
                pytest.mark.notimpl(
                    ["databricks"],
                    raises=AssertionError,
                    reason="timestamp column is discarded",
                ),
                pytest.mark.notimpl(
                    [
                        "clickhouse",
                        "exasol",
                        "impala",
                        "pyspark",
                        "risingwave",
                    ],
                    raises=com.OperationNotDefinedError,
                    reason="mode is not supported",
                ),
                pytest.mark.notimpl(
                    ["oracle"],
                    raises=(OracleDatabaseError, com.OperationNotDefinedError),
                    reason="Mode is not supported and ORA-02000: missing AS keyword",
                ),
                pytest.mark.notyet(
                    ["polars"],
                    raises=PolarsInvalidOperationError,
                    reason="type Float32 is incompatible with expected type Float64",
                ),
            ],
            id="all_cols",
        ),
        param(
            s.of_type("numeric"),
            [
                "name",
                "pos",
                "type",
                "count",
                "nulls",
                "unique",
                "mean",
                "std",
                "min",
                "p25",
                "p50",
                "p75",
                "max",
            ],
            marks=[
                pytest.mark.notimpl(
                    ["sqlite"],
                    raises=com.OperationNotDefinedError,
                    reason="quantile is not supported",
                ),
                pytest.mark.notimpl(
                    ["oracle"],
                    raises=OracleDatabaseError,
                    reason="Mode is not supported and ORA-02000: missing AS keyword",
                ),
                pytest.mark.notimpl(
                    ["polars"],
                    raises=PolarsInvalidOperationError,
                    reason="type Float32 is incompatible with expected type Float64",
                ),
            ],
            id="numeric_col",
        ),
        param(
            s.of_type("string"),
            ["name", "pos", "type", "count", "nulls", "unique", "mode"],
            marks=[
                pytest.mark.notimpl(
                    ["clickhouse", "exasol", "impala", "pyspark", "risingwave"],
                    raises=com.OperationNotDefinedError,
                    reason="mode is not supported",
                ),
                pytest.mark.notimpl(
                    ["oracle"],
                    raises=OracleDatabaseError,
                    reason="ORA-02000: missing AS keyword",
                ),
            ],
            id="string_col",
        ),
    ],
)
def test_table_describe(alltypes, selector, expected_columns):
    sometypes = alltypes.select(selector)
    expr = sometypes.describe()
    df = expr.execute()
    assert sorted(sometypes.columns) == sorted(df.name)
    assert sorted(expr.columns) == sorted(expected_columns)
    assert sorted(expr.columns) == sorted(df.columns)


@pytest.mark.notimpl(
    [
        "datafusion",
        "bigquery",
        "impala",
        "mysql",
        "mssql",
        "trino",
        "flink",
        "sqlite",
        "athena",
    ],
    raises=com.OperationNotDefinedError,
    reason="quantile is not supported",
)
@pytest.mark.notimpl(["druid"], raises=com.OperationNotDefinedError)
@pytest.mark.notyet(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason="Druid only supports trivial unions",
)
@pytest.mark.notyet(
    ["oracle"], raises=OracleDatabaseError, reason="ORA-02000: missing AS keyword"
)
def test_table_describe_large(con):
    num_cols = 129
    col_names = [f"col_{i}" for i in range(num_cols)]
    t = ibis.memtable({col: [0, 1] for col in col_names})
    result = con.execute(t.describe())
    assert set(result.name) == set(col_names)
    assert result.pos.dtype == np.int16


@pytest.mark.parametrize(
    ("ibis_op", "pandas_op"),
    [
        param(_.string_col.isin([]), lambda df: df.string_col.isin([]), id="isin"),
        param(_.string_col.notin([]), lambda df: ~df.string_col.isin([]), id="notin"),
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
    expr = alltypes.filter(ibis_op)
    expected = df.loc[pandas_op(df)].sort_values(["id"]).reset_index(drop=True)
    result = expr.execute().sort_values(["id"]).reset_index(drop=True)
    backend.assert_frame_equal(result, expected)


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
    expr = alltypes.filter(ibis_op).order_by("id")
    expected = df[pandas_op(df)].sort_values(["id"]).reset_index(drop=True)
    result = expr.execute()
    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected", "op"),
    [
        param(True, True, toolz.identity, id="true_noop"),
        param(False, False, toolz.identity, id="false_noop"),
        param(True, False, invert, id="true_invert"),
        param(False, True, invert, id="false_invert"),
    ],
)
def test_logical_negation_literal(con, expr, expected, op):
    assert con.execute(op(ibis.literal(expr)).name("tmp")) == expected


@pytest.mark.parametrize("op", [toolz.identity, invert])
def test_logical_negation_column(backend, alltypes, df, op):
    result = op(alltypes["bool_col"]).name("tmp").execute()
    expected = op(df["bool_col"])
    backend.assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize(
    ("dtype", "zero", "expected"),
    [("int64", 0, 1), ("float64", 0.0, 1.0)],
)
def test_zero_ifnull_literals(con, dtype, zero, expected):
    assert con.execute(ibis.null().cast(dtype).fill_null(0)) == zero
    assert con.execute(ibis.literal(expected, type=dtype).fill_null(0)) == expected


def test_zero_ifnull_column(backend, alltypes, df):
    expr = alltypes.int_col.nullif(1).fill_null(0).name("tmp")
    result = expr.execute().astype("int32")
    expected = df.int_col.replace(1, 0).rename("tmp").astype("int32")
    backend.assert_series_equal(result, expected)


def test_select_filter(backend, alltypes, df):
    t = alltypes

    # XXX: should we consider a builder pattern for select and filter too?
    #      this would allow us to capture the context
    # TODO(cpcloud): this now requires the additional string_col projection
    expr = t.select("int_col", "string_col").filter(t.string_col == "4")
    result = expr.execute()

    expected = df.loc[df.string_col == "4", ["int_col", "string_col"]].reset_index(
        drop=True
    )
    backend.assert_frame_equal(result, expected)


def test_select_filter_select(backend, alltypes, df):
    t = alltypes
    expr = t.select("int_col", "string_col").filter(t.string_col == "4").int_col
    result = expr.execute().rename("int_col")

    expected = df.loc[df.string_col == "4", "int_col"].reset_index(drop=True)
    backend.assert_series_equal(result, expected)


def test_between(backend, alltypes, df):
    expr = alltypes.double_col.between(5, 10)
    result = expr.execute().rename("double_col")

    expected = df.double_col.between(5, 10)
    backend.assert_series_equal(result, expected)


@pytest.mark.notyet(["flink"], reason="timestamp subtraction doesn't work")
def test_interactive(alltypes, monkeypatch):
    monkeypatch.setattr(ibis.options, "interactive", True)

    expr = alltypes.mutate(
        str_col=_.string_col.replace("1", "").nullif("2"),
        date_col=_.timestamp_col.date(),
        delta_col=lambda t: ibis.now() - t.timestamp_col,
    )

    repr(expr)


@pytest.mark.notimpl(["polars", "pyspark"])
@pytest.mark.notimpl(
    ["risingwave"],
    raises=AssertionError,
    reason='DataFrame.iloc[:, 0] (column name="playerID") are different',
)
def test_uncorrelated_subquery(backend, batting, batting_df):
    subset_batting = batting.filter(batting.yearID <= 2000)
    expr = batting.filter(_.yearID == subset_batting.yearID.max())["playerID", "yearID"]
    result = expr.execute()

    expected = batting_df[batting_df.yearID == 2000][["playerID", "yearID"]]
    backend.assert_frame_equal(result, expected)


def test_int_column(alltypes):
    expr = alltypes.mutate(x=ibis.literal(1)).x
    result = expr.execute()
    assert expr.type() == dt.int8
    assert result.dtype == np.int8


def test_int_scalar(alltypes):
    expr = alltypes.int_col.min()
    assert expr.type().is_integer()
    assert isinstance(expr.execute(), int)


@pytest.mark.notimpl(["polars", "druid"])
@pytest.mark.notyet(
    ["clickhouse"], reason="https://github.com/ClickHouse/ClickHouse/issues/6697"
)
@pytest.mark.parametrize("method_name", ["any", "notany"])
def test_exists(batting, awards_players, method_name):
    years = [1980, 1981]
    batting_years = [1871, *years]
    batting = batting.filter(batting.yearID.isin(batting_years))
    awards_players = awards_players.filter(awards_players.yearID.isin(years))
    method = methodcaller(method_name)
    expr = batting.filter(method(batting.yearID == awards_players.yearID))
    result = expr.execute()
    assert not result.empty


@pytest.mark.notimpl(
    ["datafusion", "mssql", "mysql", "pyspark", "polars", "druid", "oracle", "exasol"],
    raises=com.OperationNotDefinedError,
)
def test_typeof(con):
    # Other tests also use the typeof operation, but only this test has this operation required.
    expr = ibis.literal(1).typeof()
    result = con.execute(expr)

    assert result is not None


@pytest.mark.notyet(["impala"], reason="can't find table in subquery")
@pytest.mark.notimpl(["datafusion", "druid"])
@pytest.mark.xfail_version(pyspark=["pyspark<3.5"])
@pytest.mark.notyet(["exasol"], raises=ExaQueryError, reason="not supported by exasol")
@pytest.mark.notyet(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="https://github.com/risingwavelabs/risingwave/issues/1343",
)
@pytest.mark.notyet(
    ["mssql"],
    raises=PyODBCProgrammingError,
    reason="naked IN queries are not supported",
)
def test_isin_uncorrelated_simple(con):
    u1 = ibis.memtable({"id": [1, 2, 3]})
    a = ibis.memtable({"id": [1, 2]})

    u2 = u1.mutate(in_a=u1["id"].isin(a["id"]))
    final = u2.order_by("id")

    result = con.to_pyarrow(final)
    expected = pa.table({"id": [1, 2, 3], "in_a": [True, True, False]})
    assert result.equals(expected)


@pytest.mark.notyet(["impala"], reason="can't find table in subquery")
@pytest.mark.notimpl(["datafusion", "druid"])
@pytest.mark.xfail_version(pyspark=["pyspark<3.5"])
@pytest.mark.notyet(["exasol"], raises=ExaQueryError, reason="not supported by exasol")
@pytest.mark.notyet(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="https://github.com/risingwavelabs/risingwave/issues/1343",
)
@pytest.mark.notyet(
    ["mssql"],
    raises=PyODBCProgrammingError,
    reason="naked IN queries are not supported",
)
def test_isin_uncorrelated(
    backend, batting, awards_players, batting_df, awards_players_df
):
    expr = batting.select(
        "playerID",
        "yearID",
        has_year_id=batting.yearID.isin(awards_players.yearID),
    ).order_by(["playerID", "yearID"])
    result = expr.execute().has_year_id
    expected = (
        batting_df.sort_values(["playerID", "yearID"])
        .reset_index(drop=True)
        .yearID.isin(awards_players_df.yearID)
        .rename("has_year_id")
    )
    backend.assert_series_equal(result, expected)


def test_isin_uncorrelated_filter(
    backend, batting, awards_players, batting_df, awards_players_df
):
    expr = (
        batting.select("playerID", "yearID")
        .filter(batting.yearID.isin(awards_players.yearID))
        .order_by(["playerID", "yearID"])
    )
    result = expr.execute()
    expected = (
        batting_df.loc[
            batting_df.yearID.isin(awards_players_df.yearID), ["playerID", "yearID"]
        ]
        .sort_values(["playerID", "yearID"])
        .reset_index(drop=True)
    )
    backend.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        param(
            "bool",
            marks=[pytest.mark.notimpl(["mssql"], raises=AssertionError)],
        ),
        param(
            "bytes",
            marks=[
                pytest.mark.notyet(
                    ["exasol"], raises=ExaQueryError, reason="no binary type"
                ),
            ],
        ),
        "str",
        "int",
        "float",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "timestamp",
        "date",
        param(
            "time",
            marks=[
                pytest.mark.notyet(
                    ["exasol"], raises=ExaQueryError, reason="no time type"
                ),
                pytest.mark.notyet(
                    ["athena"], raises=PyAthenaOperationalError, reason="no time type"
                ),
                pytest.mark.notyet(
                    ["clickhouse"],
                    raises=ClickHouseInternalError,
                    reason="time type not supported in clickhouse_connect; "
                    "see https://github.com/ClickHouse/clickhouse-connect/issues/509",
                ),
            ],
        ),
    ],
)
def test_literal_na(con, dtype):
    expr = ibis.literal(None, type=dtype)
    result = con.execute(expr)
    assert pd.isna(result)


def test_memtable_bool_column(con):
    data = [True, False, True]
    t = ibis.memtable({"a": data})
    assert Counter(con.execute(t.a)) == Counter(data)


def test_memtable_construct_from_pyarrow(backend, con, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    monkeypatch.setattr(ibis.options, "default_backend", con)

    pa_t = pa.Table.from_pydict(
        {
            "a": list("abc"),
            "b": [1, 2, 3],
            "c": [1.0, 2.0, 3.0],
            "d": [None, "b", None],
        }
    )
    t = ibis.memtable(pa_t)
    backend.assert_frame_equal(
        t.order_by("a").execute().fillna(pd.NA), pa_t.to_pandas().fillna(pd.NA)
    )


@pytest.mark.notimpl(
    ["flink"], raises=TypeError, reason="doesn't support pyarrow objects yet"
)
def test_memtable_construct_from_pyarrow_c_stream(con):
    pa = pytest.importorskip("pyarrow")

    class Opaque:
        def __init__(self, table):
            self._table = table

        def __arrow_c_stream__(self, *args, **kwargs):
            return self._table.__arrow_c_stream__(*args, **kwargs)

    table = pa.table({"a": list("abc"), "b": [1, 2, 3]})

    t = ibis.memtable(Opaque(table))

    res = con.to_pyarrow(t.order_by("a"))
    assert res.equals(table)


@pytest.mark.parametrize("lazy", [False, True])
def test_memtable_construct_from_polars(backend, con, lazy):
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(
        {
            "a": list("abc"),
            "b": [1, 2, 3],
            "c": [1.0, 2.0, 3.0],
            "d": [None, "b", None],
        }
    )
    obj = df.lazy() if lazy else df
    t = ibis.memtable(obj)
    res = con.to_pandas(t.order_by("a")).fillna(pd.NA)
    sol = df.to_pandas().fillna(pd.NA)
    backend.assert_frame_equal(res, sol)


@pytest.mark.parametrize(
    "df, columns, expected",
    [
        (pd.DataFrame([("a", 1.0)], columns=["d", "f"]), ["a", "b"], ["a", "b"]),
        (pd.DataFrame([("a", 1.0)]), ["A", "B"], ["A", "B"]),
        (pd.DataFrame([("a", 1.0)], columns=["c", "d"]), None, ["c", "d"]),
        ([("a", "1.0")], None, ["col0", "col1"]),
        ([("a", "1.0")], ["d", "e"], ["d", "e"]),
    ],
)
def test_memtable_column_naming(con, monkeypatch, df, columns, expected):
    monkeypatch.setattr(ibis.options, "default_backend", con)

    t = ibis.memtable(df, columns=columns)
    assert all(t.to_pandas().columns == expected)


@pytest.mark.parametrize(
    "df, columns",
    [
        (pd.DataFrame([("a", 1.0)], columns=["d", "f"]), ["a"]),
        (pd.DataFrame([("a", 1.0)]), ["A", "B", "C"]),
        ([("a", "1.0")], ["col0", "col1", "col2"]),
        ([("a", "1.0")], ["d"]),
    ],
)
def test_memtable_column_naming_mismatch(con, monkeypatch, df, columns):
    monkeypatch.setattr(ibis.options, "default_backend", con)

    with pytest.raises(ValueError):
        ibis.memtable(df, columns=columns)


@pytest.mark.notyet(
    ["mssql", "mysql", "exasol", "impala"], reason="various syntax errors reported"
)
@pytest.mark.notyet(
    ["snowflake"],
    reason="unable to handle the varbinary geometry column",
    raises=SnowflakeProgrammingError,
)
@pytest.mark.notyet(
    ["druid"], raises=PyDruidProgrammingError, reason="doesn't support a binary type"
)
def test_memtable_from_geopandas_dataframe(con, data_dir):
    gpd = pytest.importorskip("geopandas")
    gdf = gpd.read_file(data_dir / "geojson" / "zones.geojson")[:5]

    # Read in memtable
    t = ibis.memtable(gdf)
    # Execute a few rows to force ingestion
    con.to_pandas(t.limit(2).select("geometry"))


@pytest.mark.notimpl(["oracle", "exasol"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(["druid"], raises=AssertionError)
@pytest.mark.notyet(
    ["impala", "mssql", "mysql", "sqlite"],
    reason="backend doesn't support arrays and we don't implement pivot_longer with unions yet",
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["trino"],
    reason="invalid code generated for unnesting a struct",
    raises=TrinoUserError,
)
@pytest.mark.notimpl(
    ["flink"],
    reason="invalid code generated for unnesting a struct",
    raises=Py4JJavaError,
)
def test_pivot_longer(backend):
    diamonds = backend.diamonds
    df = diamonds.execute()
    res = diamonds.pivot_longer(s.cols("x", "y", "z"), names_to="pos", values_to="xyz")
    assert res.schema().names == (
        "carat",
        "cut",
        "color",
        "clarity",
        "depth",
        "table",
        "price",
        "pos",
        "xyz",
    )
    expected = pd.melt(
        df,
        id_vars=[
            "carat",
            "cut",
            "color",
            "clarity",
            "depth",
            "table",
            "price",
        ],
        value_vars=list("xyz"),
        var_name="pos",
        value_name="xyz",
    )
    assert len(res.execute()) == len(expected)


def test_pivot_wider(backend):
    diamonds = backend.diamonds
    expr = (
        diamonds.group_by(["cut", "color"])
        .agg(carat=_.carat.mean())
        .pivot_wider(
            names_from="cut", values_from="carat", names_sort=True, values_agg="mean"
        )
    )
    df = expr.execute()
    assert set(df.columns) == {"color"} | set(
        diamonds[["cut"]].distinct().cut.execute()
    )
    assert len(df) == diamonds.color.nunique().execute()


def test_select_distinct_order_by(backend, alltypes, df):
    res = alltypes.select("int_col").distinct().order_by("int_col").to_pandas()
    sol = df[["int_col"]].drop_duplicates().sort_values("int_col")
    backend.assert_frame_equal(res, sol)


# ideally this should work, but fixing the order by + distinct problem
# introduces another projection which is_star_selection to be false in the new
# outer query
@pytest.mark.notimpl(
    "datafusion",
    raises=Exception,
    reason="bug in datafusion; it's confused by aliasing that swaps column names",
)
def test_select_distinct_order_by_alias(backend, con):
    df = pd.DataFrame({"x": [1, 2, 3, 3], "y": [10, 9, 8, 8]})
    expr = ibis.memtable(df).select(y="x", x="y").distinct().order_by("x", "y")
    sol = (
        df.drop_duplicates()
        .rename(columns={"x": "y", "y": "x"})
        .sort_values(["x", "y"])
    )
    res = con.to_pandas(expr)
    backend.assert_frame_equal(res, sol)


def test_select_distinct_order_by_expr(backend, alltypes, df):
    res = alltypes.select("int_col").distinct().order_by(-_.int_col).to_pandas()
    sol = df[["int_col"]].drop_duplicates().sort_values("int_col", ascending=False)
    backend.assert_frame_equal(res, sol)


@pytest.mark.notimpl(
    ["polars"], reason="We don't fuse these ops yet for non-SQL backends", strict=False
)
@pytest.mark.parametrize(
    "ops",
    [
        param(ops, id="-".join(ops))
        for ops in permutations(("select", "distinct", "filter", "order_by"))
        if ops.index("select") < ops.index("distinct")
    ],
)
def test_select_distinct_filter_order_by_commute(backend, alltypes, df, ops):
    """For simple versions of these ops, the order in which they're called
    doesn't matter, they're all handled in a commutative way."""
    expr = alltypes.select("int_col", "float_col", b=alltypes.id % 33)
    for op in ops:
        if op == "select":
            expr = expr.select("int_col", "b")
        elif op == "distinct":
            expr = expr.distinct()
        elif op == "filter":
            expr = expr.filter(expr.int_col > 5)
        elif op == "order_by":
            expr = expr.order_by(-expr.int_col, expr.b)

    sol = df.assign(b=df.id % 33)[["int_col", "b"]]
    sol = sol[sol.int_col > 5].drop_duplicates()
    sol = sol.set_index([-sol.int_col, sol.b]).sort_index().reset_index(drop=True)
    res = expr.to_pandas()
    backend.assert_frame_equal(res, sol)


@pytest.mark.parametrize(
    "on",
    [
        param(
            ["cut"],
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "mysql"], raises=com.OperationNotDefinedError
                ),
            ],
            id="one",
        ),
        param(
            ["clarity", "cut"],
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "mysql"], raises=com.OperationNotDefinedError
                ),
            ],
            id="many",
        ),
    ],
)
@pytest.mark.parametrize("keep", ["first", "last"])
@pytest.mark.notimpl(
    ["druid", "impala", "oracle"],
    raises=(NotImplementedError, OracleDatabaseError, com.OperationNotDefinedError),
    reason="arbitrary not implemented in the backend",
)
@pytest.mark.notimpl(
    ["polars"],
    raises=com.OperationNotDefinedError,
    reason="backend doesn't implement ops.WindowFunction",
)
@pytest.mark.notimpl(
    ["flink"],
    raises=Py4JJavaError,
    reason="backend doesn't implement deduplication",
)
@pytest.mark.notimpl(
    ["risingwave"],
    raises=com.UnsupportedOperationError,
    reason="first/last requires an order_by",
)
def test_distinct_on_keep(backend, on, keep):
    from ibis import _

    t = backend.diamonds.mutate(one=ibis.literal(1)).mutate(
        idx=ibis.row_number().over(order_by=_.one, rows=(None, 0))
    )

    expr = t.distinct(on=on, keep=keep).order_by(ibis.asc("idx"))
    result = expr.execute()
    df = t.execute()
    expected = (
        df.drop_duplicates(subset=on, keep=keep or False)
        .sort_values(by=["idx"])
        .reset_index(drop=True)
    )
    assert len(result) == len(expected)


@pytest.mark.parametrize(
    "on",
    [
        param(
            ["cut"],
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "mysql"], raises=com.OperationNotDefinedError
                ),
            ],
            id="one",
        ),
        param(
            ["clarity", "cut"],
            marks=[
                pytest.mark.notimpl(
                    ["mssql", "mysql"], raises=com.OperationNotDefinedError
                ),
            ],
            id="many",
        ),
    ],
)
@pytest.mark.notimpl(
    ["druid", "impala", "oracle"],
    raises=(NotImplementedError, OracleDatabaseError, com.OperationNotDefinedError),
    reason="arbitrary not implemented in the backend",
)
@pytest.mark.notimpl(
    ["polars"],
    raises=com.OperationNotDefinedError,
    reason="backend doesn't implement ops.WindowFunction",
)
@pytest.mark.notimpl(
    ["flink"],
    raises=Py4JJavaError,
    reason="backend doesn't implement deduplication",
)
@pytest.mark.notimpl(
    ["risingwave"],
    raises=com.UnsupportedOperationError,
    reason="first/last requires an order_by",
)
def test_distinct_on_keep_is_none(backend, on):
    from ibis import _

    t = backend.diamonds.mutate(one=ibis.literal(1)).mutate(
        idx=ibis.row_number().over(order_by=_.one, rows=(None, 0))
    )

    expr = t.distinct(on=on, keep=None).order_by(ibis.asc("idx"))
    result = expr.execute()
    df = t.execute()
    expected = (
        df.drop_duplicates(subset=on, keep=False)
        .sort_values(by=["idx"])
        .reset_index(drop=True)
    )
    assert len(result) == len(expected)


@pytest.mark.notimpl(["risingwave", "flink", "exasol"])
@pytest.mark.notyet(
    [
        "sqlite",
        "datafusion",
        "druid",  # not sure what's going on here
        "mysql",  # CHECKSUM TABLE but not column
        "trino",  # checksum returns varbinary
        "athena",
    ]
)
@pytest.mark.parametrize(
    "dtype",
    [
        param(
            "smallint",
            marks=pytest.mark.notyet(
                ["bigquery"], reason="only supports bytes and strings"
            ),
        ),
        param(
            "int",
            marks=pytest.mark.notyet(
                ["bigquery"], reason="only supports bytes and strings"
            ),
        ),
        param(
            "bigint",
            marks=pytest.mark.notyet(
                ["bigquery"], reason="only supports bytes and strings"
            ),
        ),
        param(
            "float",
            marks=pytest.mark.notyet(
                ["bigquery"], reason="only supports bytes and strings"
            ),
        ),
        param(
            "double",
            marks=pytest.mark.notyet(
                ["bigquery"], reason="only supports bytes and strings"
            ),
        ),
        "string",
    ],
)
def test_hash(backend, alltypes, dtype):
    # check that multiple executions return the same result
    h1 = alltypes[f"{dtype}_col"].hash().execute(limit=20)
    h2 = alltypes[f"{dtype}_col"].hash().execute(limit=20)
    backend.assert_series_equal(h1, h2)
    # check that the result is a signed 64-bit integer, no nulls
    assert h1.dtype == "i8"
    assert h1.notnull().all()


@pytest.mark.notimpl(["trino", "oracle", "exasol", "snowflake", "athena"])
@pytest.mark.notyet(
    [
        "datafusion",
        "druid",
        "duckdb",
        "flink",
        "impala",
        "mysql",
        "polars",
        "postgres",
        "pyspark",
        "risingwave",
        "sqlite",
        "databricks",
    ]
)
def test_hashbytes(backend, alltypes):
    h1 = alltypes.order_by("id").string_col.hashbytes().execute(limit=10)
    df = alltypes.order_by("id").execute(limit=10)

    import hashlib

    def hash_256(col):
        return hashlib.sha256(col.encode()).digest()

    h2 = df["string_col"].apply(hash_256).rename("HashBytes(string_col)")

    backend.assert_series_equal(h1, h2)


@pytest.mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "datafusion",
        "flink",
        "impala",
        "mysql",
        "oracle",
        "polars",
        "postgres",
        "risingwave",
        "snowflake",
        "trino",
        "athena",
    ]
)
@pytest.mark.notyet(["druid", "polars", "sqlite"])
def test_hexdigest(backend, alltypes):
    h1 = alltypes.order_by("id").string_col.hexdigest().execute(limit=10)
    df = alltypes.order_by("id").execute(limit=10)

    import hashlib

    def hash_256(col):
        return hashlib.sha256(col.encode()).hexdigest()

    h2 = df["string_col"].apply(hash_256).rename("HexDigest(string_col)")

    backend.assert_series_equal(h1, h2)


@pytest.mark.parametrize(
    ("from_type", "to_type", "from_val", "expected"),
    [
        param("int", "float", 0, 0.0, id="int_to_float"),
        param("float", "int", 0.0, 0, id="float_to_int"),
        param("string", "int", "0", 0, id="string_to_int"),
        param("string", "float", "0", 0.0, id="string_to_float"),
        param(
            "array<int>",
            "array<string>",
            [0, 1, 2],
            ["0", "1", "2"],
            marks=[
                pytest.mark.notimpl(["druid"], raises=PyDruidProgrammingError),
                pytest.mark.notimpl(["oracle"], raises=com.UnsupportedBackendType),
                pytest.mark.notyet(["bigquery"], raises=GoogleBadRequest),
                pytest.mark.notimpl(["snowflake"], raises=AssertionError),
                pytest.mark.never(
                    ["exasol", "impala", "mssql", "mysql", "sqlite"],
                    reason="backend doesn't support arrays",
                ),
            ],
            id="array",
        ),
        param(
            "struct<a: int, b: string>",
            "struct<a: string, b: int>",
            {"a": 0, "b": "1"},
            {"a": "0", "b": 1},
            marks=[
                pytest.mark.notimpl(["flink"], raises=Py4JJavaError),
                pytest.mark.notimpl(["druid"], raises=PyDruidProgrammingError),
                pytest.mark.notimpl(["oracle"], raises=OracleDatabaseError),
                pytest.mark.notimpl(["postgres"], raises=PsycoPgSyntaxError),
                pytest.mark.notimpl(["risingwave"], raises=PsycoPg2InternalError),
                pytest.mark.notimpl(["snowflake"], raises=AssertionError),
                pytest.mark.never(
                    ["datafusion", "exasol", "impala", "mssql", "mysql", "sqlite"],
                    reason="backend doesn't support structs",
                ),
            ],
            id="struct",
        ),
    ],
)
def test_cast(con, from_type, to_type, from_val, expected):
    expr = ibis.literal(from_val, type=from_type).cast(to_type)
    result = con.execute(expr)
    assert result == expected


@pytest.mark.notimpl(["oracle", "sqlite"])
@pytest.mark.parametrize(
    ("from_val", "to_type", "expected"),
    [
        param(0, "float", 0.0),
        param(0.0, "int", 0),
        param("0", "int", 0),
        param("0.0", "float", 0.0),
        param(
            datetime.datetime(2023, 1, 1),
            "int",
            1672531200,
            marks=[
                pytest.mark.notyet(["duckdb", "impala"], reason="casts to NULL"),
                pytest.mark.notyet(["bigquery"], raises=GoogleBadRequest),
                pytest.mark.notyet(["snowflake"], raises=SnowflakeProgrammingError),
                pytest.mark.notyet(["trino"], raises=TrinoUserError),
                pytest.mark.notyet(["athena"], raises=PyAthenaOperationalError),
                pytest.mark.notyet(["exasol"], raises=ExaQueryError),
                pytest.mark.notimpl(
                    ["druid"], reason="casts to 1672531200000 (millisecond)"
                ),
                pytest.mark.notimpl(
                    ["polars"], reason="casts to 1672531200000000000 (nanoseconds)"
                ),
                pytest.mark.notimpl(
                    ["datafusion"], reason="casts to 1672531200000000 (microseconds)"
                ),
                pytest.mark.notimpl(["mysql"], reason="returns 20230101000000"),
                pytest.mark.notyet(["mssql"], raises=PyODBCDataError),
            ],
        ),
    ],
    ids=str,
)
def test_try_cast(con, from_val, to_type, expected):
    expr = ibis.literal(from_val).try_cast(to_type)
    result = con.execute(expr)
    assert result == expected


@pytest.mark.notimpl(
    [
        "datafusion",
        "druid",
        "exasol",
        "mysql",
        "oracle",
        "postgres",
        "risingwave",
        "sqlite",
    ]
)
@pytest.mark.parametrize(
    ("from_val", "to_type"),
    [
        param("a", "int"),
        param(
            datetime.datetime(2023, 1, 1),
            "int",
            marks=[
                pytest.mark.never(
                    ["clickhouse", "pyspark", "flink", "databricks"],
                    reason="casts to 1672531200",
                ),
                pytest.mark.notyet(["bigquery"], raises=GoogleBadRequest),
                pytest.mark.notyet(["snowflake"], raises=SnowflakeProgrammingError),
                pytest.mark.notyet(["trino"], raises=TrinoUserError),
                pytest.mark.notyet(["athena"], raises=PyAthenaOperationalError),
                pytest.mark.notyet(["mssql"], raises=PyODBCDataError),
                pytest.mark.notimpl(["polars"], reason="casts to 1672531200000000000"),
            ],
        ),
    ],
    ids=str,
)
def test_try_cast_null(con, from_val, to_type):
    assert pd.isna(con.execute(ibis.literal(from_val).try_cast(to_type)))


@pytest.mark.notimpl(
    [
        "datafusion",
        "druid",
        "mysql",
        "oracle",
        "postgres",
        "risingwave",
        "snowflake",
        "sqlite",
        "exasol",
    ]
)
def test_try_cast_table(backend, con):
    df = pd.DataFrame({"a": ["1", "2", None], "b": ["1.0", "2.2", "goodbye"]})

    expected = pd.DataFrame({"a": [1.0, 2.0, None], "b": [1.0, 2.2, None]})

    t = ibis.memtable(df)

    backend.assert_frame_equal(
        con.execute(t.try_cast({"a": "int", "b": "float"}).order_by("a")), expected
    )


@pytest.mark.notimpl(
    ["datafusion", "mysql", "oracle", "postgres", "risingwave", "sqlite", "exasol"]
)
@pytest.mark.notimpl(["druid"], strict=False)
@pytest.mark.parametrize(
    ("from_val", "to_type", "func"),
    [
        param("a", "float", pd.isna, id="string-to-float"),
        param(
            datetime.datetime(2023, 1, 1),
            "float",
            pd.isna,
            marks=[
                pytest.mark.notyet(
                    ["clickhouse", "polars", "flink", "pyspark", "databricks"],
                    reason="casts this to to a number",
                ),
                pytest.mark.notyet(["bigquery"], raises=GoogleBadRequest),
                pytest.mark.notyet(["snowflake"], raises=SnowflakeProgrammingError),
                pytest.mark.notyet(["trino"], raises=TrinoUserError),
                pytest.mark.notyet(["athena"], raises=PyAthenaOperationalError),
                pytest.mark.notyet(["mssql"], raises=PyODBCDataError),
            ],
            id="datetime-to-float",
        ),
    ],
)
def test_try_cast_func(con, from_val, to_type, func):
    expr = ibis.literal(from_val).try_cast(to_type)
    result = con.execute(expr)
    assert func(result)


@pytest.mark.parametrize(
    ("slc", "expected_count_fn"),
    [
        ###################
        ### NONE/ZERO start
        # no stop
        param(slice(None, 0), lambda _: 0, id="[:0]"),
        param(
            slice(None, None),
            lambda t: t.count().to_pandas(),
            id="[:]",
        ),
        param(slice(0, 0), lambda _: 0, id="[0:0]"),
        param(
            slice(0, None),
            lambda t: t.count().to_pandas(),
            id="[0:]",
        ),
        # positive stop
        param(slice(None, 2), lambda _: 2, id="[:2]"),
        param(slice(0, 2), lambda _: 2, id="[0:2]"),
        ##################
        ### NEGATIVE start
        # zero stop
        param(slice(-3, 0), lambda _: 0, id="[-3:0]"),
        # negative stop
        param(slice(-3, -3), lambda _: 0, id="[-3:-3]"),
        param(slice(-3, -4), lambda _: 0, id="[-3:-4]"),
        param(slice(-3, -5), lambda _: 0, id="[-3:-5]"),
        ##################
        ### POSITIVE start
        # no stop
        param(
            slice(3, 0),
            lambda _: 0,
            id="[3:0]",
            marks=[
                pytest.mark.never(
                    ["impala"],
                    raises=ImpalaHiveServer2Error,
                    reason="doesn't support OFFSET without ORDER BY",
                ),
                pytest.mark.notyet(
                    ["exasol"],
                    raises=ExaQueryError,
                    reason="doesn't support OFFSET without ORDER BY",
                ),
                pytest.mark.notyet(["oracle"], raises=com.UnsupportedArgumentError),
                pytest.mark.never(
                    ["mssql"],
                    raises=PyODBCProgrammingError,
                    reason="sqlglot generates code that requires > 0 fetch rows",
                ),
            ],
        ),
        param(
            slice(3, None),
            lambda t: t.count().to_pandas() - 3,
            id="[3:]",
            marks=[
                pytest.mark.notyet(
                    ["bigquery"],
                    raises=GoogleBadRequest,
                    reason="bigquery doesn't support OFFSET without LIMIT",
                ),
                pytest.mark.notyet(["exasol"], raises=ExaQueryError),
                pytest.mark.never(
                    ["impala"],
                    raises=ImpalaHiveServer2Error,
                    reason="impala doesn't support OFFSET without ORDER BY",
                ),
                pytest.mark.notyet(["oracle"], raises=com.UnsupportedArgumentError),
            ],
        ),
        # positive stop
        param(
            slice(3, 2),
            lambda _: 0,
            id="[3:2]",
            marks=[
                pytest.mark.never(
                    ["impala"],
                    raises=ImpalaHiveServer2Error,
                    reason="doesn't support OFFSET without ORDER BY",
                ),
                pytest.mark.notyet(
                    ["exasol"],
                    raises=ExaQueryError,
                    reason="doesn't support OFFSET without ORDER BY",
                ),
                pytest.mark.notyet(["oracle"], raises=com.UnsupportedArgumentError),
                pytest.mark.never(
                    ["mssql"],
                    raises=PyODBCProgrammingError,
                    reason="sqlglot generates code that requires > 0 fetch rows",
                ),
            ],
        ),
        param(
            slice(3, 4),
            lambda _: 1,
            id="[3:4]",
            marks=[
                pytest.mark.notyet(["exasol"], raises=ExaQueryError),
                pytest.mark.notyet(["oracle"], raises=com.UnsupportedArgumentError),
                pytest.mark.notyet(
                    ["impala"],
                    raises=ImpalaHiveServer2Error,
                    reason="impala doesn't support OFFSET without ORDER BY",
                ),
            ],
        ),
    ],
)
def test_static_table_slice(backend, slc, expected_count_fn):
    t = backend.functional_alltypes

    rows = t[slc]
    count = rows.count().to_pandas()

    expected_count = expected_count_fn(t)
    assert count == expected_count


@pytest.mark.notyet("clickhouse", raises=ClickHouseDatabaseError)
@pytest.mark.parametrize(
    ("slc", "expected_count_fn"),
    [
        ### NONE/ZERO start
        # negative stop
        param(slice(None, -2), lambda t: t.count().to_pandas() - 2, id="[:-2]"),
        param(slice(0, -2), lambda t: t.count().to_pandas() - 2, id="[0:-2]"),
        # no stop
        param(slice(-3, None), lambda _: 3, id="[-3:]"),
        ##################
        ### NEGATIVE start
        # negative stop
        param(slice(-3, -2), lambda _: 1, id="[-3:-2]"),
        # positive stop
        param(
            slice(-4000, 7000),
            lambda _: 3700,
            id="[-4000:7000]",
            marks=[pytest.mark.notyet("clickhouse", raises=ClickHouseDatabaseError)],
        ),
        param(
            slice(-3, 2),
            lambda _: 0,
            id="[-3:2]",
            marks=[
                pytest.mark.never(
                    ["mssql"],
                    raises=PyODBCProgrammingError,
                    reason="sqlglot generates code that requires > 0 fetch rows",
                ),
            ],
        ),
        ##################
        ### POSITIVE start
        # negative stop
        param(slice(3, -2), lambda t: t.count().to_pandas() - 5, id="[3:-2]"),
        param(slice(3, -4), lambda t: t.count().to_pandas() - 7, id="[3:-4]"),
    ],
    ids=str,
)
@pytest.mark.notyet(
    ["mysql"],
    raises=MySQLProgrammingError,
    reason="backend doesn't support dynamic limit/offset",
)
@pytest.mark.notyet(
    ["snowflake"],
    raises=SnowflakeProgrammingError,
    reason="backend doesn't support dynamic limit/offset",
)
@pytest.mark.notyet(
    ["oracle"],
    raises=com.UnsupportedArgumentError,
    reason="Removed half-baked dynamic offset functionality for now",
)
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="risingwave doesn't support limit/offset",
)
@pytest.mark.notyet(
    ["trino"],
    raises=TrinoUserError,
    reason="backend doesn't support dynamic limit/offset",
)
@pytest.mark.notyet(
    ["athena"],
    raises=PyAthenaDatabaseError,
    reason="backend doesn't support dynamic limit/offset",
)
@pytest.mark.notimpl(["exasol"], raises=ExaQueryError)
@pytest.mark.notyet(["druid"], reason="druid doesn't support dynamic limit/offset")
@pytest.mark.notyet(["polars"], reason="polars doesn't support dynamic limit/offset")
@pytest.mark.notyet(
    ["bigquery"],
    reason="bigquery doesn't support dynamic limit/offset",
    raises=GoogleBadRequest,
)
@pytest.mark.notyet(
    ["datafusion"],
    reason='Exception: DataFusion error: Plan("LIMIT must not be negative")',
    raises=Exception,
)
@pytest.mark.never(
    ["impala"],
    reason="impala doesn't support dynamic limit/offset",
    raises=ImpalaHiveServer2Error,
)
@pytest.mark.notyet(
    ["pyspark", "databricks"],
    reason="pyspark and databricks don't support dynamic limit/offset",
)
@pytest.mark.notyet(["flink"], reason="flink doesn't support dynamic limit/offset")
def test_dynamic_table_slice(backend, slc, expected_count_fn):
    t = backend.functional_alltypes

    rows = t[slc]
    count = rows.count().to_pandas()

    expected_count = expected_count_fn(t)
    assert count == expected_count


@pytest.mark.notyet(
    ["mysql"],
    raises=MySQLProgrammingError,
    reason="backend doesn't support dynamic limit/offset",
)
@pytest.mark.notyet(
    ["snowflake"],
    raises=SnowflakeProgrammingError,
    reason="backend doesn't support dynamic limit/offset",
)
@pytest.mark.notyet(
    ["oracle"],
    raises=com.UnsupportedArgumentError,
    reason="Removed half-baked dynamic offset functionality for now",
)
@pytest.mark.notimpl(
    ["trino"],
    raises=TrinoUserError,
    reason="backend doesn't support dynamic limit/offset",
)
@pytest.mark.notyet(
    ["athena"],
    raises=PyAthenaDatabaseError,
    reason="backend doesn't support dynamic limit/offset",
)
@pytest.mark.notimpl(["exasol"], raises=ExaQueryError)
@pytest.mark.notyet(["druid"], reason="druid doesn't support dynamic limit/offset")
@pytest.mark.notyet(["polars"], reason="polars doesn't support dynamic limit/offset")
@pytest.mark.notyet(
    ["bigquery"],
    reason="bigquery doesn't support dynamic limit/offset",
    raises=GoogleBadRequest,
)
@pytest.mark.notyet(
    ["datafusion"],
    reason='Exception: DataFusion error: Plan("Unexpected expression in OFFSET clause")',
    raises=Exception,
)
@pytest.mark.never(
    ["impala"],
    reason="impala doesn't support dynamic limit/offset",
    raises=ImpalaHiveServer2Error,
)
@pytest.mark.notyet(
    ["pyspark", "databricks"],
    reason="pyspark and databricks don't support dynamic limit/offset",
)
@pytest.mark.notyet(["flink"], reason="flink doesn't support dynamic limit/offset")
@pytest.mark.notyet(
    ["mssql"],
    reason="doesn't support dynamic limit/offset; compiles incorrectly in sqlglot",
    raises=PyODBCProgrammingError,
)
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="risingwave doesn't support limit/offset",
)
def test_dynamic_table_slice_with_computed_offset(backend):
    t = backend.functional_alltypes

    col = "id"
    df = t[[col]].to_pandas()

    assert len(df) == df[col].nunique()

    n = 10

    expr = t[[col]].order_by(col)[-n:]

    expected = df.sort_values([col]).iloc[-n:].reset_index(drop=True)
    result = expr.to_pandas()

    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["druid", "risingwave"], raises=com.OperationNotDefinedError)
@pytest.mark.parametrize("method", ["row", "block"])
@pytest.mark.parametrize("subquery", [True, False], ids=["subquery", "table"])
@pytest.mark.xfail_version(pyspark=["sqlglot==25.17.0"])
def test_sample(backend, method, alltypes, subquery):
    if subquery:
        alltypes = alltypes.filter(_.int_col >= 2)

    total_rows = alltypes.count().execute()
    empty = alltypes.limit(1).execute().iloc[:0]

    df = alltypes.sample(0.1, method=method).execute()
    assert len(df) <= total_rows
    backend.assert_frame_equal(empty, df.iloc[:0])


@pytest.mark.notimpl(["druid", "risingwave"], raises=com.OperationNotDefinedError)
def test_sample_memtable(con, backend):
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    res = con.execute(ibis.memtable(df).sample(0.5))
    assert len(res) <= 4
    backend.assert_frame_equal(res.iloc[:0], df.iloc[:0])


@pytest.mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "datafusion",
        "druid",
        "flink",
        "impala",
        "mssql",
        "mysql",
        "oracle",
        "polars",
        "risingwave",
        "sqlite",
        "trino",
        "exasol",
        "pyspark",
        "databricks",
        "athena",
    ]
)
def test_sample_with_seed(backend):
    t = backend.functional_alltypes
    expr = t.sample(0.1, seed=1234)
    df1 = expr.to_pandas()
    df2 = expr.to_pandas()
    backend.assert_frame_equal(df1, df2)


def test_simple_memtable_construct(con):
    t = ibis.memtable({"a": [1, 2]})
    expr = t.a
    expected = [1.0, 2.0]
    assert sorted(con.to_pandas(expr).tolist()) == expected


def test_select_mutate_with_dict(backend):
    t = backend.functional_alltypes
    expr = t.mutate({"a": 1.0}).select("a").limit(1)

    result = expr.execute()
    expected = pd.DataFrame({"a": [1.0]})

    backend.assert_frame_equal(result, expected)

    expr = t.select({"a": ibis.literal(1.0)}).limit(1)
    backend.assert_frame_equal(result, expected)


def test_select_scalar(alltypes):
    res = alltypes.select(y=ibis.literal(1)).limit(3).execute()
    assert len(res.y) == 3
    assert (res.y == 1).all()


@pytest.mark.notimpl(["mssql"], reason="incorrect syntax")
def test_isnull_equality(con, backend, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", con)
    t = ibis.memtable({"x": ["a", "b", None], "y": ["c", None, None], "z": [1, 2, 3]})
    expr = t.mutate(out=t.x.isnull() == t.y.isnull()).order_by("z").select("out")
    result = expr.to_pandas()

    expected = pd.DataFrame({"out": [True, False, True]})

    backend.assert_frame_equal(result, expected)


@pytest.mark.never(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason="Query could not be planned. SQL query requires ordering a table by time column",
)
def test_subsequent_overlapping_order_by(con, backend, alltypes, df):
    ts = alltypes.order_by(ibis.desc("id")).order_by("id")
    result = con.execute(ts)
    expected = df.sort_values("id").reset_index(drop=True)
    backend.assert_frame_equal(result, expected)

    ts2 = ts.order_by(ibis.desc("id"))
    result = con.execute(ts2)
    expected = df.sort_values("id", ascending=False).reset_index(drop=True)
    backend.assert_frame_equal(result, expected)


@pytest.mark.never(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason=(
        "Query could not be planned. SQL query requires ordering a table by time column"
    ),
)
@pytest.mark.xfail_version(
    polars=["polars>=1.32.0"],
    raises=AssertionError,
    reason="polars ignores inner sort keys since 1.32.0",
)
def test_select_sort_sort(backend, alltypes, df):
    t = alltypes
    expr = t.order_by(t.year, t.id.desc()).order_by(t.bool_col)

    result = expr.execute().reset_index(drop=True)

    expected = df.sort_values(
        ["bool_col", "year", "id"], ascending=[True, True, False], kind="mergesort"
    ).reset_index(drop=True)

    expected1 = (
        df.sort_values(["year", "id"], ascending=[True, False], kind="mergesort")
        .sort_values(["bool_col"], kind="mergesort")
        .reset_index(drop=True)
    )

    backend.assert_frame_equal(expected, expected1)
    backend.assert_frame_equal(result, expected)


@pytest.mark.never(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason=(
        "Query could not be planned. SQL query requires ordering a table by time column"
    ),
)
@pytest.mark.xfail_version(
    polars=["polars>=1.32.0"],
    raises=AssertionError,
    reason="polars ignores inner sort keys since 1.32.0",
)
def test_select_sort_sort_deferred(backend, alltypes, df):
    t = alltypes

    expr = t.order_by(t.tinyint_col, t.bool_col, t.id).order_by(
        _.bool_col.asc(), (_.tinyint_col + 1).desc()
    )
    result = expr.execute().reset_index(drop=True)

    df = df.assign(tinyint_col_plus=df.tinyint_col + 1)
    expected = (
        df.sort_values(
            ["bool_col", "tinyint_col_plus", "tinyint_col", "id"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .drop(columns=["tinyint_col_plus"])
        .reset_index(drop=True)
    )
    expected1 = (
        df.sort_values(
            ["tinyint_col", "bool_col", "id"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        .sort_values(
            ["bool_col", "tinyint_col_plus"], ascending=[True, False], kind="mergesort"
        )
        .drop(columns=["tinyint_col_plus"])
        .reset_index(drop=True)
    )

    backend.assert_frame_equal(expected, expected1)
    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(
    ["druid", "athena"],
    raises=AttributeError,
    reason="not yet added the data for this backend",
)
def test_topk_counts_null(con):
    t = con.tables.topk
    tk = t.x.topk(10)
    tkf = tk.filter(_.x.isnull())[1]
    result = con.to_pyarrow(tkf)
    assert result[0].as_py() == 1


@pytest.mark.notyet(
    "clickhouse",
    raises=AssertionError,
    reason="ClickHouse returns False for x.isin([None])",
)
@pytest.mark.never(
    "mssql",
    raises=AssertionError,
    reason="mssql doesn't support null isin semantics in a projection because there is no bool type",
)
def test_null_isin_null_is_null(con):
    t = ibis.memtable({"x": [1]})
    expr = t.x.isin([None])
    assert pd.isna(con.to_pandas(expr).iat[0])


def test_value_counts_on_tables(backend, df):
    t = backend.functional_alltypes
    expr = t[["bigint_col", "int_col"]].value_counts().order_by(s.all())
    result = expr.execute()
    expected = (
        df.groupby(["bigint_col", "int_col"])
        .string_col.count()
        .reset_index()
        .rename(columns=dict(string_col="bigint_col_int_col_count"))
    )
    expected = expected.sort_values(expected.columns.tolist()).reset_index(drop=True)
    backend.assert_frame_equal(result, expected, check_dtype=False)


def test_union_generates_predictable_aliases(con):
    t = ibis.memtable([{"island": "Torgerson", "body_mass_g": 3750, "sex": "male"}])
    sub1 = t.inner_join(t.view(), "island").mutate(island_right=lambda t: t.island)
    sub2 = t.inner_join(t.view(), "sex").mutate(sex_right=lambda t: t.sex)
    expr = ibis.union(sub1, sub2)
    df = con.execute(expr)
    assert len(df) == 2


@pytest.mark.parametrize(
    "id_cols", [s.none(), [], s.cols()], ids=["none", "empty", "cols"]
)
def test_pivot_wider_empty_id_columns(con, backend, id_cols, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", con)
    data = pd.DataFrame(
        {
            "id": range(10),
            "actual": [0, 1, 1, 0, 0, 1, 0, 0, 0, 1],
            "prediction": [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        }
    )
    t = ibis.memtable(data)
    expr = t.mutate(
        outcome=ibis.cases(
            ((_.actual == 0) & (_.prediction == 0), "TN"),
            ((_.actual == 0) & (_.prediction == 1), "FP"),
            ((_.actual == 1) & (_.prediction == 0), "FN"),
            ((_.actual == 1) & (_.prediction == 1), "TP"),
        )
    )
    expr = expr.pivot_wider(
        id_cols=id_cols,
        names_from="outcome",
        values_from="outcome",
        values_agg=_.count(),
        names_sort=True,
    )
    result = expr.to_pandas()
    expected = pd.DataFrame({"FN": [3], "FP": [2], "TN": [4], "TP": [1]})
    backend.assert_frame_equal(result, expected)


@pytest.mark.notyet(
    ["mysql", "risingwave", "impala", "mssql", "druid", "exasol", "oracle", "flink"],
    raises=com.OperationNotDefinedError,
    reason="backend doesn't support Arbitrary agg",
)
def test_simple_pivot_wider(con, backend, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", con)
    data = pd.DataFrame({"outcome": ["yes", "no"], "counted": [3, 4]})
    t = ibis.memtable(data)
    expr = t.pivot_wider(names_from="outcome", values_from="counted", names_sort=True)
    result = expr.to_pandas()
    expected = pd.DataFrame({"no": [4], "yes": [3]})
    backend.assert_frame_equal(result, expected)


def test_named_literal(con, backend):
    lit = ibis.literal(1, type="int64").name("one")
    expr = lit.as_table()
    result = con.to_pandas(expr)
    expected = pd.DataFrame({"one": [1]})
    backend.assert_frame_equal(result, expected)


@pytest.mark.notyet(
    ["polars"],
    raises=PolarsInvalidOperationError,
    reason="n_unique isn't supported on decimal columns",
)
@pytest.mark.notyet(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason="doesn't allow casting Float64 to Decimal(38, 2)",
)
@pytest.mark.notimpl(
    ["oracle"], raises=OracleDatabaseError, reason="incorrect code generated"
)
@pytest.mark.notimpl(
    ["datafusion", "flink", "impala", "mysql", "mssql", "sqlite", "trino", "athena"],
    raises=com.OperationNotDefinedError,
    reason="quantile not implemented",
)
@pytest.mark.notimpl(
    ["druid"],
    raises=com.OperationNotDefinedError,
    reason="standard deviation not implemented",
)
@pytest.mark.notyet(
    ["bigquery"],
    raises=com.UnsupportedBackendType,
    reason="BigQuery only supports two decimal types: (38, 9) and (76, 38)",
)
def test_table_describe_with_multiple_decimal_columns(con):
    t = ibis.memtable({"a": [1, 2, 3], "b": [4, 5, 6]}).cast(
        {"a": "decimal(21, 2)", "b": "decimal(20, 2)"}
    )
    expr = t.describe()
    result = con.to_pyarrow(expr)
    assert len(result) == 2


@pytest.mark.parametrize(
    "input",
    [[], pa.table([[]], pa.schema({"x": pa.int64()}))],
    ids=["list", "pyarrow-table"],
)
@pytest.mark.notyet(["druid"], raises=PyDruidProgrammingError)
@pytest.mark.notyet(
    ["flink"], raises=ValueError, reason="flink doesn't support empty tables"
)
def test_empty_memtable(con, input):
    t = ibis.memtable(input, schema={"x": "int64"})
    assert not len(con.to_pyarrow(t))


def test_order_by_preservation(con):
    tbl = ibis.memtable([{"id": 1, "col": "a"}, {"id": 2, "col": "b"}])
    expr = tbl.order_by("id").select("col").distinct()
    assert len(con.to_pandas(expr)) == 2


def test_distinct_delete_duplicates_entirely(con):
    expr = ibis.memtable({"c1": [1, 2, 3, 6, 6]}).distinct(keep=None)
    res = con.execute(expr)
    assert set(res["c1"].tolist()) == {1, 2, 3}
