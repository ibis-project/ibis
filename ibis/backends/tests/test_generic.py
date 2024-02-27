from __future__ import annotations

import contextlib
import datetime
import decimal
from collections import Counter
from operator import invert, methodcaller, neg

import numpy as np
import pandas as pd
import pytest
import toolz
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.selectors as s
from ibis import _
from ibis.backends.conftest import is_older_than
from ibis.backends.tests.errors import (
    ClickHouseDatabaseError,
    ExaQueryError,
    GoogleBadRequest,
    ImpalaHiveServer2Error,
    MySQLProgrammingError,
    OracleDatabaseError,
    PsycoPg2InternalError,
    PyDruidProgrammingError,
    PyODBCDataError,
    PyODBCProgrammingError,
    SnowflakeProgrammingError,
    TrinoUserError,
)
from ibis.common.annotations import ValidationError

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
}


@pytest.mark.notyet(["flink"], "The runtime does not support untyped `NULL` values.")
def test_null_literal(con, backend):
    expr = ibis.null()
    result = con.execute(expr)
    assert pd.isna(result)

    with contextlib.suppress(com.OperationNotDefinedError):
        backend_name = backend.name()
        assert con.execute(expr.typeof()) == NULL_BACKEND_TYPES[backend_name]


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
}


def test_boolean_literal(con, backend):
    expr = ibis.literal(False, type=dt.boolean)
    result = con.execute(expr)
    assert not result
    assert type(result) in (np.bool_, bool)

    with contextlib.suppress(com.OperationNotDefinedError):
        backend_name = backend.name()
        assert con.execute(expr.typeof()) == BOOLEAN_BACKEND_TYPE[backend_name]


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(ibis.NA.fillna(5), 5, id="na_fillna"),
        param(ibis.literal(5).fillna(10), 5, id="non_na_fillna"),
        param(ibis.literal(5).nullif(5), None, id="nullif_null"),
        param(ibis.literal(10).nullif(5), 10, id="nullif_not_null"),
    ],
)
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
            "none_col", ibis.NA.cast("float64"), methodcaller("isnull"), id="none_col"
        ),
    ],
)
def test_isna(backend, alltypes, col, value, filt):
    table = alltypes.select(**{col: value})
    df = table.execute()

    result = table[filt(table[col])].execute().reset_index(drop=True)
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
def test_column_fillna(backend, alltypes, value):
    table = alltypes.mutate(missing=ibis.literal(value).cast("float64"))
    pd_table = table.execute()

    res = table.mutate(missing=table.missing.fillna(0.0)).execute()
    sol = pd_table.assign(missing=pd_table.missing.fillna(0.0))
    backend.assert_frame_equal(res.reset_index(drop=True), sol.reset_index(drop=True))


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(ibis.coalesce(5, None, 4), 5, id="generic"),
        param(ibis.coalesce(ibis.NA, 4, ibis.NA), 4, id="null_start_end"),
        param(ibis.coalesce(ibis.NA, ibis.NA, 3.14), 3.14, id="non_null_last"),
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


# TODO(dask) - identicalTo - #2553
@pytest.mark.notimpl(["clickhouse", "dask", "druid", "exasol"])
def test_identical_to(backend, alltypes, sorted_df):
    sorted_alltypes = alltypes.order_by("id")
    df = sorted_df
    dt = df[["tinyint_col", "double_col"]]

    ident = sorted_alltypes.tinyint_col.identical_to(sorted_alltypes.double_col)
    expr = sorted_alltypes["id", ident.name("tmp")].order_by("id")
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
    expr = sorted_alltypes[
        "id", sorted_alltypes[column].isin(elements).name("tmp")
    ].order_by("id")
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
    expr = sorted_alltypes[
        "id", sorted_alltypes[column].notin(elements).name("tmp")
    ].order_by("id")
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
    table = sorted_alltypes[predicate_fn(sorted_alltypes)].order_by("id")
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
        "risingwave",
        "sqlite",
        "snowflake",
        "polars",
        "mssql",
        "trino",
        "druid",
        "oracle",
        "exasol",
        "pandas",
        "pyspark",
        "dask",
    ]
)
@pytest.mark.never(
    ["flink"],
    reason="Flink engine does not support generic window clause with no order by",
)
# TODO(kszucs): this is not supported at the expression level
def test_filter_with_window_op(backend, alltypes, sorted_df):
    sorted_alltypes = alltypes.order_by("id")
    table = sorted_alltypes
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
            ibis.case()
            .when(table["int_col"] == 1, 20)
            .when(table["int_col"] == 0, 10)
            .else_(0)
            .end()
            .cast("int64")
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
    t = t.mutate(
        float_col=ibis.case().when(t["bool_col"], t["float_col"]).else_(np.nan).end()
    )

    # Actual test
    t = t[t.columns]
    t = t[~t["float_col"].isnan()]
    t = t.mutate(float_col=t["float_col"].cast("float64"))
    result = t.execute()

    expected = df.copy()
    expected.loc[~df["bool_col"], "float_col"] = None
    expected = expected[~expected["float_col"].isna()].reset_index(drop=True)
    expected = expected.assign(float_col=expected["float_col"].astype("float64"))

    backend.assert_series_equal(result.float_col, expected.float_col)


def test_table_fillna_invalid(alltypes):
    with pytest.raises(
        com.IbisTypeError, match=r"Column 'invalid_col' is not found in table"
    ):
        alltypes.fillna({"invalid_col": 0.0})

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
        param({"int_col": 20}, id="int"),
        param({"double_col": -1, "string_col": "missing"}, id="double-int-str"),
        param({"double_col": -1.5, "string_col": "missing"}, id="double-str"),
    ],
)
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
        alltypes.dropna(subset=["invalid_col"])

    with pytest.raises(ValidationError):
        alltypes.dropna(how="invalid")


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
            ibis.desc("id"),
            {"by": "id", "ascending": False},
        ),
        param(
            ["id", "int_col"],
            {"by": ["id", "int_col"]},
            marks=pytest.mark.xfail_version(dask=["dask<2024.2.0"]),
        ),
        param(
            ["id", ibis.desc("int_col")],
            {"by": ["id", "int_col"], "ascending": [True, False]},
            marks=pytest.mark.xfail_version(dask=["dask<2024.2.0"]),
        ),
    ],
)
@pytest.mark.notimpl(["druid"])
def test_order_by(backend, alltypes, df, key, df_kwargs):
    result = alltypes.filter(_.id < 100).order_by(key).execute()
    expected = df.loc[df.id < 100].sort_values(**df_kwargs)
    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["dask", "pandas", "polars", "mssql", "druid"])
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="function random() does not exist",
)
def test_order_by_random(alltypes):
    expr = alltypes.filter(_.id < 100).order_by(ibis.random()).limit(5)
    r1 = expr.execute()
    r2 = expr.execute()
    assert len(r1) == 5
    assert len(r2) == 5
    # Ensure that multiple executions returns different results
    assert not r1.equals(r2)


@pytest.mark.notyet(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason="Druid only supports trivial unions",
)
def test_table_info(alltypes):
    expr = alltypes.info()
    df = expr.execute()
    assert alltypes.columns == list(df.name)
    assert expr.columns == [
        "name",
        "type",
        "nullable",
        "nulls",
        "non_nulls",
        "null_frac",
        "pos",
    ]
    assert expr.columns == list(df.columns)


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
    expr = alltypes[ibis_op]
    expected = df.loc[pandas_op(df)].sort_values(["id"]).reset_index(drop=True)
    result = expr.execute().sort_values(["id"]).reset_index(drop=True)
    backend.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["druid"])
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
            marks=[pytest.mark.notimpl(["datafusion"])],
        ),
        param(
            (_.bigint_col + 1).notin(_.string_col.length() + 1),
            lambda df: ~(df.bigint_col.add(1)).isin(df.string_col.str.len().add(1)),
            id="notin_expr",
            marks=[pytest.mark.notimpl(["datafusion"])],
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
        param(True, False, invert, id="true_invert"),
        param(False, True, invert, id="false_invert"),
        param(True, False, neg, id="true_negate"),
        param(False, True, neg, id="false_negate"),
    ],
)
def test_logical_negation_literal(con, expr, expected, op):
    assert con.execute(op(ibis.literal(expr)).name("tmp")) == expected


@pytest.mark.parametrize("op", [toolz.identity, invert, neg])
def test_logical_negation_column(backend, alltypes, df, op):
    result = op(alltypes["bool_col"]).name("tmp").execute()
    expected = op(df["bool_col"])
    backend.assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize(
    ("dtype", "zero", "expected"),
    [("int64", 0, 1), ("float64", 0.0, 1.0)],
)
def test_zero_ifnull_literals(con, dtype, zero, expected):
    assert con.execute(ibis.NA.cast(dtype).fillna(0)) == zero
    assert con.execute(ibis.literal(expected, type=dtype).fillna(0)) == expected


def test_zero_ifnull_column(backend, alltypes, df):
    expr = alltypes.int_col.nullif(1).fillna(0).name("tmp")
    result = expr.execute().astype("int32")
    expected = df.int_col.replace(1, 0).rename("tmp").astype("int32")
    backend.assert_series_equal(result, expected)


def test_ifelse_select(backend, alltypes, df):
    table = alltypes
    table = table.select(
        [
            "int_col",
            (
                ibis.ifelse(table["int_col"] == 0, 42, -1)
                .cast("int64")
                .name("where_col")
            ),
        ]
    )

    result = table.execute()

    expected = df.loc[:, ["int_col"]].copy()

    expected["where_col"] = -1
    expected.loc[expected["int_col"] == 0, "where_col"] = 42

    backend.assert_frame_equal(result, expected)


def test_ifelse_column(backend, alltypes, df):
    expr = ibis.ifelse(alltypes["int_col"] == 0, 42, -1).cast("int64").name("where_col")
    result = expr.execute()

    expected = pd.Series(
        np.where(df.int_col == 0, 42, -1),
        name="where_col",
        dtype="int64",
    )

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


@pytest.mark.notimpl(["druid"])
def test_interactive(alltypes, monkeypatch):
    monkeypatch.setattr(ibis.options, "interactive", True)

    expr = alltypes.mutate(
        str_col=_.string_col.replace("1", "").nullif("2"),
        date_col=_.timestamp_col.date(),
        delta_col=lambda t: ibis.now() - t.timestamp_col,
    )

    repr(expr)


def test_correlated_subquery(alltypes):
    expr = alltypes[_.double_col > _.view().double_col]
    assert expr.compile() is not None


@pytest.mark.notimpl(["polars", "pyspark"])
@pytest.mark.broken(
    ["risingwave"],
    raises=AssertionError,
    reason='DataFrame.iloc[:, 0] (column name="playerID") are different',
)
def test_uncorrelated_subquery(backend, batting, batting_df):
    subset_batting = batting[batting.yearID <= 2000]
    expr = batting[_.yearID == subset_batting.yearID.max()]["playerID", "yearID"]
    result = expr.execute()

    expected = batting_df[batting_df.yearID == 2000][["playerID", "yearID"]]
    backend.assert_frame_equal(result, expected)


def test_int_column(alltypes):
    expr = alltypes.mutate(x=1).x
    result = expr.execute()
    assert expr.type() == dt.int8
    assert result.dtype == np.int8


@pytest.mark.notimpl(["druid", "oracle"])
@pytest.mark.never(
    ["bigquery", "sqlite", "snowflake"], reason="backend only implements int64"
)
def test_int_scalar(alltypes):
    expr = alltypes.int_col.min()
    result = expr.execute()
    assert expr.type() == dt.int32
    assert result.dtype == np.int32


@pytest.mark.notimpl(["dask", "datafusion", "pandas", "polars", "druid"])
@pytest.mark.notyet(
    ["clickhouse"], reason="https://github.com/ClickHouse/ClickHouse/issues/6697"
)
@pytest.mark.parametrize("method_name", ["any", "notany"])
def test_exists(batting, awards_players, method_name):
    years = [1980, 1981]
    batting_years = [1871, *years]
    batting = batting[batting.yearID.isin(batting_years)]
    awards_players = awards_players[awards_players.yearID.isin(years)]
    method = methodcaller(method_name)
    expr = batting[method(batting.yearID == awards_players.yearID)]
    result = expr.execute()
    assert not result.empty


@pytest.mark.notimpl(
    [
        "dask",
        "datafusion",
        "mssql",
        "mysql",
        "pandas",
        "pyspark",
        "polars",
        "druid",
        "oracle",
        "exasol",
    ],
    raises=com.OperationNotDefinedError,
)
def test_typeof(con):
    # Other tests also use the typeof operation, but only this test has this operation required.
    expr = ibis.literal(1).typeof()
    result = con.execute(expr)

    assert result is not None


@pytest.mark.broken(["polars"], reason="incorrect answer")
@pytest.mark.notyet(["impala"], reason="can't find table in subquery")
@pytest.mark.notimpl(["datafusion", "druid"])
@pytest.mark.notimpl(["pyspark"], condition=is_older_than("pyspark", "3.5.0"))
@pytest.mark.notyet(["exasol"], raises=ExaQueryError, reason="not supported by exasol")
@pytest.mark.broken(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="https://github.com/risingwavelabs/risingwave/issues/1343",
)
@pytest.mark.xfail_version(dask=["dask<2024.2.0"])
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


@pytest.mark.broken(["polars"], reason="incorrect answer")
@pytest.mark.notimpl(["druid"])
@pytest.mark.xfail_version(
    dask=["dask<2024.2.0"], reason="not supported by the backend"
)
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
                )
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
def test_memtable_column_naming(backend, con, monkeypatch, df, columns, expected):
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
def test_memtable_column_naming_mismatch(backend, con, monkeypatch, df, columns):
    monkeypatch.setattr(ibis.options, "default_backend", con)

    with pytest.raises(ValueError):
        ibis.memtable(df, columns=columns)


@pytest.mark.notimpl(
    ["dask", "pandas", "polars"], raises=NotImplementedError, reason="not a SQL backend"
)
def test_many_subqueries(con, snapshot):
    def query(t, group_cols):
        t2 = t.mutate(key=ibis.row_number().over(ibis.window(order_by=group_cols)))
        return t2.inner_join(t2[["key"]], "key")

    t = ibis.table(dict(street="str"), name="data")

    t2 = query(t, group_cols=["street"])
    t3 = query(t2, group_cols=["street"])

    snapshot.assert_match(str(ibis.to_sql(t3, dialect=con.name)), "out.sql")


@pytest.mark.notimpl(
    ["dask", "pandas", "oracle", "flink", "exasol"], raises=com.OperationNotDefinedError
)
@pytest.mark.notimpl(["druid"], raises=AssertionError)
@pytest.mark.notyet(
    ["datafusion", "impala", "mssql", "mysql", "sqlite"],
    reason="backend doesn't support arrays and we don't implement pivot_longer with unions yet",
    raises=com.OperationNotDefinedError,
)
@pytest.mark.broken(
    ["trino"],
    reason="invalid code generated for unnesting a struct",
    raises=TrinoUserError,
)
def test_pivot_longer(backend):
    diamonds = backend.diamonds
    df = diamonds.execute()
    res = diamonds.pivot_longer(s.c("x", "y", "z"), names_to="pos", values_to="xyz")
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
@pytest.mark.parametrize(
    "keep",
    [
        "first",
        param(
            "last",
            marks=pytest.mark.notimpl(
                ["bigquery", "trino"],
                raises=com.UnsupportedOperationError,
                reason="backend doesn't support how='last'",
            ),
        ),
    ],
)
@pytest.mark.notimpl(
    ["druid", "impala", "oracle"],
    raises=(
        NotImplementedError,
        OracleDatabaseError,
        com.OperationNotDefinedError,
    ),
    reason="arbitrary not implemented in the backend",
)
@pytest.mark.notimpl(
    ["datafusion"],
    raises=com.OperationNotDefinedError,
    reason="backend doesn't implement window functions",
)
@pytest.mark.notimpl(
    ["polars"],
    raises=com.OperationNotDefinedError,
    reason="backend doesn't implement ops.WindowFunction",
)
@pytest.mark.notimpl(
    ["flink"],
    raises=com.OperationNotDefinedError,
    reason="backend doesn't implement deduplication",
)
@pytest.mark.notimpl(
    ["exasol"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="function last(double precision) does not exist, do you mean left or least",
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
    ["exasol"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["datafusion"],
    raises=com.OperationNotDefinedError,
    reason="backend doesn't implement window functions",
)
@pytest.mark.notimpl(
    ["polars"],
    raises=com.OperationNotDefinedError,
    reason="backend doesn't implement ops.WindowFunction",
)
@pytest.mark.notimpl(
    ["flink"],
    raises=com.OperationNotDefinedError,
    reason="backend doesn't implement deduplication",
)
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="function first(double precision) does not exist",
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


@pytest.mark.notimpl(["dask", "pandas", "postgres", "risingwave", "flink", "exasol"])
@pytest.mark.notyet(
    [
        "sqlite",
        "datafusion",
        "druid",  # not sure what's going on here
        "mysql",  # CHECKSUM TABLE but not column
        "trino",  # checksum returns varbinary
    ]
)
def test_hash_consistent(backend, alltypes):
    h1 = alltypes.string_col.hash().execute(limit=10)
    h2 = alltypes.string_col.hash().execute(limit=10)
    backend.assert_series_equal(h1, h2)
    assert h1.dtype in ("i8", "uint64")  # polars likes returning uint64 for this


@pytest.mark.notimpl(["trino", "oracle", "exasol", "snowflake"])
@pytest.mark.notyet(
    [
        "dask",
        "datafusion",
        "druid",
        "duckdb",
        "flink",
        "impala",
        "mysql",
        "pandas",
        "polars",
        "postgres",
        "pyspark",
        "risingwave",
        "sqlite",
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
        "dask",
        "datafusion",
        "exasol",
        "flink",
        "impala",
        "mysql",
        "oracle",
        "pandas",
        "polars",
        "postgres",
        "risingwave",
        "snowflake",
        "trino",
    ]
)
@pytest.mark.notyet(
    [
        "druid",
        "polars",
        "sqlite",
    ]
)
def test_hexdigest(backend, alltypes):
    h1 = alltypes.order_by("id").string_col.hexdigest().execute(limit=10)
    df = alltypes.order_by("id").execute(limit=10)

    import hashlib

    def hash_256(col):
        return hashlib.sha256(col.encode()).hexdigest()

    h2 = df["string_col"].apply(hash_256).rename("HexDigest(string_col)")

    backend.assert_series_equal(h1, h2)


@pytest.mark.notimpl(["pandas", "dask", "oracle", "sqlite"])
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
                pytest.mark.notyet(["exasol"], raises=ExaQueryError),
                pytest.mark.broken(
                    ["druid"], reason="casts to 1672531200000 (millisecond)"
                ),
                pytest.mark.broken(
                    ["polars"], reason="casts to 1672531200000000000 (nanoseconds)"
                ),
                pytest.mark.broken(
                    ["datafusion"], reason="casts to 1672531200000000 (microseconds)"
                ),
                pytest.mark.broken(["mysql"], reason="returns 20230101000000"),
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
        "dask",
        "datafusion",
        "druid",
        "exasol",
        "mysql",
        "oracle",
        "pandas",
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
                    ["clickhouse", "pyspark", "flink"], reason="casts to 1672531200"
                ),
                pytest.mark.notyet(["bigquery"], raises=GoogleBadRequest),
                pytest.mark.notyet(["snowflake"], raises=SnowflakeProgrammingError),
                pytest.mark.notyet(["trino"], raises=TrinoUserError),
                pytest.mark.notyet(["mssql"], raises=PyODBCDataError),
                pytest.mark.broken(["polars"], reason="casts to 1672531200000000000"),
            ],
        ),
    ],
    ids=str,
)
def test_try_cast_null(con, from_val, to_type):
    assert pd.isna(con.execute(ibis.literal(from_val).try_cast(to_type)))


@pytest.mark.notimpl(
    [
        "pandas",
        "dask",
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
    [
        "pandas",
        "dask",
        "datafusion",
        "mysql",
        "oracle",
        "postgres",
        "risingwave",
        "sqlite",
        "exasol",
    ]
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
                    ["clickhouse", "polars", "flink", "pyspark"],
                    reason="casts this to to a number",
                ),
                pytest.mark.notyet(["bigquery"], raises=GoogleBadRequest),
                pytest.mark.notyet(["snowflake"], raises=SnowflakeProgrammingError),
                pytest.mark.notyet(["trino"], raises=TrinoUserError),
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
                pytest.mark.notyet(
                    ["mssql"],
                    raises=PyODBCProgrammingError,
                    reason="mssql doesn't support OFFSET without LIMIT",
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
        param(slice(-4000, 7000), lambda _: 3700, id="[-4000:7000]"),
        param(slice(-3, 2), lambda _: 0, id="[-3:2]"),
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
@pytest.mark.notimpl(["exasol"], raises=ExaQueryError)
@pytest.mark.notyet(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason="clickhouse doesn't support dynamic limit/offset",
)
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
@pytest.mark.notyet(["pyspark"], reason="pyspark doesn't support dynamic limit/offset")
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
@pytest.mark.notimpl(["exasol"], raises=ExaQueryError)
@pytest.mark.notyet(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason="clickhouse doesn't support dynamic limit/offset",
)
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
@pytest.mark.notyet(["pyspark"], reason="pyspark doesn't support dynamic limit/offset")
@pytest.mark.notyet(["flink"], reason="flink doesn't support dynamic limit/offset")
@pytest.mark.notyet(
    ["mssql"],
    reason="doesn't support dynamic limit/offset; compiles incorrectly in sqlglot",
    raises=AssertionError,
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


@pytest.mark.notimpl(["druid", "polars", "snowflake"])
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="function random() does not exist",
)
def test_sample(backend):
    t = backend.functional_alltypes.filter(_.int_col >= 2)

    total_rows = t.count().execute()
    empty = t.limit(1).execute().iloc[:0]

    df = t.sample(0.1, method="row").execute()
    assert len(df) <= total_rows
    backend.assert_frame_equal(empty, df.iloc[:0])

    df = t.sample(0.1, method="block").execute()
    assert len(df) <= total_rows
    backend.assert_frame_equal(empty, df.iloc[:0])


@pytest.mark.notimpl(["druid", "polars", "snowflake"])
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="function random() does not exist",
)
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
        "postgres",
        "risingwave",
        "snowflake",
        "sqlite",
        "trino",
        "exasol",
        "pyspark",
    ]
)
def test_sample_with_seed(backend):
    t = backend.functional_alltypes
    expr = t.sample(0.1, seed=1234)
    df1 = expr.to_pandas()
    df2 = expr.to_pandas()
    backend.assert_frame_equal(df1, df2)


def test_substitute(backend):
    val = "400"
    t = backend.functional_alltypes
    expr = (
        t.string_col.nullif("1")
        .substitute({None: val})
        .name("subs")
        .value_counts()
        .filter(lambda t: t.subs == val)
    )
    assert expr["subs_count"].execute()[0] == t.count().execute() // 10


@pytest.mark.notimpl(
    ["dask", "pandas", "polars"], raises=NotImplementedError, reason="not a SQL backend"
)
def test_simple_memtable_construct(con):
    t = ibis.memtable({"a": [1, 2]})
    expr = t.a
    expected = [1.0, 2.0]
    assert sorted(con.to_pandas(expr).tolist()) == expected
    # we can't generically check for specific sql, even with a snapshot,
    # because memtables have a unique name per table per process, so smoke test
    # it
    assert str(ibis.to_sql(expr, dialect=con.name)).startswith("SELECT")


def test_select_mutate_with_dict(backend):
    t = backend.functional_alltypes
    expr = t.mutate({"a": 1.0}).select("a").limit(1)

    result = expr.execute()
    expected = pd.DataFrame({"a": [1.0]})

    backend.assert_frame_equal(result, expected)

    expr = t.select({"a": ibis.literal(1.0)}).limit(1)
    backend.assert_frame_equal(result, expected)


@pytest.mark.broken(["mssql", "oracle"], reason="incorrect syntax")
def test_isnull_equality(con, backend, monkeypatch):
    monkeypatch.setattr(ibis.options, "default_backend", con)
    t = ibis.memtable({"x": ["a", "b", None], "y": ["c", None, None], "z": [1, 2, 3]})
    expr = t.mutate(out=t.x.isnull() == t.y.isnull()).order_by("z").select("out")
    result = expr.to_pandas()

    expected = pd.DataFrame({"out": [True, False, True]})

    backend.assert_frame_equal(result, expected)


@pytest.mark.broken(
    ["druid"],
    raises=PyDruidProgrammingError,
    reason=(
        "Query could not be planned. A possible reason is [SQL query requires ordering a "
        "table by non-time column [[id]], which is not supported."
    ),
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
