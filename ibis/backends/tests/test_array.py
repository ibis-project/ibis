from __future__ import annotations

import functools
from datetime import datetime

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
import pytz
import toolz
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.backends.tests.errors import (
    ClickHouseDatabaseError,
    GoogleBadRequest,
    ImpalaHiveServer2Error,
    MySQLOperationalError,
    PolarsComputeError,
    PsycoPg2IndeterminateDatatype,
    PsycoPg2InternalError,
    PsycoPg2ProgrammingError,
    PsycoPg2SyntaxError,
    Py4JJavaError,
    PySparkAnalysisException,
    TrinoUserError,
)

pytestmark = [
    pytest.mark.never(
        ["sqlite", "mysql", "exasol"], reason="No array support", raises=Exception
    ),
    pytest.mark.never(
        ["mssql"],
        reason="No array support",
        raises=(
            com.UnsupportedBackendType,
            com.OperationNotDefinedError,
            AssertionError,
        ),
    ),
    pytest.mark.never(
        ["mysql"],
        reason="No array support",
        raises=(
            com.UnsupportedBackendType,
            com.OperationNotDefinedError,
            MySQLOperationalError,
        ),
    ),
    pytest.mark.notyet(
        ["impala"],
        reason="No array support",
        raises=(
            com.UnsupportedBackendType,
            com.OperationNotDefinedError,
            ImpalaHiveServer2Error,
        ),
    ),
    pytest.mark.notimpl(["druid", "oracle"], raises=Exception),
]

# NB: We don't check whether results are numpy arrays or lists because this
# varies across backends. At some point we should unify the result type to be
# list.


def test_array_column(backend, alltypes, df):
    expr = ibis.array(
        [alltypes["double_col"], alltypes["double_col"], 5.0, ibis.literal(6.0)]
    )
    assert isinstance(expr, ir.ArrayColumn)

    result = expr.execute()
    expected = df.apply(
        lambda row: [row["double_col"], row["double_col"], 5.0, 6.0],
        axis=1,
    )
    backend.assert_series_equal(result, expected, check_names=False)


ARRAY_BACKEND_TYPES = {
    "clickhouse": "Array(Float64)",
    "snowflake": "ARRAY",
    "trino": "array(double)",
    "bigquery": "ARRAY",
    "duckdb": "DOUBLE[]",
    "postgres": "numeric[]",
    "risingwave": "numeric[]",
    "flink": "ARRAY<DECIMAL(2, 1) NOT NULL> NOT NULL",
}


def test_array_scalar(con):
    expr = ibis.array([1.0, 2.0, 3.0])
    assert isinstance(expr, ir.ArrayScalar)

    result = con.execute(expr.name("tmp"))
    expected = np.array([1.0, 2.0, 3.0])

    assert np.array_equal(result, expected)


@pytest.mark.notimpl(["flink", "polars"], raises=com.OperationNotDefinedError)
def test_array_repeat(con):
    expr = ibis.array([1.0, 2.0]) * 2

    result = con.execute(expr.name("tmp"))
    expected = np.array([1.0, 2.0, 1.0, 2.0])

    assert np.array_equal(result, expected)


def test_array_concat(con):
    left = ibis.literal([1, 2, 3])
    right = ibis.literal([2, 1])
    expr = left + right
    result = con.execute(expr.name("tmp"))
    assert sorted(result) == sorted([1, 2, 3, 2, 1])


def test_array_concat_variadic(con):
    left = ibis.literal([1, 2, 3])
    right = ibis.literal([2, 1])
    expr = left.concat(right, right, right)
    result = con.execute(expr.name("tmp"))
    expected = np.array([1, 2, 3, 2, 1, 2, 1, 2, 1])
    assert np.array_equal(result, expected)


# Issues #2370
@pytest.mark.notimpl(["flink"], raises=Py4JJavaError)
@pytest.mark.notyet(["trino"], raises=TrinoUserError)
def test_array_concat_some_empty(con):
    left = ibis.literal([])
    right = ibis.literal([2, 1])
    expr = left.concat(right)
    result = con.execute(expr.name("tmp"))
    expected = np.array([2, 1])
    assert np.array_equal(result, expected)


def test_array_radd_concat(con):
    left = [1]
    right = ibis.literal([2])
    expr = left + right
    result = con.execute(expr.name("tmp"))
    expected = np.array([1, 2])

    assert np.array_equal(result, expected)


def test_array_length(con):
    expr = ibis.literal([1, 2, 3]).length()
    assert con.execute(expr.name("tmp")) == 3


def test_list_literal(con):
    arr = [1, 2, 3]
    expr = ibis.literal(arr)
    result = con.execute(expr.name("tmp"))

    assert np.array_equal(result, arr)


def test_np_array_literal(con):
    arr = np.array([1, 2, 3])
    expr = ibis.literal(arr)
    result = con.execute(expr.name("tmp"))

    assert np.array_equal(result, arr)


@pytest.mark.parametrize("idx", range(3))
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
def test_array_index(con, idx):
    arr = [1, 2, 3]
    expr = ibis.literal(arr)
    expr = expr[idx]
    result = con.execute(expr)
    assert result == arr[idx]


builtin_array = toolz.compose(
    # these will almost certainly never be supported
    pytest.mark.never(
        ["mysql"],
        reason="array types are unsupported",
        raises=(
            com.OperationNotDefinedError,
            MySQLOperationalError,
            com.UnsupportedBackendType,
        ),
    ),
    pytest.mark.never(
        ["sqlite"],
        reason="array types are unsupported",
        raises=(com.UnsupportedBackendType,),
    ),
)


@builtin_array
@pytest.mark.notyet(
    ["clickhouse", "postgres"],
    reason="backend does not support nullable nested types",
    raises=AssertionError,
)
@pytest.mark.notimpl(
    ["risingwave"],
    raises=AssertionError,
    reason="Do not nest ARRAY types; ARRAY(basetype) handles multi-dimensional arrays of basetype",
)
@pytest.mark.never(
    ["bigquery"], reason="doesn't support arrays of arrays", raises=AssertionError
)
@pytest.mark.never(
    ["snowflake"],
    reason="snowflake has an extremely specialized way of implementing arrays",
    raises=AssertionError,
)
def test_array_discovery(backend):
    t = backend.array_types
    expected = ibis.schema(
        dict(
            x=dt.Array(dt.int64),
            y=dt.Array(dt.string),
            z=dt.Array(dt.float64),
            grouper=dt.string,
            scalar_column=dt.float64,
            multi_dim=dt.Array(dt.Array(dt.int64)),
        )
    )
    assert t.schema() == expected


@builtin_array
@pytest.mark.notyet(
    ["bigquery"],
    reason="BigQuery doesn't support casting array<T> to array<U>",
    raises=GoogleBadRequest,
)
@pytest.mark.notimpl(["datafusion", "flink"], raises=com.OperationNotDefinedError)
def test_unnest_simple(backend):
    array_types = backend.array_types
    expected = (
        array_types.execute()
        .x.explode()
        .reset_index(drop=True)
        .astype("Float64")
        .rename("tmp")
    )
    expr = array_types.x.cast("!array<float64>").unnest()
    result = expr.execute().astype("Float64").rename("tmp")
    assert frozenset(result.values) == frozenset(expected.values)


@builtin_array
@pytest.mark.notimpl(["datafusion", "flink"], raises=com.OperationNotDefinedError)
def test_unnest_complex(backend):
    array_types = backend.array_types
    df = array_types.execute()
    expr = (
        array_types.select(["grouper", "x"])
        .mutate(x=lambda t: t.x.unnest())
        .group_by("grouper")
        .aggregate(count_flat=lambda t: t.x.count())
        .order_by("grouper")
    )
    expected = (
        df[["grouper", "x"]]
        .explode("x")
        .groupby("grouper")
        .x.count()
        .rename("count_flat")
        .reset_index()
        .sort_values("grouper")
        .reset_index(drop=True)
    )
    result = expr.execute()
    backend.assert_frame_equal(result, expected)

    # test that unnest works with to_pyarrow
    assert len(expr.to_pyarrow()) == len(result)


@builtin_array
@pytest.mark.never(
    "pyspark", reason="pyspark throws away nulls in collect_list", raises=AssertionError
)
@pytest.mark.never(
    "clickhouse",
    reason="clickhouse throws away nulls in groupArray",
    raises=AssertionError,
)
@pytest.mark.notimpl(["datafusion", "flink"], raises=com.OperationNotDefinedError)
@pytest.mark.broken(
    "dask", reason="DataFrame.index are different", raises=AssertionError
)
def test_unnest_idempotent(backend):
    array_types = backend.array_types
    df = array_types.execute()
    expr = (
        array_types.select(
            ["scalar_column", array_types.x.cast("!array<int64>").unnest().name("x")]
        )
        .group_by("scalar_column")
        .aggregate(x=lambda t: t.x.collect())
        .order_by("scalar_column")
    )
    result = expr.execute()
    expected = (
        df[["scalar_column", "x"]].sort_values("scalar_column").reset_index(drop=True)
    )
    tm.assert_frame_equal(result, expected)


@builtin_array
@pytest.mark.notimpl(["datafusion", "flink"], raises=com.OperationNotDefinedError)
@pytest.mark.broken(
    "dask", reason="DataFrame.index are different", raises=AssertionError
)
def test_unnest_no_nulls(backend):
    array_types = backend.array_types
    df = array_types.execute()
    expr = (
        array_types.select(
            ["scalar_column", array_types.x.cast("!array<int64>").unnest().name("y")]
        )
        .filter(lambda t: t.y.notnull())
        .group_by("scalar_column")
        .aggregate(x=lambda t: t.y.collect())
        .order_by("scalar_column")
    )
    result = expr.execute()
    expected = (
        df[["scalar_column", "x"]]
        .explode("x")
        .dropna(subset=["x"])
        .groupby("scalar_column")
        .x.apply(lambda xs: [x for x in xs if x is not None])
        .reset_index()
    )
    tm.assert_frame_equal(result, expected)


@builtin_array
@pytest.mark.notimpl("dask", raises=ValueError)
@pytest.mark.notimpl(
    "pandas",
    raises=ValueError,
    reason="all the input arrays must have same number of dimensions",
)
@pytest.mark.notimpl(["datafusion", "flink"], raises=com.OperationNotDefinedError)
def test_unnest_default_name(backend):
    array_types = backend.array_types
    df = array_types.execute()
    expr = (
        array_types.x.cast("!array<int64>") + ibis.array([1]).cast("!array<int64>")
    ).unnest()
    assert expr.get_name().startswith("ArrayConcat(")

    result = expr.name("x").execute()
    expected = df.x.map(lambda x: x + [1]).explode("x")
    assert frozenset(result.astype(object).fillna(pd.NA).values) == frozenset(
        expected.fillna(pd.NA).values
    )


@pytest.mark.parametrize(
    ("start", "stop"),
    [
        (1, 3),
        (1, 1),
        (2, 3),
        (2, 5),
        (None, 3),
        (None, None),
        (3, None),
        (-3, None),
        (-3, -1),
        param(
            None,
            -3,
            marks=[
                pytest.mark.notyet(
                    ["flink"],
                    raises=AssertionError,
                    reason=(
                        "ArraySlice in Flink behaves unexpectedly when"
                        "`start` is None and `stop` is negative."
                    ),
                )
            ],
            id="nulls",
        ),
    ],
)
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(
    ["datafusion"], raises=Exception, reason="array_types table isn't defined"
)
def test_array_slice(backend, start, stop):
    array_types = backend.array_types
    expr = array_types.select(sliced=array_types.y[start:stop])
    result = expr.sliced.execute()
    expected = array_types.y.execute().map(lambda x: x[start:stop])
    assert frozenset(map(tuple, result.values)) == frozenset(
        map(tuple, expected.values)
    )


@builtin_array
@pytest.mark.notimpl(
    [
        "datafusion",
        "flink",
        "polars",
        "snowflake",
        "sqlite",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.broken(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="TODO(Kexiang): seems a bug",
)
@pytest.mark.notimpl(
    ["dask", "pandas"],
    raises=com.OperationNotDefinedError,
    reason="Operation 'ArrayMap' is not implemented for this backend",
)
@pytest.mark.notimpl(
    ["sqlite"], raises=com.UnsupportedBackendType, reason="Unsupported type: Array: ..."
)
@pytest.mark.parametrize(
    ("input", "output"),
    [
        param(
            {"a": [[1, None, 2], [4]]},
            {"a": [[2, None, 3], [5]]},
            marks=[
                pytest.mark.notyet(
                    ["bigquery"],
                    raises=GoogleBadRequest,
                    reason="BigQuery doesn't support arrays with null elements",
                )
            ],
            id="nulls",
        ),
        param({"a": [[1, 2], [4]]}, {"a": [[2, 3], [5]]}, id="no_nulls"),
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        lambda x: x + 1,
        functools.partial(lambda x, y: x + y, y=1),
        ibis._ + 1,
    ],
)
@pytest.mark.broken(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="TODO(Kexiang): seems a bug",
)
def test_array_map(con, input, output, func):
    t = ibis.memtable(input, schema=ibis.schema(dict(a="!array<int8>")))
    t = ibis.memtable(input, schema=ibis.schema(dict(a="!array<int8>")))
    expected = pd.Series(output["a"])

    expr = t.select(a=t.a.map(func))
    result = con.execute(expr.a)
    assert frozenset(map(tuple, result.values)) == frozenset(
        map(tuple, expected.values)
    )


@builtin_array
@pytest.mark.notimpl(
    [
        "dask",
        "datafusion",
        "flink",
        "pandas",
        "polars",
        "snowflake",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["dask", "pandas"],
    raises=com.OperationNotDefinedError,
    reason="Operation 'ArrayMap' is not implemented for this backend",
)
@pytest.mark.notimpl(
    ["sqlite"], raises=com.UnsupportedBackendType, reason="Unsupported type: Array..."
)
@pytest.mark.parametrize(
    ("input", "output"),
    [
        param(
            {"a": [[1, None, 2], [4]]},
            {"a": [[2], [4]]},
            id="nulls",
            marks=[
                pytest.mark.notyet(
                    ["bigquery"],
                    raises=GoogleBadRequest,
                    reason="NULLs are not allowed as array elements",
                )
            ],
        ),
        param({"a": [[1, 2], [4]]}, {"a": [[2], [4]]}, id="no_nulls"),
    ],
)
@pytest.mark.notyet(
    "risingwave",
    raises=PsycoPg2InternalError,
    reason="no support for not null column constraint",
)
@pytest.mark.parametrize(
    "predicate",
    [
        lambda x: x > 1,
        functools.partial(lambda x, y: x > y, y=1),
        ibis._ > 1,
    ],
)
def test_array_filter(con, input, output, predicate):
    t = ibis.memtable(input, schema=ibis.schema(dict(a="!array<int8>")))
    expected = pd.Series(output["a"])

    expr = t.select(a=t.a.filter(predicate))
    result = con.execute(expr.a)
    assert frozenset(map(tuple, result.values)) == frozenset(
        map(tuple, expected.values)
    )


@builtin_array
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
@pytest.mark.parametrize(
    ("col", "value"),
    [
        param(
            "x",
            1,
            marks=[
                pytest.mark.broken(
                    ["flink"],
                    raises=Py4JJavaError,
                    reason="unknown; NPE during execution",
                )
            ],
        ),
        ("y", "a"),
    ],
)
def test_array_contains(backend, con, col, value):
    t = backend.array_types
    expr = t[col].contains(value)
    result = con.execute(expr)
    expected = t[col].execute().map(lambda lst: value in lst)
    assert frozenset(result.values) == frozenset(expected.values)


@builtin_array
@pytest.mark.parametrize(
    ("a", "expected_array"),
    [
        param(
            [[1], [], [42, 42], []],
            [-1, -1, 0, -1],
            id="some-empty",
            marks=[
                pytest.mark.notyet(
                    ["flink"],
                    raises=Py4JJavaError,
                    reason="SQL validation failed; Flink does not support ARRAY[]",
                ),
                pytest.mark.broken(
                    ["datafusion"],
                    raises=Exception,
                    reason="Internal error: start_from index out of bounds",
                ),
            ],
        ),
        param(
            [[1], [1], [42, 42], [1]],
            [-1, -1, 0, -1],
            id="none-empty",
        ),
    ],
)
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
@pytest.mark.notyet(["impala"], raises=com.UnsupportedBackendType)
def test_array_position(con, a, expected_array):
    t = ibis.memtable({"a": a})
    expr = t.a.index(42)
    result = con.execute(expr)
    expected = pd.Series(expected_array, dtype="object")
    assert frozenset(result.values) == frozenset(expected.values)


@builtin_array
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
@pytest.mark.parametrize(
    ("a"),
    [
        param(
            [[3, 2], [], [42, 2], [2, 2], []],
            id="including-empty-array",
            marks=[
                pytest.mark.notyet(
                    ["flink"],
                    raises=Py4JJavaError,
                    reason="SQL validation failed; Flink does not support ARRAY[]",
                )
            ],
        ),
        param([[3, 2], [2], [42, 2], [2, 2], [2]], id="all-non-empty-arrays"),
    ],
)
def test_array_remove(con, a):
    t = ibis.memtable({"a": a})
    expr = t.a.remove(2)
    result = con.execute(expr)
    expected = pd.Series([[3], [], [42], [], []], dtype="object")
    assert frozenset(map(tuple, result.values)) == frozenset(
        map(tuple, expected.values)
    )


@builtin_array
@pytest.mark.notimpl(["datafusion", "polars"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(
    ["sqlite"], raises=com.UnsupportedBackendType, reason="Unsupported type: Array..."
)
@pytest.mark.notyet(
    ["bigquery"],
    raises=GoogleBadRequest,
    reason="BigQuery doesn't support arrays with null elements",
)
@pytest.mark.notyet(
    ["clickhouse"],
    raises=(AssertionError, TypeError),
    reason="clickhouse doesn't support nullable array types",
)
@pytest.mark.notyet(
    ["bigquery"],
    raises=(AssertionError, GoogleBadRequest),
    reason="bigquery doesn't support null elements in arrays",
)
@pytest.mark.broken(
    ["risingwave"], raises=AssertionError, reason="TODO(Kexiang): seems a bug"
)
@pytest.mark.notyet(
    ["flink"], raises=Py4JJavaError, reason="empty arrays not supported"
)
@pytest.mark.parametrize(
    ("input", "expected"),
    [
        param(
            {"a": [[1, 3, 3], [], [42, 42], [], [None], None]},
            [{3, 1}, set(), {42}, set(), {None}, None],
            id="null",
        ),
        param(
            {"a": [[1, 3, 3], [], [42, 42], [], None]},
            [{3, 1}, set(), {42}, set(), None],
            id="not_null",
        ),
    ],
)
def test_array_unique(con, input, expected):
    t = ibis.memtable(input)
    expr = t.a.unique()
    result = con.execute(expr).map(frozenset, na_action="ignore")
    expected = pd.Series(expected, dtype="object").map(frozenset, na_action="ignore")
    assert frozenset(result.values) == frozenset(expected.values)


@builtin_array
@pytest.mark.notimpl(
    ["datafusion", "flink", "polars"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.broken(
    ["risingwave"],
    raises=AssertionError,
    reason="Refer to https://github.com/risingwavelabs/risingwave/issues/14735",
)
def test_array_sort(con):
    t = ibis.memtable({"a": [[3, 2], [], [42, 42], []], "id": range(4)})
    expr = t.mutate(a=t.a.sort()).order_by("id")
    result = con.execute(expr)
    expected = pd.Series([[2, 3], [], [42, 42], []], dtype="object")

    assert frozenset(map(tuple, result["a"].values)) == frozenset(
        map(tuple, expected.values)
    )


@builtin_array
@pytest.mark.notimpl(["datafusion", "polars"], raises=com.OperationNotDefinedError)
@pytest.mark.parametrize(
    ("a", "b", "expected_array"),
    [
        param(
            [[3, 2], [], []],
            [[1, 3], [None], [5]],
            [{1, 2, 3}, {None}, {5}],
            id="including-empty-array",
            marks=[
                pytest.mark.notyet(
                    ["flink"],
                    raises=Py4JJavaError,
                    reason="SQL validation failed; Flink does not support ARRAY[]",
                ),
                pytest.mark.notyet(
                    ["bigquery"],
                    raises=GoogleBadRequest,
                    reason="BigQuery doesn't support arrays with null elements",
                ),
            ],
        ),
        param(
            [[3, 2], [1], [5]],
            [[1, 3], [1], [5]],
            [{1, 2, 3}, {1}, {5}],
            id="all-non-empty-arrays",
        ),
    ],
)
def test_array_union(con, a, b, expected_array):
    t = ibis.memtable({"a": a, "b": b})
    expr = t.a.union(t.b)
    result = con.execute(expr).map(set, na_action="ignore")
    expected = pd.Series(expected_array, dtype="object")

    assert frozenset(map(tuple, result.values)) == frozenset(
        map(tuple, expected.values)
    )


@builtin_array
@pytest.mark.notimpl(
    ["dask", "datafusion", "pandas", "polars", "flink"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["sqlite"], raises=com.UnsupportedBackendType, reason="Unsupported type: Array..."
)
@pytest.mark.broken(
    ["risingwave"],
    raises=AssertionError,
    reason="TODO(Kexiang): seems a bug",
)
@pytest.mark.parametrize(
    "data",
    [
        param(
            {"a": [[3, 2], [], []], "b": [[1, 3], [None], [5]], "c": range(3)},
            id="nulls",
            marks=[
                pytest.mark.notyet(
                    ["bigquery"],
                    raises=GoogleBadRequest,
                    reason="BigQuery doesn't support arrays with null elements",
                )
            ],
        ),
        param(
            {"a": [[3, 2], [], []], "b": [[1, 3], [], [5]], "c": range(3)},
            id="no_nulls",
        ),
    ],
)
def test_array_intersect(con, data):
    t = ibis.memtable(data)
    expr = t.select("c", d=t.a.intersect(t.b)).order_by("c").drop("c").d
    result = con.execute(expr).map(set, na_action="ignore")
    expected = pd.Series([{3}, set(), set()], dtype="object")
    assert len(result) == len(expected)

    assert frozenset(map(tuple, result.values)) == frozenset(
        map(tuple, expected.values)
    )


@builtin_array
@pytest.mark.notimpl(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason="ClickHouse won't accept dicts for struct type values",
)
@pytest.mark.notimpl(["postgres"], raises=PsycoPg2SyntaxError)
@pytest.mark.notimpl(["risingwave"], raises=PsycoPg2InternalError)
@pytest.mark.notimpl(["datafusion", "flink"], raises=com.OperationNotDefinedError)
@pytest.mark.broken(
    ["trino"], reason="inserting maps into structs doesn't work", raises=TrinoUserError
)
def test_unnest_struct(con):
    data = {"value": [[{"a": 1}, {"a": 2}], [{"a": 3}, {"a": 4}]]}
    t = ibis.memtable(data, schema=ibis.schema({"value": "!array<!struct<a: !int>>"}))
    expr = t.value.unnest()

    result = con.execute(expr)

    expected = pd.DataFrame(data).explode("value").iloc[:, 0].reset_index(drop=True)
    tm.assert_series_equal(result, expected)


@builtin_array
@pytest.mark.notimpl(
    [
        "dask",
        "datafusion",
        "druid",
        "oracle",
        "pandas",
        "polars",
        "postgres",
        "risingwave",
        "flink",
    ],
    raises=com.OperationNotDefinedError,
)
def test_zip(backend):
    t = backend.array_types

    x = t.x.execute()
    res = t.x.zip(t.x)
    assert res.type().value_type.names == ("f1", "f2")
    s = res.execute()
    assert len(s[0][0]) == len(res.type().value_type)
    assert len(x[0]) == len(s[0])

    x = t.x.execute()
    res = t.x.zip(t.x, t.x, t.x, t.x, t.x)
    assert res.type().value_type.names == ("f1", "f2", "f3", "f4", "f5", "f6")
    s = res.execute()
    assert len(s[0][0]) == len(res.type().value_type)
    assert len(x[0]) == len(s[0])


@builtin_array
@pytest.mark.notyet(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason="https://github.com/ClickHouse/ClickHouse/issues/41112",
)
@pytest.mark.notimpl(["postgres"], raises=PsycoPg2SyntaxError)
@pytest.mark.notimpl(["risingwave"], raises=PsycoPg2ProgrammingError)
@pytest.mark.notimpl(["datafusion", "flink"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(
    ["polars"],
    raises=com.OperationNotDefinedError,
    reason="polars unnest cannot be compiled outside of a projection",
)
@pytest.mark.notyet(
    ["pyspark"],
    reason="pyspark doesn't seem to support field selection on explode",
    raises=PySparkAnalysisException,
)
@pytest.mark.broken(
    ["trino"], reason="inserting maps into structs doesn't work", raises=TrinoUserError
)
def test_array_of_struct_unnest(con):
    jobs = ibis.memtable(
        {
            "steps": [
                [
                    {"status": "success"},
                    {"status": "success"},
                    {"status": None},
                    {"status": "failure"},
                ],
                [
                    {"status": None},
                    {"status": "success"},
                ],
            ]
        },
        schema=dict(steps="array<struct<status: string>>"),
    )
    expr = jobs.limit(1).steps.unnest().status
    res = con.execute(expr)
    value = res.iat[0]
    # `value` can be `None` because the order of results is arbitrary; observed
    # in the wild with the trino backend
    assert value is None or isinstance(value, str)


@pytest.fixture(scope="module")
def flatten_data():
    return {
        "empty": {"data": [[], [], []], "type": "array<!array<!int64>>"},
        "happy": {
            "data": [[["abc"]], [["bcd"]], [["def"]]],
            "type": "array<!array<!string>>",
        },
        "nulls_only": {"data": [None, None, None], "type": "array<array<string>>"},
        "mixed_nulls": {"data": [[[]], None, [[None]]], "type": "array<array<string>>"},
    }


@pytest.mark.notyet(
    ["bigquery"], reason="BigQuery doesn't support arrays of arrays", raises=TypeError
)
@pytest.mark.notyet(
    ["postgres", "risingwave"],
    reason="Postgres doesn't truly support arrays of arrays",
    raises=(
        com.OperationNotDefinedError,
        PsycoPg2IndeterminateDatatype,
        PsycoPg2InternalError,
    ),
)
@pytest.mark.parametrize(
    ("column", "expected"),
    [
        param("empty", pd.Series([[], [], []], dtype="object"), id="empty"),
        param(
            "happy", pd.Series([["abc"], ["bcd"], ["def"]], dtype="object"), id="happy"
        ),
        param(
            "nulls_only",
            pd.Series([None, None, None], dtype="object"),
            id="nulls_only",
            marks=[
                pytest.mark.notyet(
                    ["clickhouse"],
                    reason="doesn't support nullable array elements",
                    raises=ClickHouseDatabaseError,
                )
            ],
        ),
        param(
            "mixed_nulls",
            pd.Series([[], None, [None]], dtype="object"),
            id="mixed_nulls",
            marks=[
                pytest.mark.notyet(
                    ["clickhouse"],
                    reason="doesn't support nullable array elements",
                    raises=ClickHouseDatabaseError,
                )
            ],
        ),
    ],
)
@pytest.mark.notimpl(["datafusion", "flink"], raises=com.OperationNotDefinedError)
def test_array_flatten(backend, flatten_data, column, expected):
    data = flatten_data[column]
    t = ibis.memtable({column: data["data"]}, schema={column: data["type"]})
    expr = t[column].flatten()
    result = backend.connection.execute(expr)
    backend.assert_series_equal(
        result.sort_values().reset_index(drop=True),
        expected.sort_values().reset_index(drop=True),
        check_names=False,
    )


@pytest.mark.notyet(
    ["datafusion"],
    reason="range isn't implemented upstream",
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(["flink"], raises=com.OperationNotDefinedError)
@pytest.mark.parametrize("n", [-2, 0, 2])
def test_range_single_argument(con, n):
    expr = ibis.range(n)
    result = con.execute(expr)
    assert list(result) == list(range(n))


@pytest.mark.notyet(
    ["datafusion"],
    reason="range and unnest aren't implemented upstream",
    raises=com.OperationNotDefinedError,
)
@pytest.mark.parametrize("n", [-2, 0, 2])
@pytest.mark.notimpl(["polars", "flink"], raises=com.OperationNotDefinedError)
def test_range_single_argument_unnest(con, n):
    expr = ibis.range(n).unnest()
    result = con.execute(expr)
    assert frozenset(result.values) == frozenset(range(n))


@pytest.mark.parametrize("step", [-2, -1, 1, 2])
@pytest.mark.parametrize(
    ("start", "stop"),
    [
        param(-7, -7),
        param(-7, 0),
        param(-7, 7),
        param(0, -7),
        param(0, 0),
        param(0, 7),
        param(7, -7),
        param(7, 0),
        param(7, 7),
    ],
)
@pytest.mark.notyet(
    ["datafusion"],
    reason="range and unnest aren't implemented upstream",
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(["flink"], raises=com.OperationNotDefinedError)
def test_range_start_stop_step(con, start, stop, step):
    expr = ibis.range(start, stop, step)
    result = con.execute(expr)
    assert list(result) == list(range(start, stop, step))


@pytest.mark.parametrize("stop", [-7, 0, 7])
@pytest.mark.parametrize("start", [-7, 0, 7])
@pytest.mark.notyet(
    ["clickhouse"], raises=ClickHouseDatabaseError, reason="not supported upstream"
)
@pytest.mark.notyet(
    ["datafusion"], raises=com.OperationNotDefinedError, reason="not supported upstream"
)
@pytest.mark.notimpl(["flink"], raises=com.OperationNotDefinedError)
@pytest.mark.never(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="Invalid parameter step: step size cannot equal zero",
)
def test_range_start_stop_step_zero(con, start, stop):
    expr = ibis.range(start, stop, 0)
    result = con.execute(expr)
    assert list(result) == []


@pytest.mark.notimpl(
    ["polars"],
    raises=AssertionError,
    reason="ibis hasn't implemented this behavior yet",
)
@pytest.mark.notyet(
    ["datafusion", "flink"],
    raises=com.OperationNotDefinedError,
    reason="backend doesn't support unnest",
)
def test_unnest_empty_array(con):
    t = ibis.memtable({"arr": [[], ["a"], ["a", "b"]]})
    expr = t.arr.unnest()
    result = con.execute(expr)
    assert len(result) == 3


@builtin_array
@pytest.mark.notimpl(
    [
        "datafusion",
        "flink",
        "polars",
        "snowflake",
        "dask",
        "pandas",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(["sqlite"], raises=com.UnsupportedBackendType)
@pytest.mark.notyet(
    "risingwave",
    raises=PsycoPg2InternalError,
    reason="no support for not null column constraint",
)
def test_array_map_with_conflicting_names(backend, con):
    t = ibis.memtable({"x": [[1, 2]]}, schema=ibis.schema(dict(x="!array<int8>")))
    expr = t.select(a=t.x.map(lambda x: x + 1)).select(
        b=lambda t: t.a.filter(lambda a: a > 2)
    )
    result = con.execute(expr)
    expected = pd.DataFrame({"b": [[3]]})
    backend.assert_frame_equal(result, expected)


@builtin_array
@pytest.mark.notimpl(
    [
        "datafusion",
        "flink",
        "polars",
        "snowflake",
        "sqlite",
        "dask",
        "pandas",
        "sqlite",
    ],
    raises=com.OperationNotDefinedError,
)
def test_complex_array_map(con):
    def upper(token):
        return token.upper()

    def swap(token):
        return token.substitute({"abc": "ABC"})

    arr = ibis.array(["abc", "xyz"])

    expr = arr.map(upper)
    assert con.execute(expr) == ["ABC", "XYZ"]

    expr = arr.map(swap)
    assert con.execute(expr) == ["ABC", "xyz"]

    expr = arr.map(lambda token: token.substitute({"abc": "ABC"}))
    assert con.execute(expr) == ["ABC", "xyz"]


timestamp_range_tzinfos = pytest.mark.parametrize(
    "tzinfo",
    [
        param(
            None,
            id="none",
            marks=[
                pytest.mark.notyet(
                    ["bigquery"],
                    raises=com.IbisTypeError,
                    reason="bigquery doesn't support datetime ranges, only timestamp ranges",
                ),
            ],
        ),
        param(
            pytz.UTC,
            id="utc",
            marks=[
                pytest.mark.notyet(
                    ["trino"],
                    raises=TrinoUserError,
                    reason="trino doesn't support timestamp with time zone arguments to its sequence function",
                ),
                pytest.mark.notyet(
                    ["polars"],
                    raises=(TypeError, com.UnsupportedOperationError),
                    reason="polars doesn't work with dateutil timezones",
                ),
            ],
        ),
    ],
)


@pytest.mark.parametrize(
    ("start", "stop", "step", "freq"),
    [
        param(
            datetime(2017, 1, 1),
            datetime(2017, 1, 2),
            ibis.interval(hours=1),
            "1H",
            id="pos",
            marks=pytest.mark.notimpl(
                ["risingwave"],
                raises=PsycoPg2InternalError,
                reason="function make_interval() does not exist",
            ),
        ),
        param(
            datetime(2017, 1, 2),
            datetime(2017, 1, 1),
            ibis.interval(hours=-1),
            "-1H",
            id="neg_inner",
            marks=[
                pytest.mark.broken(
                    ["polars"], raises=AssertionError, reason="returns an empty array"
                ),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="function neg(interval) does not exist",
                ),
            ],
        ),
        param(
            datetime(2017, 1, 2),
            datetime(2017, 1, 1),
            -ibis.interval(hours=1),
            "-1H",
            id="neg_outer",
            marks=[
                pytest.mark.notyet(["polars"], raises=com.UnsupportedOperationError),
                pytest.mark.notyet(["bigquery"], raises=GoogleBadRequest),
                pytest.mark.notyet(
                    ["clickhouse", "snowflake"],
                    raises=com.UnsupportedOperationError,
                ),
                pytest.mark.notimpl(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="function neg(interval) does not exist",
                ),
            ],
        ),
    ],
)
@timestamp_range_tzinfos
@pytest.mark.notimpl(["flink", "datafusion"], raises=com.OperationNotDefinedError)
def test_timestamp_range(con, start, stop, step, freq, tzinfo):
    start = start.replace(tzinfo=tzinfo)
    stop = stop.replace(tzinfo=tzinfo)
    expr = ibis.range(start, stop, step)
    result = con.execute(expr)
    expected = pd.date_range(start, stop, freq=freq, inclusive="left")
    assert list(result) == expected.tolist()


@pytest.mark.parametrize(
    ("start", "stop", "step"),
    [
        param(
            datetime(2017, 1, 1, tzinfo=pytz.UTC),
            datetime(2017, 1, 2, tzinfo=pytz.UTC),
            ibis.interval(hours=0),
            id="pos",
            marks=[
                pytest.mark.notyet(["polars"], raises=PolarsComputeError),
                pytest.mark.notyet(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="function make_interval() does not exist",
                ),
            ],
        ),
        param(
            datetime(2017, 1, 1, tzinfo=pytz.UTC),
            datetime(2017, 1, 2, tzinfo=pytz.UTC),
            -ibis.interval(hours=0),
            id="neg",
            marks=[
                pytest.mark.notyet(["polars"], raises=com.UnsupportedOperationError),
                pytest.mark.notyet(["bigquery"], raises=GoogleBadRequest),
                pytest.mark.notyet(
                    ["clickhouse", "snowflake"],
                    raises=com.UnsupportedOperationError,
                ),
                pytest.mark.notyet(
                    ["risingwave"],
                    raises=PsycoPg2InternalError,
                    reason="function neg(interval) does not exist",
                ),
            ],
        ),
    ],
)
@timestamp_range_tzinfos
@pytest.mark.notimpl(["flink", "datafusion"], raises=com.OperationNotDefinedError)
def test_timestamp_range_zero_step(con, start, stop, step, tzinfo):
    start = start.replace(tzinfo=tzinfo)
    stop = stop.replace(tzinfo=tzinfo)
    expr = ibis.range(start, stop, step)
    result = con.execute(expr)
    assert list(result) == []


@pytest.mark.notimpl(
    ["impala"], raises=AssertionError, reason="backend doesn't support arrays"
)
def test_repr_timestamp_array(con, monkeypatch):
    monkeypatch.setattr(ibis.options, "interactive", True)
    assert ibis.options.interactive is True

    monkeypatch.setattr(ibis.options, "default_backend", con)
    assert ibis.options.default_backend is con

    expr = ibis.array(pd.date_range("2010-01-01", "2010-01-03", freq="D").tolist())
    assert "Translation to backend failed" not in repr(expr)


@pytest.mark.notyet(
    ["datafusion", "flink", "polars"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.broken(["pandas"], raises=ValueError, reason="reindex on duplicate values")
@pytest.mark.broken(
    ["dask"], raises=AssertionError, reason="DataFrame.index are different"
)
def test_unnest_range(con):
    expr = ibis.range(2).unnest().name("x").as_table().mutate({"y": 1.0})
    result = con.execute(expr)
    expected = pd.DataFrame({"x": np.array([0, 1], dtype="int8"), "y": [1.0, 1.0]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        param([1, ibis.literal(2)], [1, 2], id="int-int"),
        param([1.0, ibis.literal(2)], [1.0, 2.0], id="float-int"),
        param([1.0, ibis.literal(2.0)], [1.0, 2.0], id="float-float"),
        param([1, ibis.literal(2.0)], [1.0, 2.0], id="int-float"),
        param([ibis.literal(1), ibis.literal(2.0)], [1.0, 2.0], id="int-float-exprs"),
        param(
            [[1], ibis.literal([2])],
            [[1], [2]],
            id="array",
            marks=[
                pytest.mark.notyet(["bigquery"], raises=GoogleBadRequest),
                pytest.mark.broken(
                    ["polars"],
                    reason="expression input not supported with nested arrays",
                    raises=TypeError,
                ),
            ],
        ),
    ],
)
def test_array_literal_with_exprs(con, input, expected):
    expr = ibis.array(input)
    assert expr.op().shape == ds.scalar
    result = list(con.execute(expr))
    assert result == expected
