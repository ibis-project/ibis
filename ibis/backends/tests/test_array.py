from __future__ import annotations

import contextlib
import functools

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
import sqlalchemy as sa
import toolz
from pytest import param

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.types as ir

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    from clickhouse_connect.driver.exceptions import (
        DatabaseError as ClickhouseDatabaseError,
    )
except ImportError:
    ClickhouseDatabaseError = None


try:
    from google.api_core.exceptions import BadRequest
except ImportError:
    BadRequest = None

try:
    from pyspark.sql.utils import AnalysisException as PySparkAnalysisException
except ImportError:
    PySparkAnalysisException = None


try:
    from polars.exceptions import PolarsInvalidOperationError
except ImportError:
    PolarsInvalidOperationError = None

pytestmark = [
    pytest.mark.never(
        ["sqlite", "mysql", "mssql"],
        reason="No array support",
        raises=Exception,
    ),
    pytest.mark.notyet(["impala"], reason="No array support", raises=Exception),
    pytest.mark.notimpl(["druid", "oracle"], raises=Exception),
]

# NB: We don't check whether results are numpy arrays or lists because this
# varies across backends. At some point we should unify the result type to be
# list.


@pytest.mark.notimpl(["flink"], raises=com.OperationNotDefinedError)
def test_array_column(backend, alltypes, df):
    expr = ibis.array([alltypes["double_col"], alltypes["double_col"]])
    assert isinstance(expr, ir.ArrayColumn)

    result = expr.execute()
    expected = df.apply(
        lambda row: [row["double_col"], row["double_col"]],
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
    "flink": "ARRAY<DECIMAL(2, 1) NOT NULL> NOT NULL",
}


def test_array_scalar(con, backend):
    expr = ibis.array([1.0, 2.0, 3.0])
    assert isinstance(expr, ir.ArrayScalar)

    result = con.execute(expr.name("tmp"))
    expected = np.array([1.0, 2.0, 3.0])

    assert np.array_equal(result, expected)

    with contextlib.suppress(com.OperationNotDefinedError):
        backend_name = backend.name()
        assert con.execute(expr.typeof()) == ARRAY_BACKEND_TYPES[backend_name]


@pytest.mark.notimpl(["polars", "flink"], raises=com.OperationNotDefinedError)
def test_array_repeat(con):
    expr = ibis.array([1.0, 2.0]) * 2

    result = con.execute(expr.name("tmp"))
    expected = np.array([1.0, 2.0, 1.0, 2.0])

    assert np.array_equal(result, expected)


# Issues #2370
@pytest.mark.notimpl(["flink"], raises=com.OperationNotDefinedError)
def test_array_concat(con):
    left = ibis.literal([1, 2, 3])
    right = ibis.literal([2, 1])
    expr = left + right
    result = con.execute(expr.name("tmp"))
    expected = np.array([1, 2, 3, 2, 1])
    assert np.array_equal(result, expected)


# Issues #2370
@pytest.mark.notimpl(["flink"], raises=com.OperationNotDefinedError)
def test_array_concat_variadic(con):
    left = ibis.literal([1, 2, 3])
    right = ibis.literal([2, 1])
    expr = left.concat(right, right, right)
    result = con.execute(expr.name("tmp"))
    expected = np.array([1, 2, 3, 2, 1, 2, 1, 2, 1])
    assert np.array_equal(result, expected)


# Issues #2370
@pytest.mark.notimpl(["datafusion"], raises=BaseException)
@pytest.mark.notimpl(["flink"], raises=com.OperationNotDefinedError)
@pytest.mark.notyet(
    ["postgres", "trino"],
    raises=sa.exc.ProgrammingError,
    reason="backend can't infer the type of an empty array",
)
def test_array_concat_some_empty(con):
    left = ibis.literal([])
    right = ibis.literal([2, 1])
    expr = left.concat(right)
    result = con.execute(expr.name("tmp"))
    expected = np.array([2, 1])
    assert np.array_equal(result, expected)


@pytest.mark.notimpl(["flink"], raises=com.OperationNotDefinedError)
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
@pytest.mark.notimpl(["polars", "datafusion"], raises=com.OperationNotDefinedError)
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
        raises=com.OperationNotDefinedError,
    ),
    pytest.mark.never(
        ["sqlite"], reason="array types are unsupported", raises=NotImplementedError
    ),
    # someone just needs to implement these
    pytest.mark.notimpl(["flink"], raises=Exception),
)


@builtin_array
@pytest.mark.never(
    [
        "clickhouse",
        "duckdb",
        "pandas",
        "pyspark",
        "snowflake",
        "polars",
        "trino",
        "datafusion",
    ],
    reason="backend does not flatten array types",
    raises=AssertionError,
)
@pytest.mark.never(
    ["snowflake"],
    reason="snowflake has an extremely specialized way of implementing arrays",
    raises=AssertionError,
)
@pytest.mark.never(
    ["bigquery"], reason="doesn't support arrays of arrays", raises=AssertionError
)
@pytest.mark.notimpl(["dask"], raises=AssertionError)
def test_array_discovery_postgres(backend):
    t = backend.array_types
    expected = ibis.schema(
        dict(
            x=dt.Array(dt.int64),
            y=dt.Array(dt.string),
            z=dt.Array(dt.float64),
            grouper=dt.string,
            scalar_column=dt.float64,
            multi_dim=dt.Array(dt.int64),
        )
    )
    assert t.schema() == expected


@builtin_array
@pytest.mark.never(
    ["snowflake"],
    reason="snowflake has an extremely specialized way of implementing arrays",
    raises=AssertionError,
)
@pytest.mark.never(
    [
        "duckdb",
        "pandas",
        "postgres",
        "pyspark",
        "snowflake",
        "polars",
        "trino",
        "datafusion",
    ],
    reason="backend supports nullable nested types",
    raises=AssertionError,
)
@pytest.mark.never(
    ["bigquery"],
    reason="doesn't support arrays of arrays",
    raises=AssertionError,
)
@pytest.mark.never(["dask"], raises=AssertionError, reason="allows nullable types")
def test_array_discovery_clickhouse(backend):
    t = backend.array_types
    expected = ibis.schema(
        dict(
            x=dt.Array(dt.int64, nullable=False),
            y=dt.Array(dt.string, nullable=False),
            z=dt.Array(dt.float64, nullable=False),
            grouper=dt.string,
            scalar_column=dt.float64,
            multi_dim=dt.Array(
                dt.Array(dt.int64, nullable=False),
                nullable=False,
            ),
        )
    )
    assert t.schema() == expected


@builtin_array
@pytest.mark.notyet(
    ["clickhouse", "postgres"],
    reason="backend does not support nullable nested types",
    raises=AssertionError,
)
@pytest.mark.never(
    ["bigquery"],
    reason="doesn't support arrays of arrays",
    raises=AssertionError,
)
@pytest.mark.never(
    ["snowflake"],
    reason="snowflake has an extremely specialized way of implementing arrays",
    raises=AssertionError,
)
def test_array_discovery_desired(backend):
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
@pytest.mark.never(
    [
        "bigquery",
        "clickhouse",
        "dask",
        "datafusion",
        "duckdb",
        "mysql",
        "pandas",
        "polars",
        "postgres",
        "pyspark",
        "sqlite",
        "trino",
    ],
    reason="backend does not implement arrays like snowflake",
    raises=AssertionError,
)
def test_array_discovery_snowflake(backend):
    t = backend.array_types
    expected = ibis.schema(
        dict(
            x=dt.Array(dt.json),
            y=dt.Array(dt.json),
            z=dt.Array(dt.json),
            grouper=dt.string,
            scalar_column=dt.float64,
            multi_dim=dt.Array(dt.json),
        )
    )
    assert t.schema() == expected


@builtin_array
@pytest.mark.notyet(
    ["bigquery"],
    reason="BigQuery doesn't support casting array<T> to array<U>",
    raises=BadRequest,
)
@pytest.mark.notimpl(["dask"], raises=ValueError)
@pytest.mark.notimpl(["datafusion"], raises=com.OperationNotDefinedError)
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
    tm.assert_series_equal(result, expected)


@builtin_array
@pytest.mark.notimpl("dask", raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(["datafusion"], raises=com.OperationNotDefinedError)
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
    tm.assert_frame_equal(result, expected)

    # test that unnest works with to_pyarrow
    assert len(expr.to_pyarrow()) == len(result)


@builtin_array
@pytest.mark.never(
    "pyspark",
    reason="pyspark throws away nulls in collect_list",
    raises=AssertionError,
)
@pytest.mark.never(
    "clickhouse",
    reason="clickhouse throws away nulls in groupArray",
    raises=AssertionError,
)
@pytest.mark.notimpl(["dask"], raises=ValueError)
@pytest.mark.notimpl(["datafusion"], raises=com.OperationNotDefinedError)
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
@pytest.mark.notimpl("dask", raises=ValueError)
@pytest.mark.notimpl(["datafusion"], raises=com.OperationNotDefinedError)
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
@pytest.mark.notimpl(["datafusion"], raises=com.OperationNotDefinedError)
def test_unnest_default_name(backend):
    array_types = backend.array_types
    df = array_types.execute()
    expr = (
        array_types.x.cast("!array<int64>") + ibis.array([1], type="!array<int64>")
    ).unnest()
    assert expr.get_name().startswith("ArrayConcat(")

    result = expr.name("x").execute()
    expected = df.x.map(lambda x: x + [1]).explode("x")
    tm.assert_series_equal(
        result.astype(object).fillna(pd.NA), expected.fillna(pd.NA), check_dtype=False
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
        (None, -3),
        (-3, -1),
    ],
)
@pytest.mark.notimpl(["polars"], raises=com.OperationNotDefinedError)
@pytest.mark.notimpl(
    ["datafusion", "flink"], raises=Exception, reason="array_types table isn't defined"
)
@pytest.mark.notimpl(["dask"], raises=com.OperationNotDefinedError)
def test_array_slice(backend, start, stop):
    array_types = backend.array_types
    expr = array_types.select(sliced=array_types.y[start:stop])
    result = expr.execute()
    expected = pd.DataFrame(
        {"sliced": array_types.y.execute().map(lambda x: x[start:stop])}
    )
    tm.assert_frame_equal(result, expected)


@builtin_array
@pytest.mark.notimpl(
    ["datafusion", "impala", "mssql", "polars", "snowflake", "sqlite", "mysql"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["dask", "pandas"],
    raises=com.OperationNotDefinedError,
    reason="Operation 'ArrayMap' is not implemented for this backend",
)
@pytest.mark.notimpl(
    ["sqlite"],
    raises=NotImplementedError,
    reason="Unsupported type: Array: ...",
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
                    raises=BadRequest,
                    reason="BigQuery doesn't support arrays with null elements",
                )
            ],
            id="nulls",
        ),
        param({"a": [[1, 2], [4]]}, {"a": [[2, 3], [5]]}, id="no_nulls"),
    ],
)
def test_array_map(backend, con, input, output):
    t = ibis.memtable(input, schema=ibis.schema(dict(a="!array<int8>")))
    expected = pd.DataFrame(output)

    expr = t.select(a=t.a.map(lambda x: x + 1))
    result = con.execute(expr)
    backend.assert_frame_equal(result, expected)

    expr = t.select(a=t.a.map(functools.partial(lambda x, y: x + y, y=1)))
    result = con.execute(expr)
    backend.assert_frame_equal(result, expected)


@builtin_array
@pytest.mark.notimpl(
    ["dask", "datafusion", "impala", "mssql", "pandas", "polars", "snowflake", "mysql"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["dask", "pandas"],
    raises=com.OperationNotDefinedError,
    reason="Operation 'ArrayMap' is not implemented for this backend",
)
@pytest.mark.notimpl(
    ["sqlite"], raises=NotImplementedError, reason="Unsupported type: Array..."
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
                    raises=BadRequest,
                    reason="NULLs are not allowed as array elements",
                )
            ],
        ),
        param({"a": [[1, 2], [4]]}, {"a": [[2], [4]]}, id="no_nulls"),
    ],
)
def test_array_filter(backend, con, input, output):
    t = ibis.memtable(input, schema=ibis.schema(dict(a="!array<int8>")))
    expected = pd.DataFrame(output)

    expr = t.select(a=t.a.filter(lambda x: x > 1))
    result = con.execute(expr)
    backend.assert_frame_equal(result, expected)

    expr = t.select(a=t.a.filter(functools.partial(lambda x, y: x > y, y=1)))
    result = con.execute(expr)
    backend.assert_frame_equal(result, expected)


@builtin_array
@pytest.mark.notimpl(
    ["mssql", "pandas", "polars", "postgres"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(["dask"], raises=com.OperationNotDefinedError)
@pytest.mark.never(["impala"], reason="array_types table isn't defined")
def test_array_contains(backend, con):
    t = backend.array_types
    expr = t.x.contains(1)
    result = con.execute(expr)
    expected = t.x.execute().map(lambda lst: 1 in lst)
    backend.assert_series_equal(result, expected, check_names=False)


@builtin_array
@pytest.mark.notimpl(
    ["dask", "impala", "mssql", "pandas", "polars"],
    raises=com.OperationNotDefinedError,
)
def test_array_position(backend, con):
    t = ibis.memtable({"a": [[1], [], [42, 42], []]})
    expr = t.a.index(42)
    result = con.execute(expr)
    expected = pd.Series([-1, -1, 0, -1], dtype="object")
    backend.assert_series_equal(result, expected, check_names=False, check_dtype=False)


@builtin_array
@pytest.mark.notimpl(
    ["dask", "impala", "mssql", "pandas", "polars"],
    raises=com.OperationNotDefinedError,
)
def test_array_remove(backend, con):
    t = ibis.memtable({"a": [[3, 2], [], [42, 2], [2, 2], []]})
    expr = t.a.remove(2)
    result = con.execute(expr)
    expected = pd.Series([[3], [], [42], [], []], dtype="object")
    backend.assert_series_equal(result, expected, check_names=False)


@builtin_array
@pytest.mark.notimpl(
    ["dask", "datafusion", "impala", "mssql", "pandas", "polars", "mysql"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["sqlite"], raises=NotImplementedError, reason="Unsupported type: Array..."
)
@pytest.mark.notyet(
    ["bigquery"],
    raises=BadRequest,
    reason="BigQuery doesn't support arrays with null elements",
)
@pytest.mark.notyet(
    ["clickhouse"],
    raises=(AssertionError, TypeError),
    reason="clickhouse doesn't support nullable array types",
)
@pytest.mark.notyet(
    ["bigquery"],
    raises=(AssertionError, BadRequest),
    reason="bigquery doesn't support null elements in arrays",
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
def test_array_unique(backend, con, input, expected):
    t = ibis.memtable(input)
    expr = t.a.unique()
    result = con.execute(expr).map(set, na_action="ignore")
    expected = pd.Series(expected, dtype="object")
    backend.assert_series_equal(result, expected, check_names=False)


@builtin_array
@pytest.mark.notimpl(
    ["dask", "datafusion", "impala", "mssql", "pandas", "polars"],
    raises=com.OperationNotDefinedError,
)
def test_array_sort(backend, con):
    t = ibis.memtable({"a": [[3, 2], [], [42, 42], []]})
    expr = t.a.sort()
    result = con.execute(expr)
    expected = pd.Series([[2, 3], [], [42, 42], []], dtype="object")
    backend.assert_series_equal(result, expected, check_names=False)


@builtin_array
@pytest.mark.notimpl(
    ["dask", "datafusion", "impala", "mssql", "pandas", "polars"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notyet(
    ["bigquery"],
    raises=BadRequest,
    reason="BigQuery doesn't support arrays with null elements",
)
def test_array_union(con):
    t = ibis.memtable({"a": [[3, 2], [], []], "b": [[1, 3], [None], [5]]})
    expr = t.a.union(t.b)
    result = con.execute(expr).map(set, na_action="ignore")
    expected = pd.Series([{1, 2, 3}, {None}, {5}], dtype="object")
    assert len(result) == len(expected)

    for i, (lhs, rhs) in enumerate(zip(result, expected)):
        assert lhs == rhs, f"row {i:d} differs"


@pytest.mark.notimpl(
    ["dask", "datafusion", "impala", "mssql", "pandas", "polars", "mysql", "flink"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["sqlite"], raises=NotImplementedError, reason="Unsupported type: Array..."
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
                    raises=BadRequest,
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

    for i, (lhs, rhs) in enumerate(zip(result, expected)):
        assert lhs == rhs, f"row {i:d} differs"


@builtin_array
@pytest.mark.notimpl(
    ["clickhouse"],
    raises=ClickhouseDatabaseError,
    reason="ClickHouse won't accept dicts for struct type values",
)
@pytest.mark.notimpl(["postgres"], raises=sa.exc.ProgrammingError)
@pytest.mark.notimpl(["datafusion"], raises=com.OperationNotDefinedError)
def test_unnest_struct(con):
    data = {"value": [[{"a": 1}, {"a": 2}], [{"a": 3}, {"a": 4}]]}
    t = ibis.memtable(data, schema=ibis.schema({"value": "!array<!struct<a: !int>>"}))
    expr = t.value.unnest()

    result = con.execute(expr)

    expected = pd.DataFrame(data).explode("value").iloc[:, 0].reset_index(drop=True)
    tm.assert_series_equal(result, expected)


@builtin_array
@pytest.mark.never(
    ["impala", "mssql"],
    raises=com.OperationNotDefinedError,
    reason="no array support",
)
@pytest.mark.notimpl(
    ["dask", "datafusion", "druid", "oracle", "pandas", "polars", "postgres"],
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
    raises=ClickhouseDatabaseError,
    reason="https://github.com/ClickHouse/ClickHouse/issues/41112",
)
@pytest.mark.notimpl(["postgres"], raises=sa.exc.ProgrammingError)
@pytest.mark.notimpl(["datafusion"], raises=com.OperationNotDefinedError)
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
    ["bigquery"],
    reason="BigQuery doesn't support arrays of arrays",
    raises=TypeError,
)
@pytest.mark.notyet(
    ["postgres"],
    reason="Postgres doesn't truly support arrays of arrays",
    raises=com.OperationNotDefinedError,
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
                    raises=ClickhouseDatabaseError,
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
                    raises=ClickhouseDatabaseError,
                )
            ],
        ),
    ],
)
@pytest.mark.notimpl(["flink"], raises=com.OperationNotDefinedError)
@pytest.mark.notyet(["datafusion"], raises=com.OperationNotDefinedError)
def test_array_flatten(backend, flatten_data, column, expected):
    data = flatten_data[column]
    t = ibis.memtable(
        {column: data["data"]}, schema=ibis.schema({column: data["type"]})
    )
    expr = t[column].flatten()
    result = backend.connection.execute(expr)
    backend.assert_series_equal(result, expected, check_names=False)


polars_overflow = pytest.mark.notyet(
    ["polars"],
    reason="but polars overflows allocation with some inputs",
    raises=BaseException,
)


@pytest.mark.notyet(
    ["datafusion"],
    reason="range isn't implemented upstream",
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(["flink", "pandas", "dask"], raises=com.OperationNotDefinedError)
@pytest.mark.parametrize("n", [param(-2, marks=[polars_overflow]), 0, 2])
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
@pytest.mark.notimpl(
    ["polars", "flink", "pandas", "dask"], raises=com.OperationNotDefinedError
)
def test_range_single_argument_unnest(con, n):
    expr = ibis.range(n).unnest()
    result = con.execute(expr)
    tm.assert_series_equal(
        result,
        pd.Series(list(range(n)), dtype=result.dtype, name=expr.get_name()),
        check_index=False,
    )


@pytest.mark.parametrize(
    "step",
    [
        param(
            -2,
            marks=[
                pytest.mark.notyet(
                    ["polars"],
                    reason="panic upstream",
                    raises=PolarsInvalidOperationError,
                )
            ],
        ),
        param(
            -1,
            marks=[
                pytest.mark.notyet(
                    ["polars"],
                    reason="panic upstream",
                    raises=PolarsInvalidOperationError,
                )
            ],
        ),
        1,
        2,
    ],
)
@pytest.mark.parametrize(
    ("start", "stop"),
    [
        param(-7, -7),
        param(-7, 0),
        param(-7, 7),
        param(0, -7, marks=[polars_overflow]),
        param(0, 0),
        param(0, 7),
        param(7, -7, marks=[polars_overflow]),
        param(7, 0, marks=[polars_overflow]),
        param(7, 7),
    ],
)
@pytest.mark.notyet(
    ["datafusion"],
    reason="range and unnest aren't implemented upstream",
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(["flink", "pandas", "dask"], raises=com.OperationNotDefinedError)
def test_range_start_stop_step(con, start, stop, step):
    expr = ibis.range(start, stop, step)
    result = con.execute(expr)
    assert list(result) == list(range(start, stop, step))


@pytest.mark.parametrize("stop", [-7, 0, 7])
@pytest.mark.parametrize("start", [-7, 0, 7])
@pytest.mark.notyet(
    ["clickhouse"], raises=ClickhouseDatabaseError, reason="not supported upstream"
)
@pytest.mark.notyet(
    ["datafusion"], raises=com.OperationNotDefinedError, reason="not supported upstream"
)
@pytest.mark.notimpl(["flink", "pandas", "dask"], raises=com.OperationNotDefinedError)
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
