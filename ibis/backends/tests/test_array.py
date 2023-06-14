from __future__ import annotations

import contextlib
import os

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
import sqlalchemy as sa
import toolz
from packaging.version import parse as parse_version

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.types as ir

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    from polars import ComputeError as PolarsComputeError

except ImportError:
    PolarsComputeError = None

try:
    from clickhouse_driver.dbapi.errors import OperationalError
except ImportError:
    OperationalError = None

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


@pytest.mark.notimpl(["datafusion"], raises=com.OperationNotDefinedError)
def test_array_column(backend, alltypes, df):
    expr = ibis.array([alltypes['double_col'], alltypes['double_col']])
    assert isinstance(expr, ir.ArrayColumn)

    result = expr.execute()
    expected = df.apply(
        lambda row: [row['double_col'], row['double_col']],
        axis=1,
    )
    backend.assert_series_equal(result, expected, check_names=False)


ARRAY_BACKEND_TYPES = {
    'clickhouse': "Array(Float64)",
    'snowflake': "ARRAY",
    'trino': 'array(double)',
    "bigquery": "ARRAY",
    "duckdb": "DOUBLE[]",
    "postgres": "numeric[]",
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


@pytest.mark.notimpl(
    ["polars", "datafusion", "snowflake"], raises=com.OperationNotDefinedError
)
def test_array_repeat(con):
    expr = ibis.array([1.0, 2.0]) * 2

    result = con.execute(expr.name("tmp"))
    expected = np.array([1.0, 2.0, 1.0, 2.0])

    assert np.array_equal(result, expected)


# Issues #2370
@pytest.mark.notimpl(["datafusion"], raises=com.OperationNotDefinedError)
def test_array_concat(con):
    left = ibis.literal([1, 2, 3])
    right = ibis.literal([2, 1])
    expr = left + right
    result = con.execute(expr.name("tmp"))
    expected = np.array([1, 2, 3, 2, 1])
    assert np.array_equal(result, expected)


@pytest.mark.notimpl(["datafusion"], raises=com.OperationNotDefinedError)
def test_array_radd_concat(con):
    left = [1]
    right = ibis.literal([2])
    expr = left + right
    result = con.execute(expr.name("tmp"))
    expected = np.array([1, 2])

    assert np.array_equal(result, expected)


@pytest.mark.notimpl(["datafusion"], raises=com.OperationNotDefinedError)
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


duckdb_0_4_0 = pytest.mark.xfail(
    (
        # nixpkgs is patched to include the fix, so we pass these tests
        # inside the nix-shell or when they run under `nix build`
        (not any(key.startswith("NIX_") for key in os.environ))
        and (
            parse_version(getattr(duckdb, "__version__", "0.0.0"))
            == parse_version("0.4.0")
        )
    ),
    reason="DuckDB array support is broken in 0.4.0 without nix",
)


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
    pytest.mark.notimpl(["datafusion"], raises=Exception),
    duckdb_0_4_0,
)

unnest = toolz.compose(
    builtin_array,
    pytest.mark.notyet(
        ["bigquery"],
        reason="doesn't support unnest in SELECT position",
        raises=com.OperationNotDefinedError,
    ),
)


@builtin_array
@pytest.mark.never(
    ["clickhouse", "duckdb", "pandas", "pyspark", "snowflake", "polars", "trino"],
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
    ["duckdb", "pandas", "postgres", "pyspark", "snowflake", "polars", "trino"],
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


@unnest
@pytest.mark.notimpl(["dask"], raises=ValueError)
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


@unnest
@pytest.mark.notimpl("polars", raises=PolarsComputeError, reason="Series shape: (6,)")
@pytest.mark.notimpl("dask", raises=com.OperationNotDefinedError)
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


@unnest
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
@pytest.mark.notimpl("polars", raises=PolarsComputeError, reason="Series shape: (6,)")
@pytest.mark.notimpl(["dask"], raises=ValueError)
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


@unnest
@pytest.mark.notimpl("polars", raises=PolarsComputeError, reason="Series shape: (6,)")
@pytest.mark.notimpl(["dask"], raises=ValueError)
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


@unnest
@pytest.mark.notimpl("polars", raises=AssertionError, reason="Series are different")
@pytest.mark.notimpl("dask", raises=ValueError)
def test_unnest_default_name(backend):
    array_types = backend.array_types
    df = array_types.execute()
    expr = (
        array_types.x.cast("!array<int64>") + ibis.array([1], type="!array<int64>")
    ).unnest()
    assert expr.get_name().startswith("ArrayConcat(")

    result = expr.name("x").execute()
    expected = df.x.map(lambda x: x + [1]).explode("x")
    tm.assert_series_equal(result, expected, check_dtype=False)


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
    ["datafusion"], raises=Exception, reason="array_types table isn't defined"
)
@pytest.mark.notimpl(["dask"], raises=com.OperationNotDefinedError)
def test_array_slice(backend, start, stop):
    array_types = backend.array_types
    expr = array_types.select(sliced=array_types.y[start:stop])
    result = expr.execute()
    expected = pd.DataFrame(
        {'sliced': array_types.y.execute().map(lambda x: x[start:stop])}
    )
    tm.assert_frame_equal(result, expected)


@builtin_array
@pytest.mark.notimpl(
    [
        "bigquery",
        "datafusion",
        "impala",
        "mssql",
        "polars",
        "postgres",
        "snowflake",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(
    ["dask", "pandas"],
    raises=com.OperationNotDefinedError,
    reason="Operation 'ArrayMap' is not implemented for this backend",
)
def test_array_map(backend, con):
    t = ibis.memtable(
        {"a": [[1, None, 2], [4]]}, schema=ibis.schema(dict(a="!array<int8>"))
    )
    expr = t.select(a=t.a.map(lambda x: x + 1))
    result = con.execute(expr)
    expected = pd.DataFrame({"a": [[2, None, 3], [5]]})
    backend.assert_frame_equal(result, expected)


@builtin_array
@pytest.mark.notimpl(
    [
        "bigquery",
        "dask",
        "datafusion",
        "impala",
        "mssql",
        "pandas",
        "polars",
        "postgres",
        "snowflake",
    ],
    raises=com.OperationNotDefinedError,
)
def test_array_filter(backend, con):
    t = ibis.memtable(
        {"a": [[1, None, 2], [4]]}, schema=ibis.schema(dict(a="!array<int8>"))
    )
    expr = t.select(a=t.a.filter(lambda x: x > 1))
    result = con.execute(expr)
    expected = pd.DataFrame({"a": [[2], [4]]})
    backend.assert_frame_equal(result, expected)


@builtin_array
@pytest.mark.notimpl(
    ["bigquery", "datafusion", "mssql", "pandas", "polars", "postgres"],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.notimpl(["datafusion"], raises=Exception)
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
    [
        "bigquery",
        "dask",
        "datafusion",
        "impala",
        "mssql",
        "pandas",
        "polars",
        "postgres",
    ],
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
    [
        "bigquery",
        "dask",
        "datafusion",
        "impala",
        "mssql",
        "pandas",
        "polars",
        "postgres",
        "snowflake",
    ],
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
    [
        "bigquery",
        "dask",
        "datafusion",
        "impala",
        "mssql",
        "pandas",
        "polars",
        "postgres",
    ],
    raises=com.OperationNotDefinedError,
)
def test_array_unique(backend, con):
    t = ibis.memtable({"a": [[1, 3, 3], [], [42, 42], []]})
    expr = t.a.unique()
    result = con.execute(expr).map(set, na_action="ignore")
    expected = pd.Series([{3, 1}, set(), {42}, set()], dtype="object")
    backend.assert_series_equal(result, expected, check_names=False)


@builtin_array
@pytest.mark.notimpl(
    [
        "bigquery",
        "dask",
        "datafusion",
        "impala",
        "mssql",
        "pandas",
        "polars",
        "postgres",
        "snowflake",
    ],
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
    [
        "bigquery",
        "dask",
        "datafusion",
        "impala",
        "mssql",
        "pandas",
        "polars",
        "postgres",
    ],
    raises=com.OperationNotDefinedError,
)
@pytest.mark.broken(
    ["snowflake", "trino", "pyspark"],
    raises=AssertionError,
    reason="array_distinct([NULL]) seems to differ from other backends",
)
def test_array_union(con):
    t = ibis.memtable({"a": [[3, 2], [], []], "b": [[1, 3], [None], [5]]})
    expr = t.a.union(t.b)
    result = con.execute(expr).map(set, na_action="ignore")
    expected = pd.Series([{1, 2, 3}, set(), {5}], dtype="object")
    assert len(result) == len(expected)

    for i, (lhs, rhs) in enumerate(zip(result, expected)):
        assert lhs == rhs, f"row {i:d} differs"


@unnest
@builtin_array
@pytest.mark.notimpl(
    ["clickhouse"],
    raises=OperationalError,
    reason="ClickHouse won't accept dicts for struct type values",
)
@pytest.mark.notimpl(["postgres"], raises=sa.exc.ProgrammingError)
def test_unnest_struct(con):
    data = {"value": [[{'a': 1}, {'a': 2}], [{'a': 3}, {'a': 4}]]}
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
@pytest.mark.notyet(["bigquery"], raises=com.OperationNotDefinedError)
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
