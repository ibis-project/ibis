import os

import numpy as np
import pandas.testing as tm
import pytest
import toolz
from packaging.version import parse as parse_version

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir

try:
    import duckdb
except ImportError:
    duckdb = None

pytestmark = [
    pytest.mark.never(["sqlite", "mysql"], reason="No array support"),
]


@pytest.mark.notimpl(["impala", "datafusion"])
def test_array_column(backend, alltypes, df):
    expr = ibis.array([alltypes['double_col'], alltypes['double_col']])
    assert isinstance(expr, ir.ArrayColumn)

    result = expr.execute()
    expected = df.apply(
        lambda row: [row['double_col'], row['double_col']],
        axis=1,
    )
    backend.assert_series_equal(result, expected, check_names=False)


@pytest.mark.notimpl(["impala"])
def test_array_scalar(con):
    expr = ibis.array([1.0, 2.0, 3.0])
    assert isinstance(expr, ir.ArrayScalar)

    result = con.execute(expr)
    expected = np.array([1.0, 2.0, 3.0])

    # This does not check whether `result` is an np.array or a list,
    # because it varies across backends and backend configurations
    assert np.array_equal(result, expected)


# Issues #2370
@pytest.mark.notimpl(["impala", "datafusion"])
def test_array_concat(con):
    left = ibis.literal([1, 2, 3])
    right = ibis.literal([2, 1])
    expr = left + right
    result = con.execute(expr)
    expected = np.array([1, 2, 3, 2, 1])

    # This does not check whether `result` is an np.array or a list,
    # because it varies across backends and backend configurations
    assert np.array_equal(result, expected)


@pytest.mark.notimpl(["impala", "datafusion"])
def test_array_length(con):
    expr = ibis.literal([1, 2, 3]).length()
    assert con.execute(expr) == 3


@pytest.mark.notimpl(["impala"])
def test_list_literal(con):
    arr = [1, 2, 3]
    expr = ibis.literal(arr)
    result = con.execute(expr)

    # This does not check whether `result` is an np.array or a list,
    # because it varies across backends and backend configurations
    assert np.array_equal(result, arr)


@pytest.mark.notimpl(["impala"])
def test_np_array_literal(con):
    arr = np.array([1, 2, 3])
    expr = ibis.literal(arr)
    result = con.execute(expr)

    # This does not check whether `result` is an np.array or a list,
    # because it varies across backends and backend configurations
    assert np.array_equal(result, arr)


duckdb_0_4_0 = pytest.mark.xfail(
    (
        # nixpkgs is patched to include the fix, so we pass these tests
        # inside the nix-shell or when they run under `nix build`
        (not any(key.startswith("NIX_") for key in os.environ.keys()))
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
        ["mysql", "sqlite"],
        reason="array types are unsupported",
    ),
    # someone just needs to implement these
    pytest.mark.notimpl(["datafusion", "dask"]),
    # unclear if thi will ever be supported
    pytest.mark.notyet(
        ["impala"],
        reason="impala doesn't support array types",
    ),
    duckdb_0_4_0,
)

unnest = toolz.compose(
    builtin_array,
    pytest.mark.notimpl(["pandas"]),
)


@builtin_array
@pytest.mark.never(
    ["clickhouse", "pandas", "pyspark"],
    reason="backend does not flatten array types",
)
def test_array_discovery_postgres_duckdb(con):
    t = con.table("array_types")
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
    ["duckdb", "pandas", "postgres", "pyspark"],
    reason="backend supports nullable nested types",
)
def test_array_discovery_clickhouse(con):
    t = con.table("array_types")
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
    ["clickhouse", "duckdb", "postgres"],
    reason="backend does not support nullable nested types",
)
def test_array_discovery_desired(con):
    t = con.table("array_types")
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


@unnest
def test_unnest_simple(con):
    array_types = con.table("array_types")
    expected = (
        array_types.execute()
        .x.explode()
        .reset_index(drop=True)
        .astype("float64")
        .rename("tmp")
    )
    expr = array_types.x.unnest()
    result = expr.execute().rename("tmp")
    tm.assert_series_equal(result, expected)


@unnest
def test_unnest_complex(con):
    array_types = con.table("array_types")
    df = array_types.execute()
    expr = (
        array_types.select(["grouper", "x"])
        .mutate(x=lambda t: t.x.unnest())
        .groupby("grouper")
        .aggregate(count_flat=lambda t: t.x.count())
        .sort_by("grouper")
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
)
@pytest.mark.never(
    "clickhouse",
    reason="clickhouse throws away nulls in groupArray",
)
def test_unnest_idempotent(con):
    array_types = con.table("array_types")
    df = array_types.execute()
    expr = (
        array_types.select(["scalar_column", array_types.x.unnest().name("x")])
        .group_by("scalar_column")
        .aggregate(x=lambda t: t.x.collect())
        .sort_by("scalar_column")
    )
    result = expr.execute()
    expected = df[["scalar_column", "x"]]
    tm.assert_frame_equal(result, expected)


@unnest
def test_unnest_no_nulls(con):
    array_types = con.table("array_types")
    df = array_types.execute()
    expr = (
        array_types.select(["scalar_column", array_types.x.unnest().name("y")])
        .filter(lambda t: t.y.notnull())
        .group_by("scalar_column")
        .aggregate(x=lambda t: t.y.collect())
        .sort_by("scalar_column")
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
def test_unnest_unnamed(con):
    array_types = con.table("array_types")
    df = array_types.execute()
    expr = (
        array_types.x.cast("!array<int64>")
        + ibis.array([1], type="!array<int64>")
    ).unnest()
    assert not expr.has_name()
    result = expr.name("x").execute()
    expected = df.x.map(lambda x: x + [1]).explode("x")
    tm.assert_series_equal(result, expected.astype("float64"))
