from __future__ import annotations

import copy
import functools
import inspect
import itertools
import os
import string

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from packaging.version import parse as vparse

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.base import _get_backend_names
from ibis.backends.pandas.udf import udf

pytestmark = pytest.mark.benchmark


def make_t():
    return ibis.table(
        [
            ("_timestamp", "int32"),
            ("dim1", "int32"),
            ("dim2", "int32"),
            ("valid_seconds", "int32"),
            ("meas1", "int32"),
            ("meas2", "int32"),
            ("year", "int32"),
            ("month", "int32"),
            ("day", "int32"),
            ("hour", "int32"),
            ("minute", "int32"),
        ],
        name="t",
    )


@pytest.fixture(scope="module")
def t():
    return make_t()


def make_base(t):
    return t[
        (
            (t.year > 2016)
            | ((t.year == 2016) & (t.month > 6))
            | ((t.year == 2016) & (t.month == 6) & (t.day > 6))
            | ((t.year == 2016) & (t.month == 6) & (t.day == 6) & (t.hour > 6))
            | (
                (t.year == 2016)
                & (t.month == 6)
                & (t.day == 6)
                & (t.hour == 6)
                & (t.minute >= 5)
            )
        )
        & (
            (t.year < 2016)
            | ((t.year == 2016) & (t.month < 6))
            | ((t.year == 2016) & (t.month == 6) & (t.day < 6))
            | ((t.year == 2016) & (t.month == 6) & (t.day == 6) & (t.hour < 6))
            | (
                (t.year == 2016)
                & (t.month == 6)
                & (t.day == 6)
                & (t.hour == 6)
                & (t.minute <= 5)
            )
        )
    ]


@pytest.fixture(scope="module")
def base(t):
    return make_base(t)


def make_large_expr(base):
    src_table = base
    src_table = src_table.mutate(
        _timestamp=(src_table["_timestamp"] - src_table["_timestamp"] % 3600)
        .cast("int32")
        .name("_timestamp"),
        valid_seconds=300,
    )

    aggs = []
    for meas in ["meas1", "meas2"]:
        aggs.append(src_table[meas].sum().cast("float").name(meas))
    src_table = src_table.aggregate(
        aggs, by=["_timestamp", "dim1", "dim2", "valid_seconds"]
    )

    part_keys = ["year", "month", "day", "hour", "minute"]
    ts_col = src_table["_timestamp"].cast("timestamp")
    new_cols = {}
    for part_key in part_keys:
        part_col = getattr(ts_col, part_key)()
        new_cols[part_key] = part_col
    src_table = src_table.mutate(**new_cols)
    return src_table[
        [
            "_timestamp",
            "dim1",
            "dim2",
            "meas1",
            "meas2",
            "year",
            "month",
            "day",
            "hour",
            "minute",
        ]
    ]


@pytest.fixture(scope="module")
def large_expr(base):
    return make_large_expr(base)


@pytest.mark.benchmark(group="construction")
@pytest.mark.parametrize(
    "construction_fn",
    [
        pytest.param(lambda *_: make_t(), id="small"),
        pytest.param(lambda t, *_: make_base(t), id="medium"),
        pytest.param(lambda _, base: make_large_expr(base), id="large"),
    ],
)
def test_construction(benchmark, construction_fn, t, base):
    benchmark(construction_fn, t, base)


@pytest.mark.benchmark(group="builtins")
@pytest.mark.parametrize(
    "expr_fn",
    [
        pytest.param(lambda t, _base, _large_expr: t, id="small"),
        pytest.param(lambda _t, base, _large_expr: base, id="medium"),
        pytest.param(lambda _t, _base, large_expr: large_expr, id="large"),
    ],
)
@pytest.mark.parametrize("builtin", [hash, str])
def test_builtins(benchmark, expr_fn, builtin, t, base, large_expr):
    expr = expr_fn(t, base, large_expr)
    benchmark(builtin, expr)


_backends = set(_get_backend_names())
# compile is a no-op
_backends.remove("pandas")

_XFAIL_COMPILE_BACKENDS = {"dask", "pyspark", "polars"}


@pytest.mark.benchmark(group="compilation")
@pytest.mark.parametrize(
    "module",
    [
        pytest.param(
            mod,
            marks=pytest.mark.xfail(
                condition=mod in _XFAIL_COMPILE_BACKENDS,
                reason=f"{mod} backend doesn't support compiling UnboundTable",
            ),
        )
        for mod in _backends
    ],
)
@pytest.mark.parametrize(
    "expr_fn",
    [
        pytest.param(lambda t, _base, _large_expr: t, id="small"),
        pytest.param(lambda _t, base, _large_expr: base, id="medium"),
        pytest.param(lambda _t, _base, large_expr: large_expr, id="large"),
    ],
)
def test_compile(benchmark, module, expr_fn, t, base, large_expr):
    try:
        mod = getattr(ibis, module)
    except (AttributeError, ImportError) as e:
        pytest.skip(str(e))
    else:
        expr = expr_fn(t, base, large_expr)
        try:
            benchmark(mod.compile, expr)
        except (sa.exc.NoSuchModuleError, ImportError) as e:  # delayed imports
            pytest.skip(str(e))


@pytest.fixture(scope="module")
def pt():
    n = 60_000
    data = pd.DataFrame(
        {
            "key": np.random.choice(16000, size=n),
            "low_card_key": np.random.choice(30, size=n),
            "value": np.random.rand(n),
            "timestamps": pd.date_range(
                start="2023-05-05 16:37:57", periods=n, freq="s"
            ).values,
            "timestamp_strings": pd.date_range(
                start="2023-05-05 16:37:39", periods=n, freq="s"
            ).values.astype(str),
            "repeated_timestamps": pd.date_range(start="2018-09-01", periods=30).repeat(
                int(n / 30)
            ),
        }
    )

    return ibis.pandas.connect(dict(df=data)).table("df")


def high_card_group_by(t):
    return t.group_by(t.key).aggregate(avg_value=t.value.mean())


def cast_to_dates(t):
    return t.timestamps.cast(dt.date)


def cast_to_dates_from_strings(t):
    return t.timestamp_strings.cast(dt.date)


def multikey_group_by_with_mutate(t):
    return (
        t.mutate(dates=t.timestamps.cast("date"))
        .group_by(["low_card_key", "dates"])
        .aggregate(avg_value=lambda t: t.value.mean())
    )


def simple_sort(t):
    return t.order_by([t.key])


def simple_sort_projection(t):
    return t[["key", "value"]].order_by(["key"])


def multikey_sort(t):
    return t.order_by(["low_card_key", "key"])


def multikey_sort_projection(t):
    return t[["low_card_key", "key", "value"]].order_by(["low_card_key", "key"])


def low_card_rolling_window(t):
    return ibis.trailing_range_window(
        ibis.interval(days=2),
        order_by=t.repeated_timestamps,
        group_by=t.low_card_key,
    )


def low_card_grouped_rolling(t):
    return t.value.mean().over(low_card_rolling_window(t))


def high_card_rolling_window(t):
    return ibis.trailing_range_window(
        ibis.interval(days=2),
        order_by=t.repeated_timestamps,
        group_by=t.key,
    )


def high_card_grouped_rolling(t):
    return t.value.mean().over(high_card_rolling_window(t))


@udf.reduction(["double"], "double")
def my_mean(series):
    return series.mean()


def low_card_grouped_rolling_udf_mean(t):
    return my_mean(t.value).over(low_card_rolling_window(t))


def high_card_grouped_rolling_udf_mean(t):
    return my_mean(t.value).over(high_card_rolling_window(t))


@udf.analytic(["double"], "double")
def my_zscore(series):
    return (series - series.mean()) / series.std()


def low_card_window(t):
    return ibis.window(group_by=t.low_card_key)


def high_card_window(t):
    return ibis.window(group_by=t.key)


def low_card_window_analytics_udf(t):
    return my_zscore(t.value).over(low_card_window(t))


def high_card_window_analytics_udf(t):
    return my_zscore(t.value).over(high_card_window(t))


@udf.reduction(["double", "double"], "double")
def my_wm(v, w):
    return np.average(v, weights=w)


def low_card_grouped_rolling_udf_wm(t):
    return my_wm(t.value, t.value).over(low_card_rolling_window(t))


def high_card_grouped_rolling_udf_wm(t):
    return my_wm(t.value, t.value).over(low_card_rolling_window(t))


broken_pandas_grouped_rolling = pytest.mark.xfail(
    condition=vparse("1.4") <= vparse(pd.__version__) < vparse("1.4.2"),
    raises=ValueError,
    reason="https://github.com/pandas-dev/pandas/pull/44068",
)


@pytest.mark.benchmark(group="execution")
@pytest.mark.parametrize(
    "expression_fn",
    [
        pytest.param(high_card_group_by, id="high_card_group_by"),
        pytest.param(cast_to_dates, id="cast_to_dates"),
        pytest.param(cast_to_dates_from_strings, id="cast_to_dates_from_strings"),
        pytest.param(multikey_group_by_with_mutate, id="multikey_group_by_with_mutate"),
        pytest.param(simple_sort, id="simple_sort"),
        pytest.param(simple_sort_projection, id="simple_sort_projection"),
        pytest.param(multikey_sort, id="multikey_sort"),
        pytest.param(multikey_sort_projection, id="multikey_sort_projection"),
        pytest.param(
            low_card_grouped_rolling,
            id="low_card_grouped_rolling",
            marks=[broken_pandas_grouped_rolling],
        ),
        pytest.param(
            high_card_grouped_rolling,
            id="high_card_grouped_rolling",
            marks=[broken_pandas_grouped_rolling],
        ),
        pytest.param(
            low_card_grouped_rolling_udf_mean,
            id="low_card_grouped_rolling_udf_mean",
            marks=[broken_pandas_grouped_rolling],
        ),
        pytest.param(
            high_card_grouped_rolling_udf_mean,
            id="high_card_grouped_rolling_udf_mean",
            marks=[broken_pandas_grouped_rolling],
        ),
        pytest.param(low_card_window_analytics_udf, id="low_card_window_analytics_udf"),
        pytest.param(
            high_card_window_analytics_udf, id="high_card_window_analytics_udf"
        ),
        pytest.param(
            low_card_grouped_rolling_udf_wm,
            id="low_card_grouped_rolling_udf_wm",
            marks=[broken_pandas_grouped_rolling],
        ),
        pytest.param(
            high_card_grouped_rolling_udf_wm,
            id="high_card_grouped_rolling_udf_wm",
            marks=[broken_pandas_grouped_rolling],
        ),
    ],
)
def test_execute(benchmark, expression_fn, pt):
    expr = expression_fn(pt)
    benchmark(expr.execute)


@pytest.fixture(scope="module")
def part():
    return ibis.table(
        dict(
            p_partkey="int64",
            p_size="int64",
            p_type="string",
            p_mfgr="string",
        ),
        name="part",
    )


@pytest.fixture(scope="module")
def supplier():
    return ibis.table(
        dict(
            s_suppkey="int64",
            s_nationkey="int64",
            s_name="string",
            s_acctbal="decimal(15, 3)",
            s_address="string",
            s_phone="string",
            s_comment="string",
        ),
        name="supplier",
    )


@pytest.fixture(scope="module")
def partsupp():
    return ibis.table(
        dict(
            ps_partkey="int64",
            ps_suppkey="int64",
            ps_supplycost="decimal(15, 3)",
        ),
        name="partsupp",
    )


@pytest.fixture(scope="module")
def nation():
    return ibis.table(
        dict(n_nationkey="int64", n_regionkey="int64", n_name="string"),
        name="nation",
    )


@pytest.fixture(scope="module")
def region():
    return ibis.table(dict(r_regionkey="int64", r_name="string"), name="region")


@pytest.fixture(scope="module")
def tpc_h02(part, supplier, partsupp, nation, region):
    REGION = "EUROPE"
    SIZE = 25
    TYPE = "BRASS"

    expr = (
        part.join(partsupp, part.p_partkey == partsupp.ps_partkey)
        .join(supplier, supplier.s_suppkey == partsupp.ps_suppkey)
        .join(nation, supplier.s_nationkey == nation.n_nationkey)
        .join(region, nation.n_regionkey == region.r_regionkey)
    )

    subexpr = (
        partsupp.join(supplier, supplier.s_suppkey == partsupp.ps_suppkey)
        .join(nation, supplier.s_nationkey == nation.n_nationkey)
        .join(region, nation.n_regionkey == region.r_regionkey)
    )

    subexpr = subexpr[
        (subexpr.r_name == REGION) & (expr.p_partkey == subexpr.ps_partkey)
    ]

    filters = [
        expr.p_size == SIZE,
        expr.p_type.like(f"%{TYPE}"),
        expr.r_name == REGION,
        expr.ps_supplycost == subexpr.ps_supplycost.min(),
    ]
    q = expr.filter(filters)

    q = q.select(
        [
            q.s_acctbal,
            q.s_name,
            q.n_name,
            q.p_partkey,
            q.p_mfgr,
            q.s_address,
            q.s_phone,
            q.s_comment,
        ]
    )

    return q.order_by(
        [
            ibis.desc(q.s_acctbal),
            q.n_name,
            q.s_name,
            q.p_partkey,
        ]
    ).limit(100)


@pytest.mark.benchmark(group="repr")
def test_repr_tpc_h02(benchmark, tpc_h02):
    benchmark(repr, tpc_h02)


@pytest.mark.benchmark(group="repr")
def test_repr_huge_union(benchmark):
    n = 10
    raw_types = [
        "int64",
        "float64",
        "string",
        "array<struct<a: array<string>, b: map<string, array<int64>>>>",
    ]
    tables = [
        ibis.table(
            list(zip(string.ascii_letters, itertools.cycle(raw_types))),
            name=f"t{i:d}",
        )
        for i in range(n)
    ]
    expr = functools.reduce(ir.Table.union, tables)
    benchmark(repr, expr)


@pytest.mark.benchmark(group="node_args")
def test_op_argnames(benchmark):
    t = ibis.table([("a", "int64")])
    expr = t[["a"]]
    benchmark(lambda op: op.argnames, expr.op())


@pytest.mark.benchmark(group="node_args")
def test_op_args(benchmark):
    t = ibis.table([("a", "int64")])
    expr = t[["a"]]
    benchmark(lambda op: op.args, expr.op())


@pytest.mark.benchmark(group="datatype")
def test_complex_datatype_parse(benchmark):
    type_str = "array<struct<a: array<string>, b: map<string, array<int64>>>>"
    expected = dt.Array(
        dt.Struct(dict(a=dt.Array(dt.string), b=dt.Map(dt.string, dt.Array(dt.int64))))
    )
    assert dt.parse(type_str) == expected
    benchmark(dt.parse, type_str)


@pytest.mark.benchmark(group="datatype")
@pytest.mark.parametrize("func", [str, hash])
def test_complex_datatype_builtins(benchmark, func):
    datatype = dt.Array(
        dt.Struct(dict(a=dt.Array(dt.string), b=dt.Map(dt.string, dt.Array(dt.int64))))
    )
    benchmark(func, datatype)


@pytest.mark.benchmark(group="equality")
def test_large_expr_equals(benchmark, tpc_h02):
    benchmark(ir.Expr.equals, tpc_h02, copy.deepcopy(tpc_h02))


@pytest.mark.benchmark(group="datatype")
@pytest.mark.parametrize(
    "dtypes",
    [
        pytest.param(
            [
                obj
                for _, obj in inspect.getmembers(
                    dt,
                    lambda obj: isinstance(obj, dt.DataType),
                )
            ],
            id="singletons",
        ),
        pytest.param(
            dt.Array(
                dt.Struct(
                    dict(
                        a=dt.Array(dt.string),
                        b=dt.Map(dt.string, dt.Array(dt.int64)),
                    )
                )
            ),
            id="complex",
        ),
    ],
)
def test_eq_datatypes(benchmark, dtypes):
    def eq(a, b):
        assert a == b

    benchmark(eq, dtypes, copy.deepcopy(dtypes))


def multiple_joins(table, num_joins):
    for _ in range(num_joins):
        table = table.mutate(dummy=ibis.literal(""))
        table = table.left_join(table, ["dummy"])[[table]]


@pytest.mark.parametrize("num_joins", [1, 10])
@pytest.mark.parametrize("num_columns", [1, 10, 100])
def test_multiple_joins(benchmark, num_joins, num_columns):
    table = ibis.table(
        {f"col_{i:d}": "string" for i in range(num_columns)},
        name="t",
    )
    benchmark(multiple_joins, table, num_joins)


@pytest.fixture
def customers():
    return ibis.table(
        dict(
            customerid="int32",
            name="string",
            address="string",
            citystatezip="string",
            birthdate="date",
            phone="string",
            timezone="string",
            lat="float64",
            long="float64",
        ),
        name="customers",
    )


@pytest.fixture
def orders():
    return ibis.table(
        dict(
            orderid="int32",
            customerid="int32",
            ordered="timestamp",
            shipped="timestamp",
            items="string",
            total="float64",
        ),
        name="orders",
    )


@pytest.fixture
def orders_items():
    return ibis.table(
        dict(orderid="int32", sku="string", qty="int32", unit_price="float64"),
        name="orders_items",
    )


@pytest.fixture
def products():
    return ibis.table(
        dict(
            sku="string",
            desc="string",
            weight_kg="float64",
            cost="float64",
            dims_cm="string",
        ),
        name="products",
    )


@pytest.mark.benchmark(group="compilation")
@pytest.mark.parametrize(
    "module",
    [
        pytest.param(
            mod,
            marks=pytest.mark.xfail(
                condition=mod in _XFAIL_COMPILE_BACKENDS,
                reason=f"{mod} backend doesn't support compiling UnboundTable",
            ),
        )
        for mod in _backends
    ],
)
def test_compile_with_drops(
    benchmark, module, customers, orders, orders_items, products
):
    expr = (
        customers.join(orders, "customerid")
        .join(orders_items, "orderid")
        .join(products, "sku")
        .drop("customerid", "qty", "total", "items")
        .drop("dims_cm", "cost")
        .mutate(o_date=lambda t: t.shipped.date())
        .filter(lambda t: t.ordered == t.shipped)
    )

    try:
        mod = getattr(ibis, module)
    except (AttributeError, ImportError) as e:
        pytest.skip(str(e))
    else:
        try:
            benchmark(mod.compile, expr)
        except sa.exc.NoSuchModuleError as e:
            pytest.skip(str(e))


def test_repr_join(benchmark, customers, orders, orders_items, products):
    expr = (
        customers.join(orders, "customerid")
        .join(orders_items, "orderid")
        .join(products, "sku")
        .drop("customerid", "qty", "total", "items")
    )
    op = expr.op()
    benchmark(repr, op)


@pytest.mark.parametrize("overwrite", [True, False], ids=["overwrite", "no_overwrite"])
def test_insert_duckdb(benchmark, overwrite, tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("duckdb_engine")

    n_rows = int(1e4)
    table_name = "t"
    schema = ibis.schema(dict(a="int64", b="int64", c="int64"))
    t = ibis.memtable(dict.fromkeys(list("abc"), range(n_rows)), schema=schema)

    con = ibis.duckdb.connect(tmp_path / "test_insert.ddb")
    con.create_table(table_name, schema=schema)
    benchmark(con.insert, table_name, t, overwrite=overwrite)


def test_snowflake_medium_sized_to_pandas(benchmark):
    pytest.importorskip("snowflake.connector")
    pytest.importorskip("snowflake.sqlalchemy")

    if (url := os.environ.get("SNOWFLAKE_URL")) is None:
        pytest.skip("SNOWFLAKE_URL environment variable not set")

    con = ibis.connect(url)

    # LINEITEM at scale factor 1 is around 6MM rows, but we limit to 1,000,000
    # to make the benchmark fast enough for development, yet large enough to show a
    # difference if there's a performance hit
    lineitem = con.table("LINEITEM", schema="SNOWFLAKE_SAMPLE_DATA.TPCH_SF1").limit(
        1_000_000
    )

    benchmark.pedantic(lineitem.to_pandas, rounds=5, iterations=1, warmup_rounds=1)


def test_parse_many_duckdb_types(benchmark):
    parse = pytest.importorskip("ibis.backends.duckdb.datatypes").DuckDBType.from_string

    def parse_many(types):
        list(map(parse, types))

    types = ["VARCHAR", "INTEGER", "DOUBLE", "BIGINT"] * 1000
    benchmark(parse_many, types)


@pytest.fixture(scope="session")
def sql() -> str:
    return """
    SELECT t1.id as t1_id, x, t2.id as t2_id, y
    FROM t1 INNER JOIN t2
      ON t1.id = t2.id
    """


@pytest.fixture(scope="session")
def ddb(tmp_path_factory):
    duckdb = pytest.importorskip("duckdb")

    N = 20_000_000

    con = duckdb.connect()

    path = str(tmp_path_factory.mktemp("duckdb") / "data.ddb")
    sql = (
        lambda var, table, n=N: f"""
        CREATE TABLE {table} AS
        SELECT ROW_NUMBER() OVER () AS id, {var}
        FROM (
            SELECT {var}
            FROM RANGE({n}) _ ({var})
            ORDER BY RANDOM()
        )
        """
    )

    with duckdb.connect(path) as con:
        con.execute(sql("x", table="t1"))
        con.execute(sql("y", table="t2"))
    return path


def test_duckdb_to_pyarrow(benchmark, sql, ddb) -> None:
    # yes, we're benchmarking duckdb here, not ibis
    #
    # we do this to get a baseline for comparison
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(ddb, read_only=True)

    benchmark(lambda sql: con.sql(sql).to_arrow_table(), sql)


def test_ibis_duckdb_to_pyarrow(benchmark, sql, ddb) -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("duckdb_engine")

    con = ibis.duckdb.connect(ddb, read_only=True)

    expr = con.sql(sql)
    benchmark(expr.to_pyarrow)


@pytest.fixture
def diffs():
    return ibis.table(
        {
            "id": "int64",
            "validation_name": "string",
            "difference": "float64",
            "pct_difference": "float64",
            "pct_threshold": "float64",
            "validation_status": "string",
        },
        name="diffs",
    )


@pytest.fixture
def srcs():
    return ibis.table(
        {
            "id": "int64",
            "validation_name": "string",
            "validation_type": "string",
            "aggregation_type": "string",
            "table_name": "string",
            "column_name": "string",
            "primary_keys": "string",
            "num_random_rows": "string",
            "agg_value": "float64",
        },
        name="srcs",
    )


@pytest.fixture
def nrels():
    return 300


def make_big_union(t, nrels):
    return ibis.union(*[t] * nrels)


@pytest.fixture
def src(srcs, nrels):
    return make_big_union(srcs, nrels)


@pytest.fixture
def diff(diffs, nrels):
    return make_big_union(diffs, nrels)


def test_big_eq_expr(benchmark, src, diff):
    benchmark(ops.core.Node.equals, src.op(), diff.op())


def test_big_join_expr(benchmark, src, diff):
    benchmark(ir.Table.join, src, diff, ["validation_name"], how="outer")


def test_big_join_execute(benchmark, nrels):
    pytest.importorskip("duckdb")
    pytest.importorskip("duckdb_engine")

    con = ibis.duckdb.connect()

    # cache to avoid a request-per-union operand
    src = make_big_union(
        con.read_csv(
            "https://github.com/ibis-project/ibis/files/12580336/source_pivot.csv"
        )
        .rename(id="column0")
        .cache(),
        nrels,
    )

    diff = make_big_union(
        con.read_csv(
            "https://github.com/ibis-project/ibis/files/12580340/differences_pivot.csv"
        )
        .rename(id="column0")
        .cache(),
        nrels,
    )

    expr = src.join(diff, ["validation_name"], how="outer")
    t = benchmark.pedantic(expr.to_pyarrow, rounds=1, iterations=1, warmup_rounds=1)
    assert len(t)
