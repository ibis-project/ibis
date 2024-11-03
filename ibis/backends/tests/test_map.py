from __future__ import annotations

import pytest
from pytest import param

import ibis
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
from ibis.backends.tests.errors import PsycoPg2InternalError, Py4JJavaError

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
tm = pytest.importorskip("pandas.testing")
pa = pytest.importorskip("pyarrow")

pytestmark = [
    pytest.mark.never(
        ["sqlite", "mysql", "mssql"], reason="Unlikely to ever add map support"
    ),
    pytest.mark.notyet(
        ["bigquery", "impala"], reason="Backend doesn't yet implement map types"
    ),
    pytest.mark.notimpl(
        ["exasol", "polars", "druid", "oracle"],
        reason="Not yet implemented in ibis",
    ),
]

mark_notyet_postgres = pytest.mark.notyet(
    "postgres", reason="only supports string -> string"
)

mark_notyet_snowflake = pytest.mark.notyet(
    "snowflake", reason="map keys must be strings"
)

mark_notimpl_risingwave_hstore = pytest.mark.notimpl(
    ["risingwave"],
    reason="function hstore(character varying[], character varying[]) does not exist",
)

mark_notyet_datafusion = pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="only map and make_map are available"
)


@pytest.mark.notyet("clickhouse", reason="nested types can't be NULL")
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="function hstore(character varying[], character varying[]) does not exist",
)
@pytest.mark.parametrize(
    ("k", "v"),
    [
        param(["a", "b"], None, id="null_values"),
        param(None, ["c", "d"], id="null_keys"),
        param(None, None, id="null_both"),
    ],
)
@mark_notyet_datafusion
def test_map_nulls(con, k, v):
    k = ibis.literal(k, type="array<string>")
    v = ibis.literal(v, type="array<string>")
    m = ibis.map(k, v)
    assert con.execute(m) is None


@pytest.mark.notyet("clickhouse", reason="nested types can't be NULL")
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="function hstore(character varying[], character varying[]) does not exist",
)
@pytest.mark.parametrize(
    ("k", "v"),
    [
        param(None, ["c", "d"], id="null_keys"),
        param(None, None, id="null_both"),
    ],
)
@mark_notyet_datafusion
def test_map_keys_nulls(con, k, v):
    k = ibis.literal(k, type="array<string>")
    v = ibis.literal(v, type="array<string>")
    m = ibis.map(k, v)
    assert con.execute(m.keys()) is None


@pytest.mark.notyet("clickhouse", reason="nested types can't be NULL")
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="function hstore(character varying[], character varying[]) does not exist",
)
@pytest.mark.parametrize(
    "map",
    [
        param(
            ibis.map(
                ibis.literal(["a", "b"]), ibis.literal(None, type="array<string>")
            ),
            id="null_values",
        ),
        param(
            ibis.map(
                ibis.literal(None, type="array<string>"),
                ibis.literal(None, type="array<string>"),
            ),
            id="null_both",
        ),
        param(ibis.literal(None, type="map<string, string>"), id="null_map"),
    ],
)
@mark_notyet_datafusion
def test_map_values_nulls(con, map):
    assert con.execute(map.values()) is None


@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="function hstore(character varying[], character varying[]) does not exist",
)
@pytest.mark.parametrize(
    ("map", "key"),
    [
        param(
            ibis.map(
                ibis.literal(["a", "b"]), ibis.literal(["c", "d"], type="array<string>")
            ),
            ibis.literal(None, type="string"),
            marks=[
                pytest.mark.notimpl(
                    "flink",
                    raises=AssertionError,
                    reason="not yet implemented",
                    strict=False,
                ),
            ],
            id="non_null_map_null_key",
        ),
        param(
            ibis.map(
                ibis.literal(None, type="array<string>"),
                ibis.literal(None, type="array<string>"),
            ),
            "a",
            marks=[
                pytest.mark.notyet("clickhouse", reason="nested types can't be NULL")
            ],
            id="null_both_non_null_key",
        ),
        param(
            ibis.map(
                ibis.literal(None, type="array<string>"),
                ibis.literal(None, type="array<string>"),
            ),
            ibis.literal(None, type="string"),
            marks=[
                pytest.mark.notyet("clickhouse", reason="nested types can't be NULL"),
            ],
            id="null_both_null_key",
        ),
        param(
            ibis.literal(None, type="map<string, string>"),
            "a",
            marks=[
                pytest.mark.notyet("clickhouse", reason="nested types can't be NULL")
            ],
            id="null_map_non_null_key",
        ),
        param(
            ibis.literal(None, type="map<string, string>"),
            ibis.literal(None, type="string"),
            marks=[
                pytest.mark.notyet("clickhouse", reason="nested types can't be NULL")
            ],
            id="null_map_null_key",
        ),
    ],
)
@pytest.mark.parametrize("method", ["get", "contains"])
@mark_notyet_datafusion
def test_map_get_contains_nulls(con, map, key, method):
    expr = getattr(map, method)
    assert con.execute(expr(key)) is None


@pytest.mark.notyet("clickhouse", reason="nested types can't be NULL")
@pytest.mark.notimpl(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="function hstore(character varying[], character varying[]) does not exist",
)
@pytest.mark.parametrize(
    ("m1", "m2"),
    [
        param(
            ibis.literal(None, type="map<string, string>"),
            ibis.literal({"a": "b"}, type="map<string, string>"),
            id="null_and_non_null",
        ),
        param(
            ibis.literal({"a": "b"}, type="map<string, string>"),
            ibis.literal(None, type="map<string, string>"),
            id="non_null_and_null",
        ),
        param(
            ibis.literal(None, type="map<string, string>"),
            ibis.literal(None, type="map<string, string>"),
            id="null_and_null",
        ),
    ],
)
@mark_notyet_datafusion
def test_map_merge_nulls(con, m1, m2):
    concatted = m1 + m2
    assert con.execute(concatted) is None


@mark_notyet_datafusion
def test_map_table(backend):
    table = backend.map
    assert table.kv.type().is_map()
    assert not table.limit(1).execute().empty


@mark_notimpl_risingwave_hstore
@mark_notyet_datafusion
def test_column_map_values(backend):
    table = backend.map
    expr = table.select("idx", vals=table.kv.values()).order_by("idx")
    result = expr.execute().vals
    expected = pd.Series([[1, 2, 3], [4, 5, 6]], name="vals")
    backend.assert_series_equal(result, expected)


@mark_notyet_datafusion
@pytest.mark.notyet(
    ["databricks"], reason="says one thing, does something completely different"
)
def test_column_map_merge(backend):
    table = backend.map
    expr = table.select(
        "idx",
        merged=table.kv + ibis.map({"d": np.int64(1)}),
    ).order_by("idx")
    result = expr.execute().merged
    expected = pd.Series(
        [{"a": 1, "b": 2, "c": 3, "d": 1}, {"d": 1, "e": 5, "f": 6}], name="merged"
    )
    tm.assert_series_equal(result, expected)


@mark_notimpl_risingwave_hstore
@mark_notyet_datafusion
def test_literal_map_keys(con):
    mapping = ibis.literal({"1": "a", "2": "b"})
    expr = mapping.keys().name("tmp")

    result = con.execute(expr)
    # This does not check whether `result` is an np.array or a list,
    # because it varies across backends and backend configurations
    assert np.array_equal(result, ["1", "2"])


@mark_notimpl_risingwave_hstore
@mark_notyet_datafusion
def test_literal_map_values(con):
    mapping = ibis.literal({"1": "a", "2": "b"})
    expr = mapping.values().name("tmp")

    result = con.execute(expr)
    assert np.array_equal(result, ["a", "b"])


@mark_notimpl_risingwave_hstore
@mark_notyet_postgres
@mark_notyet_datafusion
def test_scalar_isin_literal_map_keys(con):
    mapping = ibis.literal({"a": 1, "b": 2})
    a = ibis.literal("a")
    c = ibis.literal("c")
    true = a.isin(mapping.keys())
    false = c.isin(mapping.keys())
    assert con.execute(true) == True  # noqa: E712
    assert con.execute(false) == False  # noqa: E712


@mark_notimpl_risingwave_hstore
@mark_notyet_postgres
@mark_notyet_datafusion
def test_map_scalar_contains_key_scalar(con):
    mapping = ibis.literal({"a": 1, "b": 2})
    a = ibis.literal("a")
    c = ibis.literal("c")
    true = mapping.contains(a)
    false = mapping.contains(c)
    assert con.execute(true) == True  # noqa: E712
    assert con.execute(false) == False  # noqa: E712


@mark_notimpl_risingwave_hstore
@mark_notyet_datafusion
def test_map_scalar_contains_key_column(backend, alltypes, df):
    value = {"1": "a", "3": "c"}
    mapping = ibis.literal(value)
    expr = mapping.contains(alltypes.string_col).name("tmp")
    result = expr.execute()
    expected = df.string_col.apply(lambda x: x in value).rename("tmp")
    backend.assert_series_equal(result, expected)


@mark_notimpl_risingwave_hstore
@mark_notyet_postgres
@mark_notyet_datafusion
def test_map_column_contains_key_scalar(backend, alltypes, df):
    expr = ibis.map(ibis.array([alltypes.string_col]), ibis.array([alltypes.int_col]))
    series = df.apply(lambda row: {row["string_col"]: row["int_col"]}, axis=1)

    result = expr.contains("1").name("tmp").execute()
    series = series.apply(lambda x: "1" in x).rename("tmp")

    backend.assert_series_equal(result, series)


@mark_notimpl_risingwave_hstore
@mark_notyet_postgres
@mark_notyet_datafusion
def test_map_column_contains_key_column(alltypes):
    map_expr = ibis.map(
        ibis.array([alltypes.string_col]), ibis.array([alltypes.int_col])
    )
    expr = map_expr.contains(alltypes.string_col).name("tmp")
    result = expr.execute()
    assert result.all()


@mark_notimpl_risingwave_hstore
@mark_notyet_postgres
@mark_notyet_datafusion
@pytest.mark.notyet(
    ["databricks"], reason="says one thing, does something completely different"
)
def test_literal_map_merge(con):
    a = ibis.literal({"a": 0, "b": 2})
    b = ibis.literal({"a": 1, "c": 3})
    expr = a + b

    assert con.execute(expr) == {"a": 1, "b": 2, "c": 3}


@mark_notimpl_risingwave_hstore
@mark_notyet_datafusion
def test_literal_map_getitem_broadcast(backend, alltypes, df):
    value = {"1": "a", "2": "b"}

    lookup_table = ibis.literal(value)
    expr = lookup_table[alltypes.string_col]

    result = expr.name("tmp").execute()
    expected = df.string_col.apply(value.get).rename("tmp")

    backend.assert_series_equal(result, expected)


keys = pytest.mark.parametrize(
    "keys",
    [
        pytest.param(["a", "b"], id="string"),
        pytest.param(
            [1, 2],
            marks=[mark_notyet_postgres, mark_notyet_snowflake],
            id="int",
        ),
        pytest.param(
            [True, False],
            marks=[mark_notyet_postgres, mark_notyet_snowflake],
            id="bool",
        ),
        pytest.param(
            [1.0, 2.0],
            marks=[
                pytest.mark.notyet(
                    "clickhouse",
                    reason="only supports str,int,bool,timestamp keys",
                    strict=False,
                ),
                mark_notyet_postgres,
                mark_notyet_snowflake,
            ],
            id="float",
        ),
        pytest.param(
            [ibis.timestamp("2021-01-01"), ibis.timestamp("2021-01-02")],
            marks=[mark_notyet_postgres, mark_notyet_snowflake],
            id="timestamp",
        ),
        pytest.param(
            [ibis.date(1, 2, 3), ibis.date(4, 5, 6)],
            marks=[
                pytest.mark.notyet(
                    "clickhouse", reason="only supports str,int,bool,timestamp keys"
                ),
                mark_notyet_postgres,
                mark_notyet_snowflake,
            ],
            id="date",
        ),
        pytest.param(
            [[1, 2], [3, 4]],
            marks=[
                pytest.mark.notyet(
                    "clickhouse", reason="only supports str,int,bool,timestamp keys"
                ),
                mark_notyet_postgres,
                mark_notyet_snowflake,
            ],
            id="array",
        ),
        pytest.param(
            [ibis.struct(dict(a=1)), ibis.struct(dict(a=2))],
            marks=[
                pytest.mark.notyet(
                    "clickhouse", reason="only supports str,int,bool,timestamp keys"
                ),
                mark_notyet_postgres,
                pytest.mark.notyet(
                    ["flink"],
                    raises=Py4JJavaError,
                    reason="does not support selecting struct key from map",
                ),
                mark_notyet_snowflake,
            ],
            id="struct",
        ),
    ],
)


values = pytest.mark.parametrize(
    "values",
    [
        pytest.param(["a", "b"], id="string"),
        pytest.param(
            [1, 2],
            marks=[
                mark_notyet_postgres,
            ],
            id="int",
        ),
        pytest.param(
            [True, False],
            marks=[
                mark_notyet_postgres,
            ],
            id="bool",
        ),
        pytest.param(
            [1.0, 2.0],
            marks=[
                mark_notyet_postgres,
            ],
            id="float",
        ),
        pytest.param(
            [ibis.timestamp("2021-01-01"), ibis.timestamp("2021-01-02")],
            marks=[
                mark_notyet_postgres,
            ],
            id="timestamp",
        ),
        pytest.param(
            [ibis.date(2021, 1, 1), ibis.date(2022, 2, 2)],
            marks=[mark_notyet_postgres],
            id="date",
        ),
        pytest.param(
            [[1, 2], [3, 4]],
            marks=[
                pytest.mark.notyet("clickhouse", reason="nested types can't be null"),
                mark_notyet_postgres,
            ],
            id="array",
        ),
        pytest.param(
            [ibis.struct(dict(a=1)), ibis.struct(dict(a=2))],
            marks=[
                pytest.mark.notyet("clickhouse", reason="nested types can't be null"),
                mark_notyet_postgres,
            ],
            id="struct",
        ),
    ],
)


@values
@keys
@mark_notimpl_risingwave_hstore
@mark_notyet_datafusion
def test_map_get_all_types(con, keys, values):
    m = ibis.map(ibis.array(keys), ibis.array(values))
    for key, val in zip(keys, values):
        if isinstance(val, ibis.Expr):
            val = con.execute(val)
        assert con.execute(m[key]) == val


@keys
@mark_notimpl_risingwave_hstore
@mark_notyet_datafusion
def test_map_contains_all_types(con, keys):
    a = ibis.array(keys)
    m = ibis.map(a, a)
    for key in keys:
        assert con.execute(m.contains(key))


@mark_notimpl_risingwave_hstore
@mark_notyet_datafusion
def test_literal_map_get_broadcast(backend, alltypes, df):
    value = {"1": "a", "2": "b"}

    lookup_table = ibis.literal(value)
    expr = lookup_table.get(alltypes.string_col, "default")

    result = expr.name("tmp").execute()
    expected = df.string_col.apply(lambda x: value.get(x, "default")).rename("tmp")

    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("keys", "values"),
    [
        param(
            ["a", "b"],
            [1, 2],
            id="string",
            marks=pytest.mark.notyet(
                ["postgres", "risingwave"],
                reason="only support maps of string -> string",
            ),
        ),
        param(["a", "b"], ["1", "2"], id="int"),
    ],
)
@mark_notimpl_risingwave_hstore
def test_map_construct_dict(con, keys, values):
    expr = ibis.map(keys, values)
    result = con.execute(expr.name("tmp"))
    assert result == dict(zip(keys, values))


@pytest.mark.notimpl(
    ["flink"],
    raises=pa.lib.ArrowInvalid,
    reason="Map array child array should have no nulls",
)
@mark_notimpl_risingwave_hstore
@mark_notyet_postgres
def test_map_construct_array_column(con, alltypes, df):
    expr = ibis.map(ibis.array([alltypes.string_col]), ibis.array([alltypes.int_col]))
    result = con.execute(expr)
    expected = df.apply(lambda row: {row["string_col"]: row["int_col"]}, axis=1)

    assert result.to_list() == expected.to_list()


@mark_notimpl_risingwave_hstore
@mark_notyet_postgres
@mark_notyet_datafusion
def test_map_get_with_compatible_value_smaller(con):
    value = ibis.literal({"A": 1000, "B": 2000})
    expr = value.get("C", 3)
    assert con.execute(expr) == 3


@mark_notimpl_risingwave_hstore
@mark_notyet_postgres
@mark_notyet_datafusion
def test_map_get_with_compatible_value_bigger(con):
    value = ibis.literal({"A": 1, "B": 2})
    expr = value.get("C", 3000)
    assert con.execute(expr) == 3000


@mark_notimpl_risingwave_hstore
@mark_notyet_postgres
@mark_notyet_datafusion
def test_map_get_with_incompatible_value_different_kind(con):
    value = ibis.literal({"A": 1000, "B": 2000})
    expr = value.get("C", 3.0)
    assert con.execute(expr) == 3.0


@mark_notimpl_risingwave_hstore
@mark_notyet_postgres
@mark_notyet_datafusion
@pytest.mark.parametrize("null_value", [None, ibis.null()])
def test_map_get_with_null_on_not_nullable(con, null_value):
    map_type = dt.Map(dt.string, dt.Int16(nullable=False))
    value = ibis.literal({"A": 1000, "B": 2000}).cast(map_type)
    expr = value.get("C", null_value)
    result = con.execute(expr)
    assert pd.isna(result)


@pytest.mark.parametrize("null_value", [None, ibis.null()])
@pytest.mark.notyet(
    ["flink"], raises=Py4JJavaError, reason="Flink cannot handle typeless nulls"
)
@mark_notimpl_risingwave_hstore
@mark_notyet_datafusion
def test_map_get_with_null_on_null_type_with_null(con, null_value):
    value = ibis.literal({"A": None, "B": None})
    expr = value.get("C", null_value)
    result = con.execute(expr)
    assert pd.isna(result)


@pytest.mark.notyet(
    ["flink"], raises=Py4JJavaError, reason="Flink cannot handle typeless nulls"
)
@mark_notimpl_risingwave_hstore
@mark_notyet_postgres
@mark_notyet_datafusion
def test_map_get_with_null_on_null_type_with_non_null(con):
    value = ibis.literal({"A": None, "B": None})
    expr = value.get("C", 1)
    assert con.execute(expr) == 1


@pytest.mark.notimpl(
    ["flink"],
    raises=exc.IbisError,
    reason="`tbl_properties` is required when creating table with schema",
)
@mark_notimpl_risingwave_hstore
@mark_notyet_datafusion
def test_map_create_table(con, temp_table):
    t = con.create_table(
        temp_table,
        schema=ibis.schema(dict(xyz="map<string, string>")),
    )
    assert t.schema()["xyz"].is_map()


@pytest.mark.notimpl(
    ["flink"],
    raises=exc.OperationNotDefinedError,
    reason="No translation rule for <class 'ibis.expr.operations.maps.MapLength'>",
)
@mark_notimpl_risingwave_hstore
@mark_notyet_datafusion
def test_map_length(con):
    expr = ibis.literal(dict(a="A", b="B")).length()
    assert con.execute(expr) == 2


@mark_notyet_datafusion
def test_map_keys_unnest(backend):
    expr = backend.map.kv.keys().unnest()
    result = expr.to_pandas()
    assert frozenset(result) == frozenset("abcdef")


@mark_notimpl_risingwave_hstore
@mark_notyet_datafusion
def test_map_contains_null(con):
    expr = ibis.map(["a"], ibis.literal([None], type="array<string>"))
    assert con.execute(expr.contains("a"))
    assert not con.execute(expr.contains("b"))
