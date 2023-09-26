from __future__ import annotations

import datetime
import re
import time
from operator import floordiv, methodcaller, truediv

import pandas as pd
import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import _
from ibis.common.annotations import ValidationError

to_sql = ibis.bigquery.compile


@pytest.fixture(scope="module")
def alltypes():
    return ibis.table(
        ibis.schema(
            dict(
                id="int32",
                bool_col="boolean",
                tinyint_col="int8",
                smallint_col="int16",
                int_col="int32",
                bigint_col="int64",
                float_col="float32",
                double_col="float64",
                date_string_col="string",
                string_col="string",
                timestamp_col=dt.Timestamp(timezone="UTC"),
                year="int32",
                month="int32",
            )
        ),
        name="functional_alltypes",
    )


@pytest.mark.parametrize(
    ("case", "dtype"),
    [
        param(datetime.date(2017, 1, 1), dt.date, id="date"),
        param(pd.Timestamp("2017-01-01"), dt.date, id="timestamp_date"),
        param("2017-01-01", dt.date, id="string_date"),
        param(datetime.datetime(2017, 1, 1, 4, 55, 59), dt.timestamp, id="datetime"),
        param("2017-01-01 04:55:59", dt.timestamp, id="string_timestamp"),
        param(pd.Timestamp("2017-01-01 04:55:59"), dt.timestamp, id="timestamp"),
    ],
)
def test_literal_year(case, dtype, snapshot):
    expr = ibis.literal(case, type=dtype).year()
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize(
    ("case", "dtype"),
    [
        param(datetime.date(2017, 1, 1), dt.date, id="date"),
        param(pd.Timestamp("2017-01-01"), dt.date, id="timestamp_date"),
        param("2017-01-01", dt.date, id="string_date"),
        param(datetime.datetime(2017, 1, 1, 4, 55, 59), dt.timestamp, id="datetime"),
        param("2017-01-01 04:55:59", dt.timestamp, id="string_timestamp"),
        param(pd.Timestamp("2017-01-01 04:55:59"), dt.timestamp, id="timestamp"),
    ],
)
def test_day_of_week(case, dtype, snapshot):
    date_var = ibis.literal(case, type=dtype)

    expr = date_var.day_of_week.index()
    snapshot.assert_match(to_sql(expr), "index.sql")

    expr = date_var.day_of_week.full_name()
    snapshot.assert_match(to_sql(expr), "name.sql")


@pytest.mark.parametrize(
    ("case", "dtype"),
    [
        param("test of hash", dt.string, id="string"),
        param(b"test of hash", dt.binary, id="binary"),
    ],
)
def test_hash(case, dtype, snapshot):
    var = ibis.literal(case, type=dtype)
    expr = var.hash()
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize(("case", "dtype"), [("test", "string"), (b"test", "binary")])
@pytest.mark.parametrize("how", ["md5", "sha1", "sha256", "sha512"])
def test_hashbytes(case, how, dtype, snapshot):
    var = ibis.literal(case, type=dtype)
    expr = var.hashbytes(how=how).name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize(
    ("case", "unit"),
    (
        param(123456789, "s", id="s"),
        param(-123456789, "ms", id="ms"),
        param(123456789, "us", id="us"),
        param(1234567891011, "ns", id="ns"),
    ),
)
def test_integer_to_timestamp(case, unit, snapshot):
    expr = ibis.literal(case, type=dt.int64).to_timestamp(unit=unit).name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize(
    ("case",),
    [
        param("a\\b\\c", id="escape_backslash"),
        param("a\ab\bc\fd\ne\rf\tg\vh", id="escape_ascii_sequences"),
        param("a'b\"c", id="escape_quote"),
        param("`~!@#$%^&*()_=+-|[]{};:/?<>", id="not_escape_special_characters"),
    ],
)
def test_literal_string(case, snapshot):
    expr = ibis.literal(case)
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize(
    ("case", "dtype"),
    [
        param(datetime.datetime(2017, 1, 1, 4, 55, 59), dt.timestamp, id="datetime"),
        param("2017-01-01 04:55:59", dt.timestamp, id="string_timestamp"),
        param(pd.Timestamp("2017-01-01 04:55:59"), dt.timestamp, id="timestamp"),
        param(datetime.time(4, 55, 59), dt.time, id="time"),
        param("04:55:59", dt.time, id="string_time"),
    ],
)
def test_literal_timestamp_or_time(case, dtype, snapshot):
    expr = ibis.literal(case, type=dtype).hour().name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_projection_fusion_only_peeks_at_immediate_parent(snapshot):
    schema = [
        ("file_date", "timestamp"),
        ("PARTITIONTIME", "date"),
        ("val", "int64"),
    ]
    table = ibis.table(schema, name="unbound_table")
    table = table[table.PARTITIONTIME < ibis.date("2017-01-01")]
    table = table.mutate(file_date=table.file_date.cast("date"))
    table = table[table.file_date < ibis.date("2017-01-01")]
    table = table.mutate(XYZ=table.val * 2)
    expr = table.join(table.view())[table]
    snapshot.assert_match(to_sql(expr), "out.sql")


unit_full_names = {
    "Y": "year",
    "Q": "quarter",
    "M": "month",
    "W": "week",
    "D": "day",
    "h": "hour",
    "m": "minute",
    "s": "second",
    "ms": "millis",
    "us": "micros",
}


@pytest.mark.parametrize(
    ("unit", "func"),
    [
        ("Y", "timestamp"),
        ("Q", "timestamp"),
        ("M", "timestamp"),
        ("W", "timestamp"),
        ("D", "timestamp"),
        ("h", "timestamp"),
        ("m", "timestamp"),
        ("s", "timestamp"),
        ("ms", "timestamp"),
        ("us", "timestamp"),
        ("Y", "date"),
        ("Q", "date"),
        ("M", "date"),
        ("W", "date"),
        ("D", "date"),
        ("h", "time"),
        ("m", "time"),
        ("s", "time"),
        ("ms", "time"),
        ("us", "time"),
    ],
    ids=lambda p: unit_full_names.get(p, p),
)
def test_temporal_truncate(unit, func, snapshot):
    dtype = getattr(dt, func)
    t = ibis.table([("a", dtype)], name="t")
    expr = t.a.truncate(unit).name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize("kind", ["date", "time"])
def test_extract_temporal_from_timestamp(kind, snapshot):
    t = ibis.table([("ts", dt.timestamp)], name="t")
    expr = getattr(t.ts, kind)().name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_now(snapshot):
    expr = ibis.now()
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_binary(snapshot):
    t = ibis.table([("value", "double")], name="t")
    expr = t["value"].cast(dt.binary)
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_substring(snapshot):
    t = ibis.table([("value", "string")], name="t")
    expr = t["value"].substr(3, 1).name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_substring_neg_length():
    t = ibis.table([("value", "string")], name="t")
    expr = t["value"].substr(3, -1).name("tmp")
    with pytest.raises(
        Exception, match=r"Length parameter must be a non-negative value\."
    ):
        to_sql(expr)


def test_bucket(snapshot):
    t = ibis.table([("value", "double")], name="t")
    buckets = [0, 1, 3]
    expr = t.value.bucket(buckets).name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize(
    ("kind", "begin", "end"),
    [
        param("preceding", None, 1, id="preceding"),
        param("following", 1, None, id="following"),
    ],
)
def test_window_unbounded(kind, begin, end, snapshot):
    t = ibis.table([("a", "int64")], name="t")
    kwargs = {kind: (begin, end)}
    expr = t.a.sum().over(ibis.window(**kwargs)).name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_large_compile():
    """Tests that compiling a large expression tree finishes within a
    reasonable amount of time."""
    num_columns = 20
    num_joins = 7

    class MockBackend(ibis.backends.bigquery.Backend):
        pass

    names = [f"col_{i}" for i in range(num_columns)]
    schema = ibis.Schema(dict.fromkeys(names, "string"))
    ibis_client = MockBackend()
    table = ops.SQLQueryResult("select * from t", schema, ibis_client).to_expr()
    for _ in range(num_joins):  # noqa: F402
        table = table.mutate(dummy=ibis.literal(""))
        table = table.left_join(table, ["dummy"])[[table]]

    start = time.time()
    table.compile()
    delta = time.time() - start
    assert delta < 10


@pytest.mark.parametrize(
    ("operation", "keywords"),
    [
        param("union", {"distinct": False}, id="union_all"),
        param("union", {"distinct": True}, id="union_distinct"),
        param("intersect", {}, id="intersect"),
        param("difference", {}, id="difference"),
    ],
)
def test_set_operation(operation, keywords, snapshot):
    t0 = ibis.table([("a", "int64")], name="t0")
    t1 = ibis.table([("a", "int64")], name="t1")
    expr = getattr(t0, operation)(t1, **keywords)
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_geospatial_point(snapshot):
    t = ibis.table([("lon", "float64"), ("lat", "float64")], name="t")
    expr = t.lon.point(t.lat).name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_geospatial_azimuth(snapshot):
    t = ibis.table([("p0", "point"), ("p1", "point")], name="t")
    expr = t.p0.azimuth(t.p1).name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_geospatial_unary_union(snapshot):
    t = ibis.table([("geog", "geography")], name="t")
    expr = t.geog.unary_union().name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize(
    ("operation", "keywords"),
    [
        param("area", {}, id="aread"),
        param("as_binary", {}, id="as_binary"),
        param("as_text", {}, id="as_text"),
        param("buffer", {"radius": 5.2}, id="buffer"),
        param("centroid", {}, id="centroid"),
        param("end_point", {}, id="end_point"),
        param("geometry_type", {}, id="geometry_type"),
        param("length", {}, id="length"),
        param("n_points", {}, id="npoints"),
        param("perimeter", {}, id="perimeter"),
        param("point_n", {"n": 3}, id="point_n"),
        param("start_point", {}, id="start_point"),
    ],
)
def test_geospatial_unary(operation, keywords, snapshot):
    t = ibis.table([("geog", "geography")], name="t")
    expr = getattr(t.geog, operation)(**keywords).name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize(
    ("operation", "keywords"),
    [
        param("contains", {}, id="contains"),
        param("covers", {}, id="covers"),
        param("covered_by", {}, id="covered_by"),
        param("d_within", {"distance": 5.2}, id="d_within"),
        param("difference", {}, id="difference"),
        param("disjoint", {}, id="disjoint"),
        param("distance", {}, id="distance"),
        param("geo_equals", {}, id="geo_equals"),
        param("intersection", {}, id="intersection"),
        param("intersects", {}, id="intersects"),
        param("max_distance", {}, id="max_distance"),
        param("touches", {}, id="touches"),
        param("union", {}, id="union"),
        param("within", {}, id="within"),
    ],
)
def test_geospatial_binary(operation, keywords, snapshot):
    t = ibis.table([("geog0", "geography"), ("geog1", "geography")], name="t")
    expr = getattr(t.geog0, operation)(t.geog1, **keywords).name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize("operation", ["x_max", "x_min", "y_max", "y_min"])
def test_geospatial_minmax(operation, snapshot):
    t = ibis.table([("geog", "geography")], name="t")
    expr = getattr(t.geog, operation)().name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize("dimension_name", ["x", "y"])
def test_geospatial_xy(dimension_name, snapshot):
    t = ibis.table([("pt", "point")], name="t")
    expr = getattr(t.pt, dimension_name)().name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_geospatial_simplify(snapshot):
    t = ibis.table([("geog", "geography")], name="t")
    expr = t.geog.simplify(5.2, preserve_collapsed=False).name("tmp")
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_geospatial_simplify_error():
    t = ibis.table([("geog", "geography")], name="t")
    expr = t.geog.simplify(5.2, preserve_collapsed=True).name("tmp")
    with pytest.raises(Exception) as exception_info:
        to_sql(expr)
    expected = "BigQuery simplify does not support preserving collapsed geometries, must pass preserve_collapsed=False"
    assert str(exception_info.value) == expected


def test_timestamp_accepts_date_literals(alltypes):
    date_string = "2009-03-01"
    p = ibis.param(dt.timestamp).name("param_0")
    expr = alltypes.mutate(param=p)
    params = {p: date_string}
    result = to_sql(expr, params=params)
    assert re.search(r"@param_\d+ AS `param`", result) is not None


@pytest.mark.parametrize("distinct", [True, False])
def test_union(alltypes, distinct, snapshot):
    expr = alltypes.union(alltypes, distinct=distinct)
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize("op", [truediv, floordiv])
def test_divide_by_zero(alltypes, op, snapshot):
    expr = op(alltypes.double_col, 0)
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_identical_to(alltypes, snapshot):
    expr = alltypes[
        _.string_col.identical_to("a") & _.date_string_col.identical_to("b")
    ]
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_to_timestamp_no_timezone(alltypes, snapshot):
    expr = alltypes.date_string_col.to_timestamp("%F")
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_to_timestamp_timezone(alltypes, snapshot):
    expr = (alltypes.date_string_col + " America/New_York").to_timestamp("%F %Z")
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize(
    "window",
    [
        param(
            ibis.window(
                preceding=1, following=0, group_by="year", order_by="timestamp_col"
            ),
            id="prec_current",
        ),
        param(
            ibis.window(
                preceding=0, following=2, group_by="year", order_by="timestamp_col"
            ),
            id="current_foll",
        ),
        param(
            ibis.window(preceding=(4, 2), group_by="year", order_by="timestamp_col"),
            id="prec_prec",
        ),
    ],
)
def test_window_function(alltypes, window, snapshot):
    expr = alltypes.mutate(win_avg=_.float_col.mean().over(window))
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize(
    "window",
    [
        param(
            ibis.range_window(
                preceding=1, following=0, group_by="year", order_by="month"
            ),
            id="prec_foll",
        ),
        param(
            ibis.range_window(
                preceding=(4, 2), group_by="year", order_by="timestamp_col"
            ),
            id="prec_prec",
        ),
    ],
)
def test_range_window_function(alltypes, window, snapshot):
    expr = alltypes.mutate(two_month_avg=_.float_col.mean().over(window))
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize(
    "preceding",
    [
        param(5, id="five"),
        param(ibis.interval(nanoseconds=1), id="nanos", marks=pytest.mark.xfail),
        param(ibis.interval(microseconds=1), id="micros"),
        param(ibis.interval(seconds=1), id="seconds"),
        param(ibis.interval(minutes=1), id="minutes"),
        param(ibis.interval(hours=1), id="hours"),
        param(ibis.interval(days=1), id="days"),
        param(2 * ibis.interval(days=1), id="two_days"),
        param(ibis.interval(weeks=1), id="week", marks=pytest.mark.xfail),
    ],
)
def test_trailing_range_window(alltypes, preceding, snapshot):
    w = ibis.trailing_range_window(preceding=preceding, order_by="timestamp_col")
    expr = alltypes.mutate(win_avg=_.float_col.mean().over(w))
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize("distinct1", [True, False])
@pytest.mark.parametrize("distinct2", [True, False])
def test_union_cte(alltypes, distinct1, distinct2, snapshot):
    t = alltypes
    expr1 = t.group_by(t.string_col).agg(metric=t.double_col.sum())
    expr2 = expr1.view()
    expr3 = expr1.view()
    expr = expr1.union(expr2, distinct=distinct1).union(expr3, distinct=distinct2)
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize("funcname", ["sum", "mean"])
def test_bool_reducers(alltypes, funcname, snapshot):
    method = methodcaller(funcname)
    expr = method(alltypes.bool_col)
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_bool_reducers_where_simple(alltypes, snapshot):
    b = alltypes.bool_col
    m = alltypes.month
    expr = b.mean(where=m > 6)
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_bool_reducers_where_conj(alltypes, snapshot):
    b = alltypes.bool_col
    m = alltypes.month
    expr2 = b.sum(where=((m > 6) & (m < 10)))
    snapshot.assert_match(to_sql(expr2), "out.sql")


@pytest.mark.parametrize("agg", ["approx_median", "approx_nunique"])
@pytest.mark.parametrize(
    "where",
    [param(lambda _: None, id="no_filter"), param(lambda t: t.month > 0, id="filter")],
)
def test_approx(alltypes, agg, where, snapshot):
    d = alltypes.double_col
    method = methodcaller(agg, where=where(alltypes))
    expr = method(d)
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize("funcname", ["bit_and", "bit_or", "bit_xor"])
@pytest.mark.parametrize(
    "where",
    [
        param(lambda _: None, id="no_filter"),
        param(lambda t: t.bigint_col > 0, id="filter"),
    ],
)
def test_bit(alltypes, funcname, where, snapshot):
    method = methodcaller(funcname, where=where(alltypes))
    expr = method(alltypes.int_col)
    snapshot.assert_match(to_sql(expr), "out.sql")


@pytest.mark.parametrize("how", ["pop", "sample"])
def test_cov(alltypes, how, snapshot):
    d = alltypes.double_col
    expr = d.cov(d, how=how)
    snapshot.assert_match(to_sql(expr), "out.sql")


def test_cov_invalid_how(alltypes):
    d = alltypes.double_col

    with pytest.raises(ValidationError):
        d.cov(d, how="error")


def test_compile_toplevel(snapshot):
    t = ibis.table([("foo", "double")], name="t0")

    # it works!
    expr = t.foo.sum()
    result = to_sql(expr)
    snapshot.assert_match(result, "out.sql")


def test_scalar_param_scope(alltypes):
    t = alltypes
    param = ibis.param("timestamp")
    result = to_sql(t.mutate(param=param), params={param: "2017-01-01"})
    assert re.search(r"@param_\d+ AS `param`", result) is not None


def test_cast_float_to_int(alltypes, snapshot):
    expr = alltypes.double_col.cast("int64")
    result = to_sql(expr)
    snapshot.assert_match(result, "out.sql")


def test_unnest(snapshot):
    table = ibis.table(
        dict(
            rowindex="int",
            repeated_struct_col=dt.Array(
                dt.Struct(
                    dict(
                        nested_struct_col=dt.Array(
                            dt.Struct(
                                dict(
                                    doubly_nested_array="array<int>",
                                    doubly_nested_field="string",
                                )
                            )
                        )
                    )
                )
            ),
        ),
        name="array_test",
    )
    repeated_struct_col = table.repeated_struct_col

    # Works as expected :-)
    result = ibis.bigquery.compile(
        table.select("rowindex", repeated_struct_col.unnest())
    )
    snapshot.assert_match(result, "out_one_unnest.sql")

    result = ibis.bigquery.compile(
        table.select(
            "rowindex", level_one=repeated_struct_col.unnest().nested_struct_col
        ).select(level_two=lambda t: t.level_one.unnest())
    )
    snapshot.assert_match(result, "out_two_unnests.sql")


def test_compile_in_memory_table(snapshot):
    t = ibis.memtable({"Column One": [1, 2, 3]})
    result = ibis.bigquery.compile(t)
    snapshot.assert_match(result, "out.sql")
