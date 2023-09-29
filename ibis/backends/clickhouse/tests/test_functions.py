from __future__ import annotations

import math
from datetime import date, datetime
from operator import methodcaller

import pandas as pd
import pandas.testing as tm
import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis import literal as L
from ibis import udf

pytest.importorskip("clickhouse_connect")


@pytest.mark.parametrize("to_type", ["int8", "int16", "float32", "float", "!float64"])
def test_cast_double_col(alltypes, to_type, snapshot):
    expr = alltypes.double_col.cast(to_type)
    result = expr.compile()
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize(
    "to_type",
    [
        "int8",
        "int16",
        "!string",
        "timestamp",
        "date",
        "!map<string, int64>",
        "!struct<a: string, b: int64>",
    ],
)
def test_cast_string_col(alltypes, to_type, snapshot):
    expr = alltypes.string_col.cast(to_type)
    snapshot.assert_match(expr.compile(), "out.sql")


@pytest.mark.parametrize(
    "column",
    [
        "id",
        "bool_col",
        "tinyint_col",
        "smallint_col",
        "int_col",
        "bigint_col",
        "float_col",
        "double_col",
        "date_string_col",
        "string_col",
        "timestamp_col",
        "year",
        "month",
    ],
)
def test_noop_cast(alltypes, column, snapshot):
    col = alltypes[column]
    result = col.cast(col.type())
    assert result.equals(col)
    snapshot.assert_match(result.compile(), "out.sql")


def test_timestamp_cast(alltypes, snapshot):
    target = dt.Timestamp(nullable=False)
    result1 = alltypes.timestamp_col.cast(target)
    result2 = alltypes.int_col.cast(target)

    assert isinstance(result1, ir.TimestampColumn)
    assert isinstance(result2, ir.TimestampColumn)

    snapshot.assert_match(result1.compile(), "out1.sql")
    snapshot.assert_match(result2.compile(), "out2.sql")


def test_timestamp_now(con, snapshot):
    expr = ibis.now()
    snapshot.assert_match(con.compile(expr), "out.sql")


@pytest.mark.parametrize("unit", ["y", "m", "d", "w", "h", "minute"])
def test_timestamp_truncate(con, unit, snapshot):
    stamp = ibis.timestamp("2009-05-17 12:34:56")
    expr = stamp.truncate(unit)
    snapshot.assert_match(con.compile(expr), "out.sql")


@pytest.mark.parametrize(("value", "expected"), [(0, None), (5.5, 5.5)])
def test_nullifzero(con, value, expected):
    with pytest.warns(FutureWarning):
        result = con.execute(L(value).nullifzero())
    if expected is None:
        assert pd.isnull(result)
    else:
        assert result == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L(None).isnull(), True),
        (L(1).isnull(), False),
        (L(None).notnull(), False),
        (L(1).notnull(), True),
    ],
)
def test_isnull_notnull(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (ibis.coalesce(5, None, 4), 5),
        (ibis.coalesce(ibis.NA, 4, ibis.NA), 4),
        (ibis.coalesce(ibis.NA, ibis.NA, 3.14), 3.14),
    ],
)
def test_coalesce(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (ibis.NA.fillna(5), 5),
        (L(5).fillna(10), 5),
        (L(5).nullif(5), None),
        (L(10).nullif(5), 10),
    ],
)
def test_fillna_nullif(con, expr, expected):
    result = con.execute(expr)
    if expected is None:
        assert pd.isnull(result)
    else:
        assert result == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (L("foo_bar"), "String"),
        (L(5), "UInt8"),
        (L(1.2345), "Float64"),
        (L(datetime(2015, 9, 1, hour=14, minute=48, second=5)), "DateTime"),
        (L(date(2015, 9, 1)), "Date"),
        param(
            ibis.NA,
            "Null",
            marks=pytest.mark.xfail(
                raises=AssertionError,
                reason=(
                    "Client/server version mismatch not handled in the "
                    "clickhouse driver"
                ),
            ),
        ),
    ],
)
def test_typeof(con, value, expected):
    assert con.execute(value.typeof()) == expected


@pytest.mark.parametrize(("value", "expected"), [("foo_bar", 7), ("", 0)])
def test_string_length(con, value, expected):
    assert con.execute(L(value).length()) == expected


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (methodcaller("substr", 0, 3), "foo"),
        (methodcaller("substr", 4, 3), "bar"),
        (methodcaller("substr", 1), "oo_bar"),
    ],
)
def test_string_substring(con, op, expected):
    value = L("foo_bar")
    assert con.execute(op(value)) == expected


def test_string_column_substring(con, alltypes, snapshot):
    expr = alltypes.string_col.substr(2)
    snapshot.assert_match(expr.compile(), "out1.sql")
    assert len(con.execute(expr))

    expr = alltypes.string_col.substr(0, 3)
    snapshot.assert_match(expr.compile(), "out2.sql")
    assert len(con.execute(expr))


def test_string_reverse(con):
    assert con.execute(L("foo").reverse()) == "oof"


def test_string_upper(con):
    assert con.execute(L("foo").upper()) == "FOO"


def test_string_lower(con):
    assert con.execute(L("FOO").lower()) == "foo"


def test_string_lenght(con):
    assert con.execute(L("FOO").length()) == 3


@pytest.mark.parametrize(
    ("value", "op", "expected"),
    [
        (L("foobar"), methodcaller("contains", "bar"), True),
        (L("foobar"), methodcaller("contains", "foo"), True),
        (L("foobar"), methodcaller("contains", "baz"), False),
        (L("100%"), methodcaller("contains", "%"), True),
        (L("a_b_c"), methodcaller("contains", "_"), True),
    ],
)
def test_string_contains(con, op, value, expected):
    assert con.execute(op(value)) == expected


def test_re_replace(con):
    expr1 = L("Hello, World!").re_replace(".", r"\0\0")
    expr2 = L("Hello, World!").re_replace("^", "here: ")

    assert con.execute(expr1) == "HHeelllloo,,  WWoorrlldd!!"
    assert con.execute(expr2) == "here: Hello, World!"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(L("a"), 0), (L("b"), 1), (L("d"), -1)],  # TODO: what's the expected?
)
def test_find_in_set(con, value, expected):
    vals = list("abc")
    expr = value.find_in_set(vals)
    assert con.execute(expr) == expected


def test_string_column_find_in_set(con, alltypes, snapshot):
    s = alltypes.string_col
    vals = list("abc")

    expr = s.find_in_set(vals)
    snapshot.assert_match(expr.compile(), "out.sql")
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L("foobar").find("bar"), 3),
        (L("foobar").find("baz"), -1),
        (L("foobar").like("%bar"), True),
        (L("foobar").like("foo%"), True),
        (L("foobar").like("%baz%"), False),
        (L("foobar").like(["%bar"]), True),
        (L("foobar").like(["foo%"]), True),
        (L("foobar").like(["%baz%"]), False),
        (L("foobar").like(["%bar", "foo%"]), True),
        (L("foobarfoo").replace("foo", "H"), "HbarH"),
    ],
)
def test_string_find_like(con, expr, expected):
    assert con.execute(expr) == expected


def test_string_column_like(con, alltypes, snapshot):
    expr = alltypes.string_col.like("foo%")
    snapshot.assert_match(expr.compile(), "out1.sql")
    assert len(con.execute(expr))

    expr = alltypes.string_col.like(["foo%", "%bar"])
    snapshot.assert_match(expr.compile(), "out2.sql")
    assert len(con.execute(expr))


def test_string_column_find(con, alltypes, snapshot):
    s = alltypes.string_col

    expr = s.find("a")
    snapshot.assert_match(expr.compile(), "out1.sql")
    assert len(con.execute(expr))

    expr = s.find(s)
    snapshot.assert_match(expr.compile(), "out2.sql")
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    "call",
    [
        param(methodcaller("log"), id="log"),
        param(methodcaller("log2"), id="log2"),
        param(methodcaller("log10"), id="log10"),
        param(methodcaller("round"), id="round"),
        param(methodcaller("round", 0), id="round_0"),
        param(methodcaller("round", 2), id="round_2"),
        param(methodcaller("exp"), id="exp"),
        param(methodcaller("abs"), id="abs"),
        param(methodcaller("ceil"), id="ceil"),
        param(methodcaller("sqrt"), id="sqrt"),
        param(methodcaller("sign"), id="sign"),
    ],
)
def test_translate_math_functions(con, alltypes, call, snapshot):
    expr = call(alltypes.double_col)
    snapshot.assert_match(expr.compile(), "out.sql")
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        pytest.param(L(-5).abs(), 5, id="abs_neg"),
        pytest.param(L(5).abs(), 5, id="abs"),
        pytest.param(L(5.5).round(), 6.0, id="round"),
        pytest.param(L(5.556).round(2), 5.56, id="round_places"),
        pytest.param(L(5.556).ceil(), 6.0, id="ceil"),
        pytest.param(L(5.556).floor(), 5.0, id="floor"),
        pytest.param(L(5.556).sign(), 1, id="sign"),
        pytest.param(L(-5.556).sign(), -1, id="sign_neg"),
        pytest.param(L(0).sign(), 0, id="sign_zero"),
        pytest.param(L(5.556).sqrt(), math.sqrt(5.556), id="sqrt"),
        pytest.param(L(5.556).log(2), math.log(5.556, 2), id="log2_arg"),
        pytest.param(L(5.556).log2(), math.log(5.556, 2), id="log2"),
        pytest.param(L(5.556).log10(), math.log10(5.556), id="log10"),
        # clickhouse has different functions for exp/ln that are faster
        # than the defaults, but less precise
        #
        # we can't use the e() function as it still gives different results
        # from `math.exp`
        pytest.param(
            L(5.556).exp().round(8),
            round(math.exp(5.556), 8),
            id="exp",
        ),
        pytest.param(
            L(5.556).ln().round(7),
            round(math.log(5.556), 7),
            id="ln",
        ),
    ],
)
def test_math_functions(con, expr, expected):
    assert con.execute(expr) == expected


def test_greatest_least(con, alltypes, snapshot):
    expr = ibis.greatest(alltypes.int_col, 10)
    snapshot.assert_match(expr.compile(), "out1.sql")
    assert len(con.execute(expr))

    expr = ibis.greatest(alltypes.int_col, alltypes.bigint_col)
    snapshot.assert_match(expr.compile(), "out2.sql")
    assert len(con.execute(expr))

    expr = ibis.least(alltypes.int_col, 10)
    snapshot.assert_match(expr.compile(), "out3.sql")
    assert len(con.execute(expr))

    expr = ibis.least(alltypes.int_col, alltypes.bigint_col)
    snapshot.assert_match(expr.compile(), "out4.sql")
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L("abcd").re_search("[a-z]"), True),
        (L("abcd").re_search(r"[\d]+"), False),
        (L("1222").re_search(r"[\d]+"), True),
    ],
)
def test_regexp(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(L("abcd").re_extract("([a-z]+)", 0), "abcd", id="simple"),
        # (L('abcd').re_extract('(ab)(cd)', 1), 'cd'),
        # valid group number but no match => None
        param(L("abcd").re_extract(r"(\\d)", 0), None, id="valid_group_no_match"),
        # match but not a valid group number => NULL
        param(L("abcd").re_extract("abcd", 3), None, id="invalid_group_match"),
    ],
)
def test_regexp_extract(con, expr, expected):
    assert con.execute(expr) == expected


def test_column_regexp_extract(con, alltypes, snapshot):
    expr = alltypes.string_col.re_extract(r"[\d]+", 3)
    snapshot.assert_match(expr.compile(), "out.sql")
    assert len(con.execute(expr))


def test_column_regexp_replace(con, alltypes, snapshot):
    expr = alltypes.string_col.re_replace(r"[\d]+", "aaa")
    snapshot.assert_match(expr.compile(), "out.sql")
    assert len(con.execute(expr))


def test_numeric_builtins_work(alltypes, df):
    expr = alltypes.double_col
    result = expr.execute()
    expected = df.double_col.fillna(0)
    tm.assert_series_equal(result, expected)


def test_null_column(alltypes):
    t = alltypes
    nrows = t.count().execute()
    expr = t.mutate(na_column=ibis.NA).na_column
    result = expr.execute()
    expected = pd.Series([None] * nrows, name="na_column")
    tm.assert_series_equal(result, expected)


def test_literal_none_to_nullable_column(alltypes):
    # GH: 2985
    t = alltypes
    nrows = t.count().execute()
    expr = t.mutate(
        ibis.literal(None, dt.String(nullable=True)).name("nullable_string_column")
    )
    result = expr["nullable_string_column"].execute()
    expected = pd.Series([None] * nrows, name="nullable_string_column")
    tm.assert_series_equal(result, expected)


def test_timestamp_from_integer(con, alltypes, snapshot):
    # timestamp_col has datetime type
    expr = alltypes.int_col.to_timestamp()
    snapshot.assert_match(expr.compile(), "out.sql")
    assert len(con.execute(expr))


def test_count_distinct_with_filter(alltypes):
    expr = alltypes.string_col.nunique(where=alltypes.string_col.cast("int64") > 1)
    result = expr.execute()
    expected = alltypes.string_col.execute()
    expected = expected[expected.astype("int64") > 1].nunique()
    assert result == expected


@pytest.mark.parametrize(
    ("sep", "where_case"),
    [
        param(",", None, id="comma_none"),
        param("-", None, id="minus_none"),
        param(",", 0, id="comma_zero"),
    ],
)
def test_group_concat(alltypes, sep, where_case, snapshot):
    where = None if where_case is None else alltypes.bool_col == where_case
    expr = alltypes.string_col.group_concat(sep, where)
    snapshot.assert_match(expr.compile(), "out.sql")


def test_hash(alltypes, snapshot):
    expr = alltypes.string_col.hash()
    snapshot.assert_match(expr.compile(), "out.sql")


def test_udf_in_array_map(alltypes):
    @udf.scalar.builtin(name="plus")
    def my_add(a: int, b: int) -> int:
        ...

    n = 5
    expr = (
        alltypes[alltypes.int_col == 1]
        .limit(n)
        .int_col.collect()
        .map(lambda x: my_add(x, 1))
    )
    result = expr.execute()
    assert result == [2] * n


def test_udf_in_array_filter(alltypes):
    @udf.scalar.builtin(name="equals")
    def my_eq(a: int, b: int) -> bool:
        ...

    expr = alltypes.int_col.collect().filter(lambda x: my_eq(x, 1))
    result = expr.execute()
    assert set(result) == {1}
