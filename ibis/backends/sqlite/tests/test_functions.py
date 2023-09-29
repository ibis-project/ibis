from __future__ import annotations

import math
import sqlite3
import uuid

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
from packaging.version import parse
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis import config
from ibis import literal as L

sa = pytest.importorskip("sqlalchemy")


@pytest.mark.parametrize(
    ("func", "expected"),
    [
        (
            lambda t: t.double_col.cast(dt.int8),
            lambda at: sa.cast(at.c.double_col, sa.SMALLINT),
        ),
        (
            lambda t: t.string_col.cast(dt.float64),
            lambda at: sa.cast(at.c.string_col, sa.REAL),
        ),
        (
            lambda t: t.string_col.cast(dt.float32),
            lambda at: sa.cast(at.c.string_col, sa.REAL),
        ),
    ],
)
def test_cast(alltypes, alltypes_sqla, translate, func, expected):
    expr = func(alltypes)
    assert translate(expr.op()) == str(expected(alltypes_sqla.alias("t0")))


@pytest.mark.parametrize(
    ("func", "expected_func"),
    [
        param(
            lambda t: t.timestamp_col.cast(dt.timestamp),
            lambda at: at.c.timestamp_col,
            id="timestamp_col",
        ),
        param(
            lambda t: t.int_col.cast(dt.timestamp),
            lambda at: sa.func.datetime(at.c.int_col, "unixepoch"),
            id="cast_integer_to_timestamp",
        ),
    ],
)
def test_timestamp_cast_noop(
    alltypes, func, translate, alltypes_sqla, expected_func, sqla_compile
):
    # See GH #592
    result = func(alltypes)
    expected = expected_func(alltypes_sqla.alias("t0"))
    assert translate(result.op()) == sqla_compile(expected)


def test_timestamp_functions(con):
    value = ibis.timestamp("2015-09-01 14:48:05.359")
    expr = value.strftime("%Y%m%d")
    expected = "20150901"
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L(3) + L(4), 7),
        (L(3) - L(4), -1),
        (L(3) * L(4), 12),
        (L(12) / L(4), 3),
        (L(12) ** L(2), 144),
        (L(12) % L(5), 2),
    ],
)
def test_binary_arithmetic(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L(7) / L(2), 3.5),
        (L(7) // L(2), 3),
        (L(7).floordiv(2), 3),
        (L(2).rfloordiv(7), 3),
    ],
)
def test_div_floordiv(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(("lit", "expected"), [(L(0), None), (L(5.5), 5.5)])
def test_nullifzero(con, lit, expected):
    with pytest.warns(FutureWarning):
        expr = lit.nullifzero()
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"), [(L("foo_bar").length(), 7), (L("").length(), 0)]
)
def test_string_length(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L("foo_bar").left(3), "foo"),
        (L("foo_bar").right(3), "bar"),
        (L("foo_bar").substr(0, 3), "foo"),
        (L("foo_bar").substr(4, 3), "bar"),
        (L("foo_bar").substr(1), "oo_bar"),
    ],
)
def test_string_substring(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L("   foo   ").lstrip(), "foo   "),
        (L("   foo   ").rstrip(), "   foo"),
        (L("   foo   ").strip(), "foo"),
    ],
)
def test_string_strip(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [(L("foo").upper(), "FOO"), (L("FOO").lower(), "foo")],
)
def test_string_upper_lower(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L("foobar").contains("bar"), True),
        (L("foobar").contains("foo"), True),
        (L("foobar").contains("baz"), False),
    ],
)
def test_string_contains(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [(L("foobar").find("bar"), 3), (L("foobar").find("baz"), -1)],
)
def test_string_find(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L("foobar").like("%bar"), True),
        (L("foobar").like("foo%"), True),
        (L("foobar").like("%baz%"), False),
        (L("foobar").like(["%bar"]), True),
        (L("foobar").like(["foo%"]), True),
        (L("foobar").like(["%baz%"]), False),
        (L("foobar").like(["%bar", "foo%"]), True),
    ],
)
def test_string_like(con, expr, expected):
    assert con.execute(expr) == expected


def test_str_replace(con):
    expr = L("foobarfoo").replace("foo", "H")
    expected = "HbarH"
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L(-5).abs(), 5),
        (L(5).abs(), 5),
        (ibis.least(L(5), L(10), L(1)), 1),
        (ibis.greatest(L(5), L(10), L(1)), 10),
        (L(5.5).round(), 6.0),
        (L(5.556).round(2), 5.56),
        (L(5.556).sqrt(), math.sqrt(5.556)),
        (L(5.556).ceil(), 6.0),
        (L(5.556).floor(), 5.0),
        (L(5.556).exp(), math.exp(5.556)),
        (L(5.556).sign(), 1),
        (L(-5.556).sign(), -1),
        (L(0).sign(), 0),
        (L(5.556).log(2), math.log(5.556, 2)),
        (L(5.556).ln(), math.log(5.556)),
        (L(5.556).log2(), math.log(5.556, 2)),
        (L(5.556).log10(), math.log10(5.556)),
    ],
)
def test_math_functions(con, expr, expected):
    assert con.execute(expr) == expected


NULL_STRING = L(None).cast(dt.string)
NULL_INT64 = L(None).cast(dt.int64)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L("abcd").re_search("[a-z]"), True),
        (L("abcd").re_search(r"[\d]+"), False),
        (L("1222").re_search(r"[\d]+"), True),
        (L("abcd").re_search(None), None),
        (NULL_STRING.re_search("[a-z]"), None),
        (NULL_STRING.re_search(NULL_STRING), None),
    ],
)
def test_regexp_search(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L("abcd").re_replace("[ab]", ""), "cd"),
        (L(None).cast(dt.string).re_replace(NULL_STRING, NULL_STRING), None),
        (L("abcd").re_replace(NULL_STRING, NULL_STRING), None),
        (L("abcd").re_replace("a", NULL_STRING), None),
        (L("abcd").re_replace(NULL_STRING, "a"), None),
        (NULL_STRING.re_replace("a", NULL_STRING), None),
        (NULL_STRING.re_replace(NULL_STRING, "a"), None),
    ],
)
def test_regexp_replace(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (L("1222").re_extract(r"1(22)\d+", 1).cast("int64"), 22),
        (L("abcd").re_extract(r"(\d+)", 1), None),
        (L("1222").re_extract("([a-z]+)", 1), None),
        (L("1222").re_extract(r"1(22)\d+", 2), None),
        # extract nulls
        (NULL_STRING.re_extract(NULL_STRING, NULL_INT64), None),
        (L("abcd").re_extract(NULL_STRING, NULL_INT64), None),
        (L("abcd").re_extract("a", NULL_INT64), None),
        (L("abcd").re_extract(NULL_STRING, 1), None),
        (NULL_STRING.re_extract("a", NULL_INT64), None),
        (NULL_STRING.re_extract(NULL_STRING, 1), None),
    ],
)
def test_regexp_extract(con, expr, expected):
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
    assert con.execute(expr) == expected


def test_numeric_builtins_work(alltypes, df):
    expr = alltypes.double_col.fillna(0).name("tmp")
    result = expr.execute()
    expected = df.double_col.fillna(0)
    expected.name = "tmp"
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("func", "expected_func"),
    [
        (
            lambda t: (t.double_col > 20).ifelse(10, -20),
            lambda df: pd.Series(
                np.where(df.double_col > 20, 10, -20), name="tmp", dtype="int8"
            ),
        ),
        (
            lambda t: (t.double_col > 20).ifelse(10, -20).abs(),
            lambda df: pd.Series(
                np.where(df.double_col > 20, 10, -20), name="tmp", dtype="int8"
            ).abs(),
        ),
    ],
)
def test_ifelse(alltypes, df, func, expected_func):
    expr = func(alltypes).name("tmp")
    result = expr.execute()
    expected = expected_func(df)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("func", "expected_func"),
    [
        # tier and histogram
        param(
            lambda d: d.bucket([0, 10, 25, 50, 100]),
            lambda s: pd.cut(s, [0, 10, 25, 50, 100], right=False, labels=False).astype(
                "int8"
            ),
            id="default",
        ),
        param(
            lambda d: d.bucket([0, 10, 25, 50], include_over=True),
            lambda s: pd.cut(
                s, [0, 10, 25, 50, np.inf], right=False, labels=False
            ).astype("int8"),
            id="include_over",
        ),
        param(
            lambda d: d.bucket([0, 10, 25, 50], close_extreme=False),
            lambda s: pd.cut(s, [0, 10, 25, 50], right=False, labels=False),
            id="no_close_extreme",
        ),
        param(
            lambda d: d.bucket([0, 10, 25, 50], closed="right", close_extreme=False),
            lambda s: pd.cut(
                s,
                [0, 10, 25, 50],
                include_lowest=False,
                right=True,
                labels=False,
            ),
            id="closed_right_no_close_extreme",
        ),
        param(
            lambda d: d.bucket([10, 25, 50, 100], include_under=True),
            lambda s: pd.cut(s, [0, 10, 25, 50, 100], right=False, labels=False).astype(
                "int8"
            ),
            id="include_under",
        ),
    ],
)
def test_bucket(alltypes, df, func, expected_func):
    expr = func(alltypes.double_col)
    result = expr.execute()
    expected = expected_func(df.double_col)

    tm.assert_series_equal(result, expected, check_names=False)


def test_category_label(alltypes, df):
    bins = [0, 10, 25, 50, 100]
    labels = ["a", "b", "c", "d"]
    expr = alltypes.double_col.bucket(bins).label(labels)
    result = expr.execute()
    result = pd.Series(pd.Categorical(result, ordered=True))

    result.name = "double_col"

    expected = pd.cut(df.double_col, bins, labels=labels, right=False)
    tm.assert_series_equal(result, expected)


@pytest.mark.xfail(
    parse(sqlite3.sqlite_version) < parse("3.8.3"),
    raises=sa.exc.OperationalError,
    reason="SQLite versions < 3.8.3 do not support the WITH statement",
)
def test_union(alltypes):
    t = alltypes

    expr = (
        t.group_by("string_col")
        .aggregate(t.double_col.sum().name("foo"))
        .order_by("string_col")
    )

    t1 = expr.limit(4)
    t2 = expr.limit(4, offset=4)
    t3 = expr.limit(8)

    result = t1.union(t2).execute()
    expected = t3.execute()

    assert (result.string_col == expected.string_col).all()


@pytest.mark.parametrize(
    ("func", "expected_func"),
    [
        (
            lambda t, cond: t.bool_col.count(),
            lambda df, cond: df.bool_col.count(),
        ),
        (lambda t, cond: t.bool_col.any(), lambda df, cond: df.bool_col.any()),
        (lambda t, cond: t.bool_col.all(), lambda df, cond: df.bool_col.all()),
        (
            lambda t, cond: t.bool_col.notany(),
            lambda df, cond: ~df.bool_col.any(),
        ),
        (
            lambda t, cond: t.bool_col.notall(),
            lambda df, cond: ~df.bool_col.all(),
        ),
        (
            lambda t, cond: t.double_col.sum(),
            lambda df, cond: df.double_col.sum(),
        ),
        (
            lambda t, cond: t.double_col.mean(),
            lambda df, cond: df.double_col.mean(),
        ),
        (
            lambda t, cond: t.double_col.min(),
            lambda df, cond: df.double_col.min(),
        ),
        (
            lambda t, cond: t.double_col.max(),
            lambda df, cond: df.double_col.max(),
        ),
        (
            lambda t, cond: t.double_col.var(),
            lambda df, cond: df.double_col.var(),
        ),
        (
            lambda t, cond: t.double_col.std(),
            lambda df, cond: df.double_col.std(),
        ),
        (
            lambda t, cond: t.double_col.var(how="sample"),
            lambda df, cond: df.double_col.var(ddof=1),
        ),
        (
            lambda t, cond: t.double_col.std(how="pop"),
            lambda df, cond: df.double_col.std(ddof=0),
        ),
        (
            lambda t, cond: t.bool_col.count(where=cond),
            lambda df, cond: df.bool_col[cond].count(),
        ),
        (
            lambda t, cond: t.double_col.sum(where=cond),
            lambda df, cond: df.double_col[cond].sum(),
        ),
        (
            lambda t, cond: t.double_col.mean(where=cond),
            lambda df, cond: df.double_col[cond].mean(),
        ),
        (
            lambda t, cond: t.double_col.min(where=cond),
            lambda df, cond: df.double_col[cond].min(),
        ),
        (
            lambda t, cond: t.double_col.max(where=cond),
            lambda df, cond: df.double_col[cond].max(),
        ),
        (
            lambda t, cond: t.double_col.var(where=cond),
            lambda df, cond: df.double_col[cond].var(),
        ),
        (
            lambda t, cond: t.double_col.std(where=cond),
            lambda df, cond: df.double_col[cond].std(),
        ),
        (
            lambda t, cond: t.double_col.var(where=cond, how="sample"),
            lambda df, cond: df.double_col[cond].var(),
        ),
        (
            lambda t, cond: t.double_col.std(where=cond, how="pop"),
            lambda df, cond: df.double_col[cond].std(ddof=0),
        ),
    ],
)
def test_aggregations_execute(alltypes, func, df, expected_func):
    cond = alltypes.string_col.isin(["1", "7"])
    expr = func(alltypes, cond)
    result = expr.execute()
    expected = expected_func(df, df.string_col.isin(["1", "7"]))

    np.testing.assert_allclose(result, expected)


def test_not_contains(alltypes, df):
    n = 100
    table = alltypes.limit(n)
    expr = table.string_col.notin(["1", "7"])
    result = expr.execute()
    expected = ~df.head(n).string_col.isin(["1", "7"])
    tm.assert_series_equal(result, expected, check_names=False)


def test_distinct_aggregates(alltypes, df):
    expr = alltypes.double_col.nunique()
    result = expr.execute()
    expected = df.double_col.nunique()
    assert result == expected


def test_not_exists_works(alltypes):
    t = alltypes
    t2 = t.view()

    expr = t[-((t.string_col == t2.string_col).any())]
    expr.execute()


def test_interactive_repr_shows_error(alltypes):
    # #591. Doing this in SQLite because so many built-in functions are not
    # available

    expr = alltypes.double_col.approx_median()

    with config.option_context("interactive", True):
        result = repr(expr)
        assert "no translation rule" in result.lower()


def test_subquery(alltypes, df):
    t = alltypes

    expr = t.mutate(d=t.double_col.fillna(0)).limit(1000).group_by("string_col").size()
    result = expr.execute()
    expected = (
        df.assign(d=df.double_col.fillna(0))
        .head(1000)
        .groupby("string_col")
        .size()
        .reset_index()
        .rename(columns={0: "CountStar()"})
    )
    tm.assert_frame_equal(result, expected)


def test_filter(alltypes, df):
    expr = alltypes.filter(alltypes.year == 2010).float_col
    result = expr.execute().squeeze().reset_index(drop=True)
    expected = df.query("year == 2010").float_col
    assert len(result) == len(expected)


@pytest.mark.parametrize("column", [lambda t: "float_col", lambda t: t["float_col"]])
def test_column_access_after_sort(alltypes, df, column):
    expr = alltypes.order_by(column(alltypes)).head(10).string_col
    result = expr.execute()
    expected = df.sort_values("float_col").string_col.head(10).reset_index(drop=True)
    tm.assert_series_equal(result, expected)


@pytest.fixture
def mj1(con, temp_table):
    return con.create_table(
        temp_table,
        pd.DataFrame(dict(id1=[1, 2], val1=[10.0, 20.0])),
        schema=ibis.schema(dict(id1="int64", val1="float64")),
    )


@pytest.fixture
def mj2(con, temp_table_orig):
    return con.create_table(
        temp_table_orig,
        pd.DataFrame(dict(id2=[1, 2], val2=[15.0, 25.0])),
        schema=ibis.schema(dict(id2="int64", val2="float64")),
    )


def test_simple_join(mj1, mj2):
    joined = mj1.join(mj2, mj1.id1 == mj2.id2)
    result = joined.val2.execute()
    assert len(result) == 2


def test_anonymous_aggregate(alltypes, df):
    expr = alltypes[alltypes.double_col > alltypes.double_col.mean()]
    result = expr.execute()
    expected = df[df.double_col > df.double_col.mean()].reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


def test_head(alltypes):
    t = alltypes
    result = t.head().execute()
    expected = t.limit(5).execute()
    tm.assert_frame_equal(result, expected)


def test_identical_to(alltypes):
    t = alltypes
    dt = t[["tinyint_col", "double_col"]].execute()
    expr = t.tinyint_col.identical_to(t.double_col)
    result = expr.execute()
    expected = (dt.tinyint_col.isnull() & dt.double_col.isnull()) | (
        dt.tinyint_col == dt.double_col
    )
    expected.name = result.name
    tm.assert_series_equal(result, expected)


@pytest.mark.xfail(
    raises=AttributeError,
    reason="truncate method is not yet implemented",
)
def test_truncate_method(con, alltypes):
    expr = alltypes.limit(5)
    name = str(uuid.uuid4())
    t = con.create_table(name, expr)
    assert len(t.execute()) == 5
    t.truncate()
    assert len(t.execute()) == 0


def test_truncate_from_connection(con, alltypes):
    expr = alltypes.limit(5)
    name = str(uuid.uuid4())
    t = con.create_table(name, expr)
    assert len(t.execute()) == 5
    con.truncate_table(name)
    assert len(t.execute()) == 0


def test_not(alltypes):
    t = alltypes.limit(10)
    expr = t.select([(~t.double_col.isnull()).name("double_col")])
    result = expr.execute().double_col
    expected = ~t.execute().double_col.isnull()
    tm.assert_series_equal(result, expected)


def test_compile_with_named_table():
    t = ibis.table([("a", "string")], name="t")
    result = ibis.sqlite.compile(t.a)
    st = sa.table("t", sa.column("a", sa.String)).alias("t0")
    assert str(result) == str(sa.select(st.c.a))


def test_compile_with_unnamed_table():
    t = ibis.table([("a", "string")])
    result = ibis.sqlite.compile(t.a)
    st = sa.table(t.op().name, sa.column("a", sa.String)).alias("t0")
    assert str(result) == str(sa.select(st.c.a))


def test_compile_with_multiple_unnamed_tables():
    t = ibis.table([("a", "string")])
    s = ibis.table([("b", "string")])
    join = t.join(s, t.a == s.b)
    result = ibis.sqlite.compile(join)
    sqla_t = sa.table(t.op().name, sa.column("a", sa.String)).alias("t0")
    sqla_s = sa.table(s.op().name, sa.column("b", sa.String)).alias("t1")
    sqla_join = sqla_t.join(sqla_s, sqla_t.c.a == sqla_s.c.b)
    expected = sa.select(sqla_t.c.a, sqla_s.c.b).select_from(sqla_join)
    assert str(result) == str(expected)


def test_compile_with_one_unnamed_table():
    t = ibis.table([("a", "string")])
    s = ibis.table([("b", "string")], name="s")
    join = t.join(s, t.a == s.b)
    result = ibis.sqlite.compile(join)
    sqla_t = sa.table(t.op().name, sa.column("a", sa.String)).alias("t0")
    sqla_s = sa.table("s", sa.column("b", sa.String)).alias("t1")
    sqla_join = sqla_t.join(sqla_s, sqla_t.c.a == sqla_s.c.b)
    expected = sa.select(sqla_t.c.a, sqla_s.c.b).select_from(sqla_join)
    assert str(result) == str(expected)


def test_scalar_parameter(alltypes):
    start_string, end_string = "2009-03-01", "2010-07-03"

    start = ibis.param(dt.date)
    end = ibis.param(dt.date)
    t = alltypes
    col = t.date_string_col.cast("date")
    expr = col.between(start, end).name("result")
    result = expr.execute(params={start: start_string, end: end_string})

    expected_expr = col.between(start_string, end_string).name("result")
    expected = expected_expr.execute()
    tm.assert_series_equal(result, expected)


def test_count_on_order_by(con, snapshot):
    t = con.table("batting")
    sort_key = ibis.desc(t.playerID)
    sorted_table = t.order_by(sort_key)
    expr = sorted_table.count()
    result = str(ibis.to_sql(expr, dialect="sqlite"))
    snapshot.assert_match(result, "out.sql")


def test_memtable_compilation(con):
    expr = ibis.memtable({"a": [1, 2, 3]})
    t = con.compile(expr)
    assert t.exported_columns[0].name == "a"
