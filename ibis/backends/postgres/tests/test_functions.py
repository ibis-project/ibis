from __future__ import annotations

import operator
import string
import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
from pytest import param

import ibis
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis import config
from ibis import literal as L
from ibis.backends.conftest import WINDOWS

pytest.importorskip("psycopg2")
sa = pytest.importorskip("sqlalchemy")

from sqlalchemy.dialects import postgresql  # noqa: E402


@pytest.mark.parametrize(
    ("left_func", "right_func"),
    [
        param(
            lambda t: t.double_col.cast("int8"),
            lambda at: sa.cast(at.c.double_col, sa.SMALLINT),
            id="double_to_int8",
        ),
        param(
            lambda t: t.double_col.cast("int16"),
            lambda at: sa.cast(at.c.double_col, sa.SMALLINT),
            id="double_to_int16",
        ),
        param(
            lambda t: t.string_col.cast("double"),
            lambda at: sa.cast(at.c.string_col, postgresql.DOUBLE_PRECISION),
            id="string_to_double",
        ),
        param(
            lambda t: t.string_col.cast("float32"),
            lambda at: sa.cast(at.c.string_col, postgresql.REAL),
            id="string_to_float",
        ),
        param(
            lambda t: t.string_col.cast("decimal"),
            lambda at: sa.cast(at.c.string_col, sa.NUMERIC()),
            id="string_to_decimal_no_params",
        ),
        param(
            lambda t: t.string_col.cast("decimal(9, 3)"),
            lambda at: sa.cast(at.c.string_col, sa.NUMERIC(9, 3)),
            id="string_to_decimal_params",
        ),
    ],
)
def test_cast(alltypes, alltypes_sqla, translate, left_func, right_func):
    left = left_func(alltypes)
    right = right_func(alltypes_sqla.alias("t0"))
    assert str(translate(left.op()).compile()) == str(right.compile())


def test_date_cast(alltypes, alltypes_sqla, translate):
    result = alltypes.date_string_col.cast("date")
    expected = sa.cast(alltypes_sqla.alias("t0").c.date_string_col, sa.DATE)
    assert str(translate(result.op())) == str(expected)


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
def test_noop_cast(alltypes, alltypes_sqla, translate, column):
    col = alltypes[column]
    result = col.cast(col.type())
    expected = alltypes_sqla.alias("t0").c[column]
    assert result.equals(col)
    assert str(translate(result.op())) == str(expected)


def test_timestamp_cast_noop(alltypes, alltypes_sqla, translate):
    # See GH #592
    result1 = alltypes.timestamp_col.cast("timestamp")
    result2 = alltypes.int_col.cast("timestamp")

    assert isinstance(result1, ir.TimestampColumn)
    assert isinstance(result2, ir.TimestampColumn)

    expected1 = alltypes_sqla.alias("t0").c.timestamp_col
    expected2 = sa.cast(
        sa.func.to_timestamp(alltypes_sqla.alias("t0").c.int_col), sa.TIMESTAMP()
    )

    assert str(translate(result1.op())) == str(expected1)
    assert str(translate(result2.op())) == str(expected2)


@pytest.mark.parametrize(
    "pattern",
    [
        # there could be pathological failure at midnight somewhere, but
        # that's okay
        "%Y%m%d %H",
        # test quoting behavior
        'DD BAR %w FOO "DD"',
        'DD BAR %w FOO "D',
        'DD BAR "%w" FOO "D',
        'DD BAR "%d" FOO "D',
        param(
            'DD BAR "%c" FOO "D',
            marks=pytest.mark.xfail(
                condition=WINDOWS,
                raises=exc.UnsupportedOperationError,
                reason="Locale-specific format specs not available on Windows",
            ),
        ),
        param(
            'DD BAR "%x" FOO "D',
            marks=pytest.mark.xfail(
                condition=WINDOWS,
                raises=exc.UnsupportedOperationError,
                reason="Locale-specific format specs not available on Windows",
            ),
        ),
        param(
            'DD BAR "%X" FOO "D',
            marks=pytest.mark.xfail(
                condition=WINDOWS,
                raises=exc.UnsupportedOperationError,
                reason="Locale-specific format specs not available on Windows",
            ),
        ),
    ],
)
def test_strftime(con, pattern):
    value = ibis.timestamp("2015-09-01 14:48:05.359")
    raw_value = datetime(
        year=2015,
        month=9,
        day=1,
        hour=14,
        minute=48,
        second=5,
        microsecond=359000,
    )
    assert con.execute(value.strftime(pattern)) == raw_value.strftime(pattern)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        param(L("foo_bar"), "text", id="text"),
        param(L(5), "integer", id="integer"),
        param(ibis.NA, "null", id="null"),
        # TODO(phillipc): should this really be double?
        param(L(1.2345), "numeric", id="numeric"),
        param(
            L(
                datetime(
                    2015,
                    9,
                    1,
                    hour=14,
                    minute=48,
                    second=5,
                    microsecond=359000,
                )
            ),
            "timestamp without time zone",
            id="timestamp_without_time_zone",
        ),
        param(L(date(2015, 9, 1)), "date", id="date"),
    ],
)
def test_typeof(con, value, expected):
    assert con.execute(value.typeof()) == expected


@pytest.mark.parametrize(("value", "expected"), [(0, None), (5.5, 5.5)])
def test_nullifzero(con, value, expected):
    with pytest.warns(FutureWarning):
        assert con.execute(L(value).nullifzero()) == expected


@pytest.mark.parametrize(("value", "expected"), [("foo_bar", 7), ("", 0)])
def test_string_length(con, value, expected):
    assert con.execute(L(value).length()) == expected


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        param(operator.methodcaller("left", 3), "foo", id="left"),
        param(operator.methodcaller("right", 3), "bar", id="right"),
        param(operator.methodcaller("substr", 0, 3), "foo", id="substr_0_3"),
        param(operator.methodcaller("substr", 4, 3), "bar", id="substr_4, 3"),
        param(operator.methodcaller("substr", 1), "oo_bar", id="substr_1"),
    ],
)
def test_string_substring(con, op, expected):
    value = L("foo_bar")
    assert con.execute(op(value)) == expected


@pytest.mark.parametrize(
    ("opname", "expected"),
    [("lstrip", "foo   "), ("rstrip", "   foo"), ("strip", "foo")],
)
def test_string_strip(con, opname, expected):
    op = operator.methodcaller(opname)
    value = L("   foo   ")
    assert con.execute(op(value)) == expected


@pytest.mark.parametrize(
    ("opname", "count", "char", "expected"),
    [("lpad", 6, " ", "   foo"), ("rpad", 6, " ", "foo   ")],
)
def test_string_pad(con, opname, count, char, expected):
    op = operator.methodcaller(opname, count, char)
    value = L("foo")
    assert con.execute(op(value)) == expected


def test_string_reverse(con):
    assert con.execute(L("foo").reverse()) == "oof"


def test_string_upper(con):
    assert con.execute(L("foo").upper()) == "FOO"


def test_string_lower(con):
    assert con.execute(L("FOO").lower()) == "foo"


@pytest.mark.parametrize(
    ("haystack", "needle", "expected"),
    [
        ("foobar", "bar", True),
        ("foobar", "foo", True),
        ("foobar", "baz", False),
        ("100%", "%", True),
        ("a_b_c", "_", True),
    ],
)
def test_string_contains(con, haystack, needle, expected):
    value = L(haystack)
    expr = value.contains(needle)
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("foo bar foo", "Foo Bar Foo"), ("foobar Foo", "Foobar Foo")],
)
def test_capitalize(con, value, expected):
    assert con.execute(L(value).capitalize()) == expected


def test_repeat(con):
    expr = L("bar ").repeat(3)
    assert con.execute(expr) == "bar bar bar "


def test_re_replace(con):
    expr = L("fudge|||chocolate||candy").re_replace("\\|{2,3}", ", ")
    assert con.execute(expr) == "fudge, chocolate, candy"


def test_translate(con):
    expr = L("faab").translate("a", "b")
    assert con.execute(expr) == "fbbb"


@pytest.mark.parametrize(
    ("raw_value", "expected"), [("a", 0), ("b", 1), ("d", -1), (None, 3)]
)
def test_find_in_set(con, raw_value, expected):
    value = L(raw_value, dt.string)
    haystack = ["a", "b", "c", None]
    expr = value.find_in_set(haystack)
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("raw_value", "opname", "expected"),
    [
        (None, "isnull", True),
        (1, "isnull", False),
        (None, "notnull", False),
        (1, "notnull", True),
    ],
)
def test_isnull_notnull(con, raw_value, opname, expected):
    lit = L(raw_value)
    op = operator.methodcaller(opname)
    expr = op(lit)
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(L("foobar").find("bar"), 3, id="find_pos"),
        param(L("foobar").find("baz"), -1, id="find_neg"),
        param(L("foobar").like("%bar"), True, id="like_left_pattern"),
        param(L("foobar").like("foo%"), True, id="like_right_pattern"),
        param(L("foobar").like("%baz%"), False, id="like_both_sides_pattern"),
        param(L("foobar").like(["%bar"]), True, id="like_list_left_side"),
        param(L("foobar").like(["foo%"]), True, id="like_list_right_side"),
        param(L("foobar").like(["%baz%"]), False, id="like_list_both_sides"),
        param(L("foobar").like(["%bar", "foo%"]), True, id="like_list_multiple"),
        param(L("foobarfoo").replace("foo", "H"), "HbarH", id="replace"),
        param(L("a").ascii_str(), ord("a"), id="ascii_str"),
    ],
)
def test_string_functions(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(L("abcd").re_search("[a-z]"), True, id="re_search_match"),
        param(L("abcd").re_search(r"[\d]+"), False, id="re_search_no_match"),
        param(L("1222").re_search(r"[\d]+"), True, id="re_search_match_number"),
    ],
)
def test_regexp(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(L("abcd").re_extract("([a-z]+)", 0), "abcd", id="re_extract_whole"),
        param(L("abcd").re_extract("(ab)(cd)", 1), "ab", id="re_extract_first"),
        param(L("abcd").re_extract("(ab)(cd)", 2), "cd", id="re_extract_second"),
        # valid group number but no match => empty string
        param(L("abcd").re_extract(r"(\d)", 0), None, id="re_extract_no_match"),
        # match but not a valid group number => NULL
        param(L("abcd").re_extract("abcd", 3), None, id="re_extract_match"),
    ],
)
def test_regexp_extract(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(ibis.NA.fillna(5), 5, id="filled"),
        param(L(5).fillna(10), 5, id="not_filled"),
        param(L(5).nullif(5), None, id="nullif_null"),
        param(L(10).nullif(5), 10, id="nullif_not_null"),
    ],
)
def test_fillna_nullif(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(ibis.coalesce(5, None, 4), 5, id="first"),
        param(ibis.coalesce(ibis.NA, 4, ibis.NA), 4, id="second"),
        param(ibis.coalesce(ibis.NA, ibis.NA, 3.14), 3.14, id="third"),
    ],
)
def test_coalesce(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(ibis.coalesce(ibis.NA, ibis.NA), None, id="all_null"),
        param(
            ibis.coalesce(
                ibis.NA.cast("int8"),
                ibis.NA.cast("int8"),
                ibis.NA.cast("int8"),
            ),
            None,
            id="all_nulls_with_all_cast",
        ),
    ],
)
def test_coalesce_all_na(con, expr, expected):
    assert con.execute(expr) is None


def test_coalesce_all_na_double(con):
    expr = ibis.coalesce(ibis.NA, ibis.NA, ibis.NA.cast("double"))
    assert np.isnan(con.execute(expr))


def test_numeric_builtins_work(alltypes, df):
    expr = alltypes.double_col.fillna(0)
    result = expr.execute()
    expected = df.double_col.fillna(0)
    expected.name = "Coalesce()"
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("op", "pandas_op"),
    [
        param(
            lambda t: (t.double_col > 20).ifelse(10, -20),
            lambda df: pd.Series(np.where(df.double_col > 20, 10, -20), dtype="int8"),
            id="simple",
        ),
        param(
            lambda t: (t.double_col > 20).ifelse(10, -20).abs(),
            lambda df: pd.Series(
                np.where(df.double_col > 20, 10, -20), dtype="int8"
            ).abs(),
            id="abs",
        ),
    ],
)
def test_ifelse(alltypes, df, op, pandas_op):
    expr = op(alltypes)
    result = expr.execute()
    result.name = None
    expected = pandas_op(df)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("func", "pandas_func"),
    [
        # tier and histogram
        param(
            lambda d: d.bucket([0, 10, 25, 50, 100]),
            lambda s: pd.cut(s, [0, 10, 25, 50, 100], right=False, labels=False).astype(
                "int8"
            ),
            id="include_over_false",
        ),
        param(
            lambda d: d.bucket([0, 10, 25, 50], include_over=True),
            lambda s: pd.cut(
                s, [0, 10, 25, 50, np.inf], right=False, labels=False
            ).astype("int8"),
            id="include_over_true",
        ),
        param(
            lambda d: d.bucket([0, 10, 25, 50], close_extreme=False),
            lambda s: pd.cut(s, [0, 10, 25, 50], right=False, labels=False),
            id="close_extreme_false",
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
            id="closed_right",
        ),
        param(
            lambda d: d.bucket([10, 25, 50, 100], include_under=True),
            lambda s: pd.cut(s, [0, 10, 25, 50, 100], right=False, labels=False).astype(
                "int8"
            ),
            id="include_under_true",
        ),
    ],
)
def test_bucket(alltypes, df, func, pandas_func):
    expr = func(alltypes.double_col)
    result = expr.execute()
    expected = pandas_func(df.double_col)
    tm.assert_series_equal(result, expected, check_names=False)


def test_category_label(alltypes, df):
    t = alltypes
    d = t.double_col

    bins = [0, 10, 25, 50, 100]
    labels = ["a", "b", "c", "d"]
    bucket = d.bucket(bins)
    expr = bucket.label(labels)
    result = expr.execute()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    result = pd.Series(pd.Categorical(result, ordered=True))

    result.name = "double_col"

    expected = pd.cut(df.double_col, bins, labels=labels, right=False)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("distinct", [True, False])
def test_union_cte(alltypes, distinct, snapshot):
    t = alltypes
    expr1 = t.group_by(t.string_col).aggregate(metric=t.double_col.sum())
    expr2 = expr1.view()
    expr3 = expr1.view()
    expr = expr1.union(expr2, distinct=distinct).union(expr3, distinct=distinct)
    result = " ".join(
        line.strip()
        for line in str(
            expr.compile().compile(compile_kwargs={"literal_binds": True})
        ).splitlines()
    )
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize(
    ("func", "pandas_func"),
    [
        param(
            lambda t, cond: t.bool_col.count(),
            lambda df, cond: df.bool_col.count(),
            id="count",
        ),
        param(
            lambda t, cond: t.bool_col.any(),
            lambda df, cond: df.bool_col.any(),
            id="any",
        ),
        param(
            lambda t, cond: t.bool_col.all(),
            lambda df, cond: df.bool_col.all(),
            id="all",
        ),
        param(
            lambda t, cond: t.bool_col.notany(),
            lambda df, cond: ~df.bool_col.any(),
            id="notany",
        ),
        param(
            lambda t, cond: t.bool_col.notall(),
            lambda df, cond: ~df.bool_col.all(),
            id="notall",
        ),
        param(
            lambda t, cond: t.double_col.sum(),
            lambda df, cond: df.double_col.sum(),
            id="sum",
        ),
        param(
            lambda t, cond: t.double_col.mean(),
            lambda df, cond: df.double_col.mean(),
            id="mean",
        ),
        param(
            lambda t, cond: t.double_col.min(),
            lambda df, cond: df.double_col.min(),
            id="min",
        ),
        param(
            lambda t, cond: t.double_col.max(),
            lambda df, cond: df.double_col.max(),
            id="max",
        ),
        param(
            lambda t, cond: t.double_col.var(),
            lambda df, cond: df.double_col.var(),
            id="var",
        ),
        param(
            lambda t, cond: t.double_col.std(),
            lambda df, cond: df.double_col.std(),
            id="std",
        ),
        param(
            lambda t, cond: t.double_col.var(how="sample"),
            lambda df, cond: df.double_col.var(ddof=1),
            id="samp_var",
        ),
        param(
            lambda t, cond: t.double_col.std(how="pop"),
            lambda df, cond: df.double_col.std(ddof=0),
            id="pop_std",
        ),
        param(
            lambda t, cond: t.bool_col.count(where=cond),
            lambda df, cond: df.bool_col[cond].count(),
            id="count_where",
        ),
        param(
            lambda t, cond: t.double_col.sum(where=cond),
            lambda df, cond: df.double_col[cond].sum(),
            id="sum_where",
        ),
        param(
            lambda t, cond: t.double_col.mean(where=cond),
            lambda df, cond: df.double_col[cond].mean(),
            id="mean_where",
        ),
        param(
            lambda t, cond: t.double_col.min(where=cond),
            lambda df, cond: df.double_col[cond].min(),
            id="min_where",
        ),
        param(
            lambda t, cond: t.double_col.max(where=cond),
            lambda df, cond: df.double_col[cond].max(),
            id="max_where",
        ),
        param(
            lambda t, cond: t.double_col.var(where=cond),
            lambda df, cond: df.double_col[cond].var(),
            id="var_where",
        ),
        param(
            lambda t, cond: t.double_col.std(where=cond),
            lambda df, cond: df.double_col[cond].std(),
            id="std_where",
        ),
        param(
            lambda t, cond: t.double_col.var(where=cond, how="sample"),
            lambda df, cond: df.double_col[cond].var(),
            id="samp_var_where",
        ),
        param(
            lambda t, cond: t.double_col.std(where=cond, how="pop"),
            lambda df, cond: df.double_col[cond].std(ddof=0),
            id="pop_std_where",
        ),
    ],
)
def test_aggregations(alltypes, df, func, pandas_func):
    table = alltypes.limit(100)
    df = df.head(table.count().execute())

    cond = table.string_col.isin(["1", "7"])
    expr = func(table, cond)
    result = expr.execute()
    expected = pandas_func(df, cond.execute())

    np.testing.assert_allclose(result, expected)


def test_not_contains(alltypes, df):
    n = 100
    table = alltypes.limit(n)
    expr = table.string_col.notin(["1", "7"])
    result = expr.execute()
    expected = ~df.head(n).string_col.isin(["1", "7"])
    tm.assert_series_equal(result, expected, check_names=False)


def test_group_concat(alltypes, df):
    expr = alltypes.string_col.group_concat()
    result = expr.execute()
    expected = ",".join(df.string_col.dropna())
    assert result == expected


def test_distinct_aggregates(alltypes, df):
    expr = alltypes.limit(100).double_col.nunique()
    result = expr.execute()
    assert result == df.head(100).double_col.nunique()


def test_not_exists(alltypes, df):
    t = alltypes
    t2 = t.view()

    expr = t[~((t.string_col == t2.string_col).any())]
    result = expr.execute()

    left, right = df, t2.execute()
    expected = left[left.string_col != right.string_col]

    tm.assert_frame_equal(result, expected, check_index_type=False, check_dtype=False)


def test_interactive_repr_shows_error(alltypes):
    # #591. Doing this in PostgreSQL because so many built-in functions are
    # not available

    expr = alltypes.int_col.convert_base(10, 2)

    with config.option_context("interactive", True):
        result = repr(expr)

    assert "no translation rule" in result.lower()


def test_subquery(alltypes, df):
    t = alltypes

    expr = t.mutate(d=t.double_col.fillna(0)).limit(1000).group_by("string_col").size()
    result = expr.execute().sort_values("string_col").reset_index(drop=True)
    expected = (
        df.assign(d=df.double_col.fillna(0))
        .head(1000)
        .groupby("string_col")
        .string_col.count()
        .rename("CountStar()")
        .reset_index()
        .sort_values("string_col")
        .reset_index(drop=True)
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func", ["mean", "sum", "min", "max"])
def test_simple_window(alltypes, func, df):
    t = alltypes
    f = getattr(t.double_col, func)
    df_f = getattr(df.double_col, func)
    result = t.select((t.double_col - f()).name("double_col")).execute().double_col
    expected = df.double_col - df_f()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["mean", "sum", "min", "max"])
def test_rolling_window(alltypes, func, df):
    t = alltypes
    df = (
        df[["double_col", "timestamp_col"]]
        .sort_values("timestamp_col")
        .reset_index(drop=True)
    )
    window = ibis.window(order_by=t.timestamp_col, preceding=6, following=0)
    f = getattr(t.double_col, func)
    df_f = getattr(df.double_col.rolling(7, min_periods=0), func)
    result = t.select(f().over(window).name("double_col")).execute().double_col
    expected = df_f()
    tm.assert_series_equal(result, expected)


def test_rolling_window_with_mlb(alltypes):
    t = alltypes
    window = ibis.trailing_window(
        preceding=ibis.rows_with_max_lookback(3, ibis.interval(days=5)),
        order_by=t.timestamp_col,
    )
    expr = t["double_col"].sum().over(window)
    with pytest.raises(NotImplementedError):
        expr.execute()


@pytest.mark.parametrize("func", ["mean", "sum", "min", "max"])
def test_partitioned_window(alltypes, func, df):
    t = alltypes
    window = ibis.window(
        group_by=t.string_col,
        order_by=t.timestamp_col,
        preceding=6,
        following=0,
    )

    def roller(func):
        def rolled(df):
            torder = df.sort_values("timestamp_col")
            rolling = torder.double_col.rolling(7, min_periods=0)
            return getattr(rolling, func)()

        return rolled

    f = getattr(t.double_col, func)
    expr = f().over(window).name("double_col")
    result = t.select(expr).execute().double_col
    expected = df.groupby("string_col").apply(roller(func)).reset_index(drop=True)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["sum", "min", "max"])
def test_cumulative_simple_window(alltypes, func, df):
    t = alltypes
    f = getattr(t.double_col, func)
    col = t.double_col - f().over(ibis.cumulative_window())
    expr = t.select(col.name("double_col"))
    result = expr.execute().double_col
    expected = df.double_col - getattr(df.double_col, "cum%s" % func)()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["sum", "min", "max"])
def test_cumulative_partitioned_window(alltypes, func, df):
    t = alltypes
    df = df.sort_values("string_col").reset_index(drop=True)
    window = ibis.cumulative_window(group_by=t.string_col)
    f = getattr(t.double_col, func)
    expr = t.select((t.double_col - f().over(window)).name("double_col"))
    result = expr.execute().double_col
    expected = df.groupby(df.string_col).double_col.transform(
        lambda c: c - getattr(c, "cum%s" % func)()
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["sum", "min", "max"])
def test_cumulative_ordered_window(alltypes, func, df):
    t = alltypes
    df = df.sort_values("timestamp_col").reset_index(drop=True)
    window = ibis.cumulative_window(order_by=t.timestamp_col)
    f = getattr(t.double_col, func)
    expr = t.select((t.double_col - f().over(window)).name("double_col"))
    result = expr.execute().double_col
    expected = df.double_col - getattr(df.double_col, "cum%s" % func)()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["sum", "min", "max"])
def test_cumulative_partitioned_ordered_window(alltypes, func, df):
    t = alltypes
    df = df.sort_values(["string_col", "timestamp_col"]).reset_index(drop=True)
    window = ibis.cumulative_window(order_by=t.timestamp_col, group_by=t.string_col)
    f = getattr(t.double_col, func)
    expr = t.select((t.double_col - f().over(window)).name("double_col"))
    result = expr.execute().double_col
    method = operator.methodcaller(f"cum{func}")
    expected = df.groupby(df.string_col).double_col.transform(lambda c: c - method(c))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("func", "shift_amount"), [("lead", -1), ("lag", 1)], ids=["lead", "lag"]
)
def test_analytic_shift_functions(alltypes, df, func, shift_amount):
    method = getattr(alltypes.double_col, func)
    expr = method(1)
    result = expr.execute().rename("double_col")
    expected = df.double_col.shift(shift_amount)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("func", "expected_index"), [("first", -1), ("last", 0)], ids=["first", "last"]
)
def test_first_last_value(alltypes, df, func, expected_index):
    col = alltypes.order_by(ibis.desc(alltypes.string_col)).double_col
    method = getattr(col, func)
    # test that we traverse into expression trees
    expr = (1 + method()) - 1
    result = expr.execute()
    expected = df.double_col.iloc[expected_index]
    assert result == expected


def test_null_column(alltypes):
    t = alltypes
    nrows = t.count().execute()
    expr = t.mutate(na_column=ibis.NA).na_column
    result = expr.execute()
    tm.assert_series_equal(result, pd.Series([None] * nrows, name="na_column"))


def test_null_column_union(alltypes, df):
    t = alltypes
    s = alltypes[["double_col"]].mutate(string_col=ibis.NA.cast("string"))
    expr = t[["double_col", "string_col"]].union(s)
    result = expr.execute()
    nrows = t.count().execute()
    expected = pd.concat(
        [
            df[["double_col", "string_col"]],
            pd.concat(
                [
                    df[["double_col"]],
                    pd.DataFrame({"string_col": [None] * nrows}),
                ],
                axis=1,
            ),
        ],
        axis=0,
        ignore_index=True,
    )
    tm.assert_frame_equal(result, expected)


def test_window_with_arithmetic(alltypes, df):
    t = alltypes
    w = ibis.window(order_by=t.timestamp_col)
    expr = t.mutate(new_col=ibis.row_number().over(w) / 2)

    df = df[["timestamp_col"]].sort_values("timestamp_col").reset_index(drop=True)
    expected = df.assign(new_col=[x / 2.0 for x in range(len(df))])
    result = expr["timestamp_col", "new_col"].execute()
    tm.assert_frame_equal(result, expected)


def test_anonymous_aggregate(alltypes, df):
    t = alltypes
    expr = t[t.double_col > t.double_col.mean()]
    result = expr.execute()
    expected = df[df.double_col > df.double_col.mean()].reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


@pytest.fixture
def array_types(con):
    return con.table("array_types")


def test_array_length(array_types):
    expr = array_types.select(
        array_types.x.length().name("x_length"),
        array_types.y.length().name("y_length"),
        array_types.z.length().name("z_length"),
    )
    result = expr.execute()
    expected = pd.DataFrame(
        {
            "x_length": [3, 2, 2, 3, 3, 4],
            "y_length": [3, 2, 2, 3, 3, 4],
            "z_length": [3, 2, 2, 0, None, 4],
        }
    )

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("column", "value_type"),
    [("x", dt.int64), ("y", dt.string), ("z", dt.double)],
)
def test_array_schema(array_types, column, value_type):
    assert array_types[column].type() == dt.Array(value_type)


def test_array_collect(array_types):
    expr = array_types.group_by(array_types.grouper).aggregate(
        collected=lambda t: t.scalar_column.collect()
    )
    result = expr.execute().sort_values("grouper").reset_index(drop=True)
    expected = pd.DataFrame(
        {
            "grouper": list("abc"),
            "collected": [[1.0, 2.0, 3.0], [4.0, 5.0], [6.0]],
        }
    )[["grouper", "collected"]]
    tm.assert_frame_equal(result, expected, check_column_type=False)


@pytest.mark.parametrize("index", [0, 1, 3, 4, 11, -1, -3, -4, -11])
def test_array_index(array_types, index):
    expr = array_types[array_types.y[index].name("indexed")]
    result = expr.execute()
    expected = pd.DataFrame(
        {
            "indexed": array_types.y.execute().map(
                lambda x: x[index] if -len(x) <= index < len(x) else None
            )
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("n", [1, 3, 4, 7, -2])
@pytest.mark.parametrize(
    "mul",
    [
        param(lambda x, n: x * n, id="mul"),
        param(lambda x, n: n * x, id="rmul"),
    ],
)
def test_array_repeat(array_types, n, mul):
    expr = array_types.select(mul(array_types.x, n).name("repeated"))
    result = expr.execute()
    expected = pd.DataFrame(
        {"repeated": array_types.x.execute().map(lambda x, n=n: mul(x, n))}
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "catop",
    [
        param(lambda x, y: x + y, id="concat"),
        param(lambda x, y: y + x, id="rconcat"),
    ],
)
def test_array_concat(array_types, catop):
    t = array_types
    x, y = t.x.cast("array<string>").name("x"), t.y
    expr = t.select(catop(x, y).name("catted"))
    result = expr.execute()
    tuples = t.select(x, y).execute().itertuples(index=False)
    expected = pd.DataFrame({"catted": [catop(i, j) for i, j in tuples]})
    tm.assert_frame_equal(result, expected)


def test_array_concat_mixed_types(array_types):
    with pytest.raises(TypeError):
        array_types.y + array_types.x.cast("array<double>")


@pytest.fixture
def t(con, temp_table):
    with con.begin() as c:
        c.exec_driver_sql(
            f"CREATE TABLE {con._quote(temp_table)} (id SERIAL PRIMARY KEY, name TEXT)"
        )
    return con.table(temp_table)


@pytest.fixture
def s(con, t, temp_table2):
    temp_table = t.op().name
    assert temp_table != temp_table2

    with con.begin() as c:
        c.exec_driver_sql(
            f"""
            CREATE TABLE {con._quote(temp_table2)} (
              id SERIAL PRIMARY KEY,
              left_t_id INTEGER REFERENCES {con._quote(temp_table)},
              cost DOUBLE PRECISION
            )
            """
        )
    return con.table(temp_table2)


@pytest.fixture
def trunc(con, temp_table):
    quoted = con._quote(temp_table)
    with con.begin() as c:
        c.exec_driver_sql(f"CREATE TABLE {quoted} (id SERIAL PRIMARY KEY, name TEXT)")
        c.exec_driver_sql(f"INSERT INTO {quoted} (name) VALUES ('a'), ('b'), ('c')")
    return con.table(temp_table)


def test_semi_join(con, t, s):
    t_a = con._get_sqla_table(t.op().name).alias("t0")
    s_a = con._get_sqla_table(s.op().name).alias("t1")

    expr = t.semi_join(s, t.id == s.id)
    result = expr.compile().compile(compile_kwargs={"literal_binds": True})
    base = (
        sa.select(t_a.c.id, t_a.c.name)
        .where(sa.exists(sa.select(1).where(t_a.c.id == s_a.c.id)))
        .subquery()
    )
    expected = sa.select(base.c.id, base.c.name)
    assert str(result) == str(expected)


def test_anti_join(con, t, s):
    t_a = con._get_sqla_table(t.op().name).alias("t0")
    s_a = con._get_sqla_table(s.op().name).alias("t1")

    expr = t.anti_join(s, t.id == s.id)
    result = expr.compile().compile(compile_kwargs={"literal_binds": True})
    base = (
        sa.select(t_a.c.id, t_a.c.name)
        .where(~sa.exists(sa.select(1).where(t_a.c.id == s_a.c.id)))
        .subquery()
    )
    expected = sa.select(base.c.id, base.c.name)
    assert str(result) == str(expected)


def test_create_table_from_expr(con, trunc, temp_table2):
    con.create_table(temp_table2, obj=trunc)
    t = con.table(temp_table2)
    assert list(t["name"].execute()) == list("abc")


def test_truncate_table(con, trunc):
    assert list(trunc["name"].execute()) == list("abc")
    con.truncate_table(trunc.op().name)
    assert not len(trunc.execute())


def test_head(con):
    t = con.table("functional_alltypes")
    result = t.head().execute()
    expected = t.limit(5).execute()
    tm.assert_frame_equal(result, expected)


def test_identical_to(con, df):
    # TODO: abstract this testing logic out into parameterized fixtures
    t = con.table("functional_alltypes")
    dt = df[["tinyint_col", "double_col"]]
    expr = t.tinyint_col.identical_to(t.double_col)
    result = expr.execute()
    expected = (dt.tinyint_col.isnull() & dt.double_col.isnull()) | (
        dt.tinyint_col == dt.double_col
    )
    expected.name = result.name
    tm.assert_series_equal(result, expected)


def test_analytic_functions(alltypes, snapshot):
    expr = alltypes.select(
        rank=alltypes.double_col.rank(),
        dense_rank=alltypes.double_col.dense_rank(),
        cume_dist=alltypes.double_col.cume_dist(),
        ntile=alltypes.double_col.ntile(7),
        percent_rank=alltypes.double_col.percent_rank(),
    )
    snapshot.assert_match(str(ibis.to_sql(expr)), "out.sql")


@pytest.mark.parametrize("opname", ["invert", "neg"])
def test_not_and_negate_bool(con, opname, df):
    op = getattr(operator, opname)
    t = con.table("functional_alltypes").limit(10)
    expr = t.select(op(t.bool_col).name("bool_col"))
    result = expr.execute().bool_col
    expected = op(df.head(10).bool_col)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "field",
    [
        "tinyint_col",
        "smallint_col",
        "int_col",
        "bigint_col",
        "float_col",
        "double_col",
        "year",
        "month",
    ],
)
def test_negate_non_boolean(con, field, df):
    t = con.table("functional_alltypes").limit(10)
    expr = t.select((-t[field]).name(field))
    result = expr.execute()[field]
    expected = -df.head(10)[field]
    tm.assert_series_equal(result, expected)


def test_negate_boolean(con, df):
    t = con.table("functional_alltypes").limit(10)
    expr = t.select((-t.bool_col).name("bool_col"))
    result = expr.execute().bool_col
    expected = -df.head(10).bool_col
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("opname", ["sum", "mean", "min", "max", "std", "var"])
def test_boolean_reduction(alltypes, opname, df):
    op = operator.methodcaller(opname)
    expr = op(alltypes.bool_col)
    result = expr.execute()
    assert result == op(df.bool_col)


def test_timestamp_with_timezone(con):
    t = con.table("tzone")
    result = t.ts.execute()
    assert str(result.dtype.tz)


@pytest.fixture(
    params=[
        None,
        "UTC",
        "America/New_York",
        "America/Los_Angeles",
        "Europe/Paris",
        "Chile/Continental",
        "Asia/Tel_Aviv",
        "Asia/Tokyo",
        "Africa/Nairobi",
        "Australia/Sydney",
    ]
)
def tz(request):
    return request.param


@pytest.fixture
def tzone_compute(con, temp_table, tz):
    schema = ibis.schema([("ts", dt.Timestamp(tz)), ("b", "double"), ("c", "string")])
    con.create_table(temp_table, schema=schema, temp=True)
    t = con.table(temp_table)

    n = 10
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2017-04-01", periods=n, tz=tz).values,
            "b": np.arange(n).astype("float64"),
            "c": list(string.ascii_lowercase[:n]),
        }
    )

    df.to_sql(
        temp_table,
        con.con,
        index=False,
        if_exists="append",
        dtype={"ts": sa.TIMESTAMP(timezone=True), "b": sa.FLOAT, "c": sa.TEXT},
    )

    yield t


def test_ts_timezone_is_preserved(tzone_compute, tz):
    assert dt.Timestamp(tz).equals(tzone_compute.ts.type())


def test_timestamp_with_timezone_select(tzone_compute, tz):
    ts = tzone_compute.ts.execute()
    assert str(getattr(ts.dtype, "tz", None)) == str(tz)


def test_timestamp_type_accepts_all_timezones(con):
    with con.begin() as c:
        cur = c.exec_driver_sql("SELECT name FROM pg_timezone_names").fetchall()
    assert all(dt.Timestamp(row.name).timezone == row.name for row in cur)


@pytest.mark.parametrize(
    ("left", "right", "type"),
    [
        param(L("2017-04-01"), date(2017, 4, 2), dt.date, id="ibis_date"),
        param(date(2017, 4, 2), L("2017-04-01"), dt.date, id="python_date"),
        param(
            L("2017-04-01 01:02:33"),
            datetime(2017, 4, 1, 1, 3, 34),
            dt.timestamp,
            id="ibis_timestamp",
        ),
        param(
            datetime(2017, 4, 1, 1, 3, 34),
            L("2017-04-01 01:02:33"),
            dt.timestamp,
            id="python_datetime",
        ),
    ],
)
@pytest.mark.parametrize("opname", ["eq", "ne", "lt", "le", "gt", "ge"])
def test_string_temporal_compare(con, opname, left, right, type):
    op = getattr(operator, opname)
    expr = op(left, right)
    result = con.execute(expr)
    left_raw = con.execute(L(left).cast(type))
    right_raw = con.execute(L(right).cast(type))
    expected = op(left_raw, right_raw)
    assert result == expected


@pytest.mark.parametrize(
    ("left", "right"),
    [
        param(L("2017-03-31").cast(dt.date), date(2017, 4, 2), id="ibis_date"),
        param(date(2017, 3, 31), L("2017-04-02").cast(dt.date), id="python_date"),
        param(
            L("2017-03-31 00:02:33").cast(dt.timestamp),
            datetime(2017, 4, 1, 1, 3, 34),
            id="ibis_timestamp",
        ),
        param(
            datetime(2017, 3, 31, 0, 2, 33),
            L("2017-04-01 01:03:34").cast(dt.timestamp),
            id="python_datetime",
        ),
    ],
)
@pytest.mark.parametrize(
    "op",
    [
        param(
            lambda left, right: ibis.timestamp("2017-04-01 00:02:34").between(
                left, right
            ),
            id="timestamp",
        ),
        param(
            lambda left, right: (
                ibis.timestamp("2017-04-01").cast(dt.date).between(left, right)
            ),
            id="date",
        ),
    ],
)
def test_string_temporal_compare_between(con, op, left, right):
    expr = op(left, right)
    result = con.execute(expr)
    assert isinstance(result, (bool, np.bool_))
    assert result


def test_scalar_parameter(con):
    start_string, end_string = "2009-03-01", "2010-07-03"

    start = ibis.param(dt.date)
    end = ibis.param(dt.date)
    t = con.table("functional_alltypes")
    col = t.date_string_col.cast("date")
    expr = col.between(start, end).name("res")
    expected_expr = col.between(start_string, end_string).name("res")

    result = expr.execute(params={start: start_string, end: end_string})
    expected = expected_expr.execute()
    tm.assert_series_equal(result, expected)


def test_string_to_binary_cast(con):
    t = con.table("functional_alltypes").limit(10)
    expr = t.string_col.cast("binary")
    result = expr.execute()
    name = expr.get_name()
    sql_string = (
        f"SELECT decode(string_col, 'escape') AS \"{name}\" "
        "FROM functional_alltypes LIMIT 10"
    )
    with con.begin() as c:
        cur = c.exec_driver_sql(sql_string)
        raw_data = [row[0][0] for row in cur]
    expected = pd.Series(raw_data, name=name)
    tm.assert_series_equal(result, expected)


def test_string_to_binary_round_trip(con):
    t = con.table("functional_alltypes").limit(10)
    expr = t.string_col.cast("binary").cast("string")
    result = expr.execute()
    name = expr.get_name()
    sql_string = (
        "SELECT encode(decode(string_col, 'escape'), 'escape') AS "
        f'"{name}"'
        "FROM functional_alltypes LIMIT 10"
    )
    with con.begin() as c:
        cur = c.exec_driver_sql(sql_string)
        expected = pd.Series([row[0][0] for row in cur], name=name)
    tm.assert_series_equal(result, expected)
