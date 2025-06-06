from __future__ import annotations

import operator
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis import literal as L

pytest.importorskip("psycopg2")


@pytest.mark.parametrize(("value", "expected"), [(0, None), (5.5, 5.5)])
def test_nullif_zero(con, value, expected):
    assert con.execute(L(value).nullif(0)) == expected


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
        param(ibis.null().fill_null(5), 5, id="filled"),
        param(L(5).fill_null(10), 5, id="not_filled"),
        param(L(5).nullif(5), None, id="nullif_null"),
        param(L(10).nullif(5), 10, id="nullif_not_null"),
    ],
)
def test_fill_null_nullif(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(ibis.coalesce(5, None, 4), 5, id="first"),
        param(ibis.coalesce(ibis.null(), 4, ibis.null()), 4, id="second"),
        param(ibis.coalesce(ibis.null(), ibis.null(), 3.14), 3.14, id="third"),
    ],
)
def test_coalesce(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        ibis.coalesce(ibis.null(), ibis.null()),
        ibis.coalesce(
            ibis.null().cast("int8"),
            ibis.null().cast("int8"),
            ibis.null().cast("int8"),
        ),
    ],
    ids=["all_null", "all_nulls_with_all_cast"],
)
def test_coalesce_all_na(con, expr):
    assert con.execute(expr) is None


def test_coalesce_all_na_double(con):
    expr = ibis.coalesce(ibis.null(), ibis.null(), ibis.null().cast("double"))
    assert np.isnan(con.execute(expr))


def test_numeric_builtins_work(alltypes, df):
    expr = alltypes.double_col.fill_null(0)
    result = expr.execute()
    expected = df.double_col.fillna(0).rename(expr.get_name())
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
    expr = bucket.cases(*enumerate(labels), else_=None)
    result = expr.execute()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    result = pd.Series(pd.Categorical(result, ordered=True))

    result.name = "double_col"

    expected = pd.cut(df.double_col, bins, labels=labels, right=False)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("distinct", [True, False])
def test_union_cte(alltypes, distinct, assert_sql):
    t = alltypes
    expr1 = t.group_by(t.string_col).aggregate(metric=t.double_col.sum())
    expr2 = expr1.view()
    expr3 = expr1.view()
    expr = expr1.union(expr2, distinct=distinct).union(expr3, distinct=distinct)
    assert_sql(expr)


@pytest.mark.parametrize(
    ("func", "pandas_func"),
    [
        param(
            lambda t, _: t.bool_col.count(),
            lambda df, _: df.bool_col.count(),
            id="count",
        ),
        param(
            lambda t, _: t.double_col.mean(),
            lambda df, _: df.double_col.mean(),
            id="mean",
        ),
        param(
            lambda t, _: t.double_col.min(),
            lambda df, _: df.double_col.min(),
            id="min",
        ),
        param(
            lambda t, _: t.double_col.max(),
            lambda df, _: df.double_col.max(),
            id="max",
        ),
        param(
            lambda t, _: t.double_col.var(),
            lambda df, _: df.double_col.var(),
            id="var",
        ),
        param(
            lambda t, _: t.double_col.std(),
            lambda df, _: df.double_col.std(),
            id="std",
        ),
        param(
            lambda t, _: t.double_col.var(how="sample"),
            lambda df, _: df.double_col.var(ddof=1),
            id="samp_var",
        ),
        param(
            lambda t, _: t.double_col.std(how="pop"),
            lambda df, _: df.double_col.std(ddof=0),
            id="pop_std",
        ),
        param(
            lambda t, cond: t.bool_col.count(where=cond),
            lambda df, cond: df.bool_col[cond].count(),
            id="count_where",
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

    expr = t.filter(~((t.string_col == t2.string_col).any()))
    result = expr.execute()

    left, right = df, t2.execute()
    expected = left[left.string_col != right.string_col]

    tm.assert_frame_equal(result, expected, check_index_type=False, check_dtype=False)


def test_subquery(alltypes, df):
    t = alltypes

    expr = (
        t.mutate(d=t.double_col.fill_null(0)).limit(1000).group_by("string_col").size()
    )
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
@pytest.mark.xfail(
    reason="Window function with empty PARTITION BY is not supported yet"
)
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


@pytest.mark.parametrize("func", ["mean", "sum", "min", "max"])
@pytest.mark.xfail(
    reason="Window function with empty PARTITION BY is not supported yet"
)
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
@pytest.mark.xfail(
    reason="Window function with empty PARTITION BY is not supported yet"
)
def test_cumulative_simple_window(alltypes, func, df):
    t = alltypes
    f = getattr(t.double_col, func)
    col = t.double_col - f().over(ibis.cumulative_window())
    expr = t.select(col.name("double_col"))
    result = expr.execute().double_col
    expected = df.double_col - getattr(df.double_col, f"cum{func}")()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["sum", "min", "max"])
@pytest.mark.xfail(
    reason="Window function with empty PARTITION BY is not supported yet"
)
def test_cumulative_ordered_window(alltypes, func, df):
    t = alltypes
    df = df.sort_values("timestamp_col").reset_index(drop=True)
    window = ibis.cumulative_window(order_by=t.timestamp_col)
    f = getattr(t.double_col, func)
    expr = t.select((t.double_col - f().over(window)).name("double_col"))
    result = expr.execute().double_col
    expected = df.double_col - getattr(df.double_col, f"cum{func}")()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("func", "shift_amount"), [("lead", -1), ("lag", 1)], ids=["lead", "lag"]
)
@pytest.mark.xfail(
    reason="Window function with empty PARTITION BY is not supported yet"
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
@pytest.mark.xfail(reason="Unsupported expr: (first(t0.double_col) + 1) - 1")
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
    expr = t.mutate(na_column=ibis.null()).na_column
    result = expr.execute()
    tm.assert_series_equal(result, pd.Series([None] * nrows, name="na_column"))


@pytest.mark.xfail(
    reason="Window function with empty PARTITION BY is not supported yet"
)
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
    expr = t.filter(t.double_col > t.double_col.mean())
    result = expr.execute()
    expected = df[df.double_col > df.double_col.mean()].reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


def test_array_length(con):
    array_types = con.table("array_types")
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
    result_sorted = result.sort_values(
        by=["x_length", "y_length", "z_length"], na_position="first"
    ).reset_index(drop=True)
    expected_sorted = expected.sort_values(
        by=["x_length", "y_length", "z_length"], na_position="first"
    ).reset_index(drop=True)
    tm.assert_frame_equal(result_sorted, expected_sorted)


def custom_sort_none_first(arr):
    return sorted(arr, key=lambda x: (x is not None, x))


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


def test_invert_bool(con, df):
    t = con.table("functional_alltypes").limit(10)
    expr = t.select((~t.bool_col).name("bool_col"))
    result = expr.execute().bool_col
    expected = ~df.head(10).bool_col
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


@pytest.mark.parametrize(
    ("left", "right", "type"),
    [
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


@pytest.mark.xfail(
    reason="function make_date(integer, integer, integer) does not exist"
)
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
        c.execute(sql_string)
        raw_data = [row[0][0] for row in c.fetchall()]
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
        c.execute(sql_string)
        expected = pd.Series([row[0][0] for row in c.fetchall()], name=name)
    tm.assert_series_equal(result, expected)
