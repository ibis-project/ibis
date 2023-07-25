from __future__ import annotations

import pytest
from pytest import param

import ibis
import ibis.common.exceptions as com
from ibis import window
from ibis.backends.impala.compiler import ImpalaCompiler
from ibis.tests.util import assert_equal

pytest.importorskip("impala")


@pytest.fixture(scope="module")
def alltypes(mockcon):
    return mockcon.table("alltypes")


def assert_sql_equal(expr, snapshot, out="out.sql"):
    result = ImpalaCompiler.to_sql(expr)
    snapshot.assert_match(result, out)


def test_aggregate_in_projection(alltypes, snapshot):
    t = alltypes
    proj = t[t, (t.f / t.f.sum()).name("normed_f")]
    assert_sql_equal(proj, snapshot)


def test_add_default_order_by(alltypes, snapshot):
    t = alltypes

    first = t.f.first().name("first")
    last = t.f.last().name("last")
    lag = t.f.lag().name("lag")
    diff = (t.f.lead() - t.f).name("fwd_diff")
    lag2 = t.f.lag().over(window(order_by=t.d)).name("lag2")
    grouped = t.group_by("g")
    proj = grouped.mutate([lag, diff, first, last, lag2])
    assert_sql_equal(proj, snapshot)


@pytest.mark.parametrize(
    "window",
    [
        param(window(preceding=0), id="prec_0"),
        param(window(following=0), id="foll_0"),
        param(window(preceding=5), id="prec_5"),
        param(window(preceding=5, following=0), id="prec_5_foll_0"),
        param(window(preceding=5, following=2), id="prec_5_foll_2"),
        param(window(following=2), id="foll_2"),
        param(window(following=2, preceding=0), id="foll_2_prec_0"),
        param(window(following=(5, 10)), id="foll_5_10"),
        param(window(preceding=(10, 5)), id="foll_10_5"),
        param(ibis.cumulative_window(), id="cumulative"),
        param(ibis.trailing_window(10), id="trailing_10"),
    ],
)
def test_window_frame_specs(alltypes, window, snapshot):
    t = alltypes

    w2 = window.order_by(t.f)
    expr = t.select(foo=t.d.sum().over(w2))
    assert_sql_equal(expr, snapshot)


def test_window_rows_with_max_lookback(alltypes):
    mlb = ibis.rows_with_max_lookback(3, ibis.interval(days=3))
    t = alltypes
    w = ibis.trailing_window(mlb, order_by=t.i)
    expr = t.a.sum().over(w)
    with pytest.raises(NotImplementedError):
        ImpalaCompiler.to_sql(expr)


@pytest.mark.parametrize("name", ["sum", "min", "max", "mean"])
def test_cumulative_functions(alltypes, name, snapshot):
    t = alltypes
    w = ibis.window(order_by=t.d)

    func = getattr(t.f, name)
    cumfunc = getattr(t.f, f"cum{name}")

    expr = cumfunc().over(w).name("foo")
    expected = func().over(ibis.cumulative_window(order_by=t.d)).name("foo")

    expr1 = t.select(expr)
    expr2 = t.select(expected)

    assert_sql_equal(expr1, snapshot, "out1.sql")
    assert_sql_equal(expr2, snapshot, "out2.sql")


def test_nested_analytic_function(alltypes, snapshot):
    t = alltypes

    w = window(order_by=t.f)
    expr = (t.f - t.f.lag()).lag().over(w).name("foo")
    result = t.select(expr)
    assert_sql_equal(result, snapshot)


def test_rank_functions(alltypes, snapshot):
    t = alltypes

    proj = t[t.g, t.f.rank().name("minr"), t.f.dense_rank().name("denser")]
    assert_sql_equal(proj, snapshot)


def test_multiple_windows(alltypes, snapshot):
    t = alltypes

    w = window(group_by=t.g)

    expr = t.f.sum().over(w) - t.f.sum()
    proj = t.select(t.g, result=expr)

    assert_sql_equal(proj, snapshot)


def test_order_by_desc(alltypes, snapshot):
    t = alltypes

    w = window(order_by=ibis.desc(t.f))

    proj = t[t.f, ibis.row_number().over(w).name("revrank")]
    assert_sql_equal(proj, snapshot, "out1.sql")

    expr = t.group_by("g").order_by(ibis.desc(t.f))[t.d.lag().name("foo"), t.a.max()]
    assert_sql_equal(expr, snapshot, "out2.sql")


def test_row_number_does_not_require_order_by(alltypes, snapshot):
    t = alltypes

    expr = t.group_by(t.g).mutate(ibis.row_number().name("foo"))
    assert_sql_equal(expr, snapshot, "out1.sql")

    expr = t.group_by(t.g).order_by(t.f).mutate(ibis.row_number().name("foo"))
    assert_sql_equal(expr, snapshot, "out2.sql")


def test_row_number_properly_composes_with_arithmetic(alltypes, snapshot):
    t = alltypes
    w = ibis.window(order_by=t.f)
    expr = t.mutate(new=ibis.row_number().over(w) / 2)
    assert_sql_equal(expr, snapshot)


@pytest.mark.parametrize(
    ["column", "op"],
    [("f", "approx_nunique"), ("f", "approx_median"), ("g", "group_concat")],
)
def test_unsupported_aggregate_functions(alltypes, column, op):
    t = alltypes
    w = ibis.window(order_by=t.d)
    expr = getattr(t[column], op)()
    proj = t.select(foo=expr.over(w))
    with pytest.raises(com.TranslationError):
        ImpalaCompiler.to_sql(proj)


def test_propagate_nested_windows(alltypes, snapshot):
    # GH #469
    t = alltypes

    w = ibis.window(group_by=t.g, order_by=t.f)

    col = (t.f - t.f.lag()).lag()

    # propagate down here!
    result = col.over(w)
    ex_expr = (t.f - t.f.lag().over(w)).lag().over(w)
    assert_equal(result, ex_expr)

    expr = t.select(col.over(w).name("foo"))
    assert_sql_equal(expr, snapshot)
