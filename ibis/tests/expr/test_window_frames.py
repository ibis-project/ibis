from __future__ import annotations

import pytest
from pytest import param

import ibis
import ibis.expr.builders as bl
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.annotations import ValidationError
from ibis.common.exceptions import IbisInputError
from ibis.common.patterns import NoMatch, Pattern


@pytest.fixture
def t(alltypes):
    return alltypes


def test_window_boundary():
    # the boundary value must be either numeric or interval
    b = ops.WindowBoundary(5, preceding=False)
    assert b.value == ops.Literal(5, dtype=dt.int8)

    b = ops.WindowBoundary(3.12, preceding=True)
    assert b.value == ops.Literal(3.12, dtype=dt.double)

    oneday = ops.Literal(1, dtype=dt.Interval("D"))
    b = ops.WindowBoundary(oneday, preceding=False)
    assert b.value == oneday

    with pytest.raises(ValidationError):
        ops.WindowBoundary("foo", preceding=True)


def test_window_boundary_typevars():
    lit = ops.Literal(1, dtype=dt.Interval("D"))

    p = Pattern.from_typehint(ops.WindowBoundary[dt.Integer, ds.Any])
    b = ops.WindowBoundary(5, preceding=False)
    assert p.match(b, {}) == b
    assert p.match(ops.WindowBoundary(5.0, preceding=False), {}) is NoMatch
    assert p.match(ops.WindowBoundary(lit, preceding=True), {}) is NoMatch

    p = Pattern.from_typehint(ops.WindowBoundary[dt.Interval, ds.Any])
    b = ops.WindowBoundary(lit, preceding=True)
    assert p.match(b, {}) == b


def test_window_boundary_coercions():
    RowsWindowBoundary = ops.WindowBoundary[dt.Integer, ds.Any]
    p = Pattern.from_typehint(RowsWindowBoundary)
    assert p.match(1, {}) == RowsWindowBoundary(ops.Literal(1, dtype=dt.int8), False)


def test_window_builder_rows():
    w0 = bl.WindowBuilder()

    assert w0.start is None
    assert w0.end is None
    with pytest.raises(ValidationError):
        w0.rows(5)

    w1 = w0.rows(5, 10)
    assert w1 is not w0
    assert w1.start == ops.WindowBoundary(5, preceding=False)
    assert w1.end == ops.WindowBoundary(10, preceding=False)
    assert w1.how == "rows"

    w2 = w0.rows(-5, 10)
    assert w2.start == ops.WindowBoundary(5, preceding=True)
    assert w2.end == ops.WindowBoundary(10, preceding=False)
    assert w2.how == "rows"

    with pytest.raises(IbisInputError):
        w0.rows(-5, -10)

    w3 = w0.rows(-5, -4)
    assert w3.start == ops.WindowBoundary(5, preceding=True)
    assert w3.end == ops.WindowBoundary(4, preceding=True)
    assert w3.how == "rows"

    w4 = w0.rows(5, None)
    assert w4.start == ops.WindowBoundary(5, preceding=False)
    assert w4.end is None
    assert w4.how == "rows"

    w5 = w0.rows(None, 10)
    assert w5.start is None
    assert w5.end == ops.WindowBoundary(10, preceding=False)
    assert w5.how == "rows"

    w6 = w0.rows(None, None)
    assert w6.start is None
    assert w6.end is None
    assert w6.how == "rows"

    with pytest.raises(ValidationError):
        w0.rows(5, ibis.interval(days=1))
    with pytest.raises(ValidationError):
        w0.rows(ibis.interval(days=1), 10)


def test_window_builder_range():
    w0 = bl.WindowBuilder()

    assert w0.start is None
    assert w0.end is None
    with pytest.raises(ValidationError):
        w0.range(5)

    w1 = w0.range(5, 10)
    assert w1 is not w0
    assert w1.start == ops.WindowBoundary(5, preceding=False)
    assert w1.end == ops.WindowBoundary(10, preceding=False)
    assert w1.how == "range"

    w2 = w0.range(-5, 10)
    assert w2.start == ops.WindowBoundary(5, preceding=True)
    assert w2.end == ops.WindowBoundary(10, preceding=False)
    assert w2.how == "range"

    with pytest.raises(IbisInputError):
        w0.range(-5, -10)

    w3 = w0.range(-5, -3)
    assert w3.start == ops.WindowBoundary(5, preceding=True)
    assert w3.end == ops.WindowBoundary(3, preceding=True)
    assert w3.how == "range"

    w4 = w0.range(5, None)
    assert w4.start == ops.WindowBoundary(5, preceding=False)
    assert w4.end is None
    assert w4.how == "range"

    w5 = w0.range(None, 10)
    assert w5.start is None
    assert w5.end == ops.WindowBoundary(10, preceding=False)
    assert w5.how == "range"

    w6 = w0.range(None, None)
    assert w6.start is None
    assert w6.end is None
    assert w6.how == "range"

    w7 = w0.range(ibis.interval(days=1), ibis.interval(days=2))
    assert w7.start == ops.WindowBoundary(ibis.interval(days=1), preceding=False)
    assert w7.end == ops.WindowBoundary(ibis.interval(days=2), preceding=False)
    assert w7.how == "range"

    w8 = w0.range(-ibis.interval(days=1), ibis.interval(days=2))
    assert w8.start == ops.WindowBoundary(ibis.interval(days=1), preceding=True)
    assert w8.end == ops.WindowBoundary(ibis.interval(days=2), preceding=False)
    assert w8.how == "range"

    w9 = w0.range(-ibis.interval(days=1), 10)
    assert w9.start == ops.WindowBoundary(ibis.interval(days=1), preceding=True)
    value = ibis.literal(10).cast("interval('D')")
    assert w9.end == ops.WindowBoundary(value, preceding=False)
    assert w9.how == "range"

    w10 = w0.range(5, ibis.interval(seconds=11))
    value = ibis.literal(5).cast("interval('s')")
    assert w10.start == ops.WindowBoundary(value, preceding=False)
    assert w10.end == ops.WindowBoundary(ibis.interval(seconds=11), preceding=False)
    assert w10.how == "range"


def test_window_builder_between():
    w0 = bl.WindowBuilder()

    w1 = w0.between(None, 5)
    assert w1.start is None
    assert w1.end == ops.WindowBoundary(5, preceding=False)
    assert w1.how == "rows"

    w2 = w0.between(1, 3)
    assert w2.start == ops.WindowBoundary(1, preceding=False)
    assert w2.end == ops.WindowBoundary(3, preceding=False)
    assert w2.how == "rows"

    w3 = w0.between(-1, None)
    assert w3.start == ops.WindowBoundary(1, preceding=True)
    assert w3.end is None
    assert w1.how == "rows"

    w4 = w0.between(None, None)
    assert w4.start is None
    assert w4.end is None
    assert w1.how == "rows"

    w5 = w0.between(ibis.interval(days=1), ibis.interval(days=2))
    assert w5.start == ops.WindowBoundary(ibis.interval(days=1), preceding=False)
    assert w5.end == ops.WindowBoundary(ibis.interval(days=2), preceding=False)
    assert w5.how == "range"

    w6 = w0.between(-ibis.interval(days=1), ibis.interval(days=2))
    assert w6.start == ops.WindowBoundary(ibis.interval(days=1), preceding=True)
    assert w6.end == ops.WindowBoundary(ibis.interval(days=2), preceding=False)
    assert w6.how == "range"

    w7 = w0.between(-ibis.interval(days=1), 10)
    assert w7.start == ops.WindowBoundary(ibis.interval(days=1), preceding=True)
    value = ibis.literal(10).cast("interval('D')")
    assert w7.end == ops.WindowBoundary(value, preceding=False)
    assert w7.how == "range"

    w8 = w0.between(5, ibis.interval(seconds=11))
    value = ibis.literal(5).cast("interval('s')")
    assert w8.start == ops.WindowBoundary(value, preceding=False)
    assert w8.end == ops.WindowBoundary(ibis.interval(seconds=11), preceding=False)
    assert w8.how == "range"

    w9 = w0.between(-0.5, 1.5)
    assert w9.start == ops.WindowBoundary(0.5, preceding=True)
    assert w9.end == ops.WindowBoundary(1.5, preceding=False)
    assert w9.how == "range"


def test_window_api_supports_value_expressions(t):
    w = ibis.window(between=(t.d, t.d + 1), group_by=t.b, order_by=t.c)
    func = t.a.sum().over(w).op()
    expected = ops.WindowFunction(
        func=t.a.sum(),
        how="rows",
        start=ops.WindowBoundary(t.d, preceding=False),
        end=ops.WindowBoundary(t.d + 1, preceding=False),
        group_by=(t.b,),
        order_by=(t.c,),
    )
    assert func == expected


def test_window_api_supports_scalar_order_by(t):
    window = ibis.window(order_by=ibis.null())
    expr = t.a.sum().over(window).op()
    expected = ops.WindowFunction(
        t.a.sum(),
        how="rows",
        start=None,
        end=None,
        group_by=(),
        order_by=(ibis.null().op(),),
    )
    assert expr == expected

    rand = ibis.random()
    window = ibis.window(order_by=rand)
    expr = t.a.sum().over(window).op()
    expected = ops.WindowFunction(
        t.a.sum(),
        how="rows",
        start=None,
        end=None,
        group_by=(),
        order_by=[rand],
    )
    assert expr == expected


def test_window_api_properly_determines_how():
    assert ibis.window(between=(None, 5)).how == "rows"
    assert ibis.window(between=(1, 3)).how == "rows"
    assert ibis.window(5).how == "rows"
    assert ibis.window(ibis.interval(days=3)).how == "range"
    assert ibis.window(3.1).how == "range"
    assert ibis.window(following=3.14).how == "range"
    assert ibis.window(following=3).how == "rows"


def test_window_api_properly_determines_how_numpy():
    np = pytest.importorskip("numpy")
    assert ibis.window(np.int64(7)).how == "rows"


def test_window_api_mutually_exclusive_options():
    with pytest.raises(IbisInputError):
        ibis.window(between=(None, 5), preceding=3)
    with pytest.raises(IbisInputError):
        ibis.window(between=(None, 5), following=3)
    with pytest.raises(IbisInputError):
        ibis.window(rows=(None, 5), preceding=3)
    with pytest.raises(IbisInputError):
        ibis.window(rows=(None, 5), following=3)
    with pytest.raises(IbisInputError):
        ibis.window(range=(None, 5), preceding=3)
    with pytest.raises(IbisInputError):
        ibis.window(range=(None, 5), following=3)
    with pytest.raises(IbisInputError):
        ibis.window(rows=(None, 5), between=(None, 5))
    with pytest.raises(IbisInputError):
        ibis.window(rows=(None, 5), range=(None, 5))
    with pytest.raises(IbisInputError):
        ibis.window(range=(None, 5), between=(None, 5))


def test_window_builder_methods(t):
    w1 = ibis.window(preceding=5, following=1, group_by=t.a, order_by=t.b)

    w2 = w1.group_by(t.c)
    expected = ibis.window(preceding=5, following=1, group_by=[t.a, t.c], order_by=t.b)
    assert w2 == expected

    w3 = w1.order_by(t.d)
    expected = ibis.window(preceding=5, following=1, group_by=t.a, order_by=[t.b, t.d])
    assert w3 == expected


@pytest.mark.parametrize(
    ["method", "is_preceding"],
    [
        (ibis.preceding, True),
        (ibis.following, False),
    ],
)
def test_window_api_preceding_following(method, is_preceding):
    p0 = method(5).op()
    assert isinstance(p0, ops.WindowBoundary)
    assert isinstance(p0.value, ops.Literal)
    assert p0.value.value == 5
    assert p0.preceding == is_preceding

    p1 = method(-5).op()
    assert p1.value.value == -5
    assert p1.preceding == is_preceding

    p2 = method(ibis.interval(days=1)).op()
    assert p2.value.value == 1
    assert p2.preceding == is_preceding

    p3 = method(ibis.interval(days=-1)).op()
    assert p3.value.value == -1
    assert p3.preceding == is_preceding

    t = ibis.table([("a", "int64")], name="t")
    p4 = method(t.a).op()
    assert p4.value == t.a.op()


def test_window_api_trailing_range():
    t = ibis.table([("col", "int64")], name="t")
    start = ibis.interval(days=1)
    end = ibis.literal(0).cast(start.type())

    window = ibis.trailing_range_window(start, order_by="col")
    expr = t.col.sum().over(window).op()

    expected = ops.WindowFunction(
        t.col.sum(),
        how="range",
        start=ops.WindowBoundary(start, preceding=True),
        end=ops.WindowBoundary(end, preceding=False),
        group_by=(),
        order_by=[t.col],
    )
    assert expr == expected


@pytest.mark.parametrize(
    ["a", "b"],
    [
        (ibis.window(preceding=1), ibis.window(rows=(-1, None))),
        (ibis.window(following=0), ibis.window(rows=(None, 0))),
        (ibis.window(preceding=1, following=0), ibis.window(rows=(-1, 0))),
        (ibis.window(following=(1, None)), ibis.window(rows=(1, None))),
        (ibis.window(preceding=(None, 1)), ibis.window(rows=(None, -1))),
        (
            # GH-3305
            ibis.window(following=(ibis.literal(1), None)),
            ibis.window(rows=(1, None)),
        ),
        (ibis.range_window(preceding=10, following=0), ibis.window(range=(-10, 0))),
        (ibis.range_window(preceding=(4, 2)), ibis.window(range=(-4, -2))),
        (
            # GH-3305
            ibis.range_window(following=(ibis.interval(seconds=1), None)),
            ibis.window(range=(ibis.interval(seconds=1), None)),
        ),
    ],
)
def test_window_api_legacy_to_new(a, b):
    assert a.how == b.how
    assert a.start == b.start
    assert a.end == b.end
    assert a.orderings == b.orderings
    assert a.groupings == b.groupings


@pytest.mark.parametrize(
    "case",
    [
        param(dict(preceding=(1, 3)), id="double_preceding"),
        param(dict(preceding=(3, 1), following=2), id="preceding_and_following"),
        param(dict(preceding=(3, 1), following=(2, 4)), id="preceding_and_following2"),
        param(dict(preceding=-1), id="negative_preceding"),
        param(dict(following=-1), id="negative_following"),
        param(dict(preceding=(-1, 2)), id="invalid_preceding"),
        param(dict(following=(2, -1)), id="invalid_following"),
    ],
)
def test_window_api_preceding_following_invalid(case):
    with pytest.raises(IbisInputError):
        ibis.window(**case)

    with pytest.raises(IbisInputError):
        ibis.rows_window(**case)

    with pytest.raises(IbisInputError):
        ibis.range_window(**case)


@pytest.mark.parametrize(
    ("kind", "begin", "end"),
    [
        ("preceding", None, None),
        ("preceding", 1, None),
        ("preceding", -1, 1),
        ("preceding", 1, -1),
        ("preceding", -1, -1),
        ("following", None, None),
        ("following", None, 1),
        ("following", -1, 1),
        ("following", 1, -1),
        ("following", -1, -1),
    ],
)
def test_window_api_preceding_following_invalid_tuple(kind, begin, end):
    kwargs = {kind: (begin, end)}
    with pytest.raises(IbisInputError):
        ibis.window(**kwargs)


def test_window_bind_to_table(t):
    spec = ibis.window(group_by="g", order_by=ibis.desc("f"))

    window = t.a.sum().over(spec).op()
    expected = ops.WindowFunction(
        t.a.sum(),
        how="rows",
        start=None,
        end=None,
        group_by=[t.g],
        order_by=[t.f.desc()],
    )
    assert window == expected


def test_window_bind_value_expression_using_over(t):
    # GH #542
    w = ibis.window(group_by="g", order_by="f")

    window = t.f.lag().over(w).op()
    expected = ops.WindowFunction(
        t.f.lag(),
        how="rows",
        start=None,
        end=None,
        group_by=[t.g],
        order_by=[t.f.asc()],
    )
    assert window == expected


def test_window_analysis_propagate_nested_windows(t):
    # GH #469
    w = ibis.window(group_by=t.g, order_by=t.f)
    col = (t.f - t.f.lag()).lag()

    # propagate down here!
    result = col.over(w)
    expected = (t.f - t.f.lag().over(w)).lag().over(w)
    assert result.equals(expected)


def test_window_analysis_combine_group_by(t):
    w = ibis.window(group_by=t.g, order_by=t.f)

    diff = t.d - t.d.lag()
    grouped = t.group_by("g").order_by("f")

    expr = grouped[t, diff.name("diff")]
    expr2 = grouped.mutate(diff=diff)
    expr3 = grouped.mutate([diff.name("diff")])

    window_expr = (t.d - t.d.lag().over(w)).name("diff")
    expected = t.select([t, window_expr])

    assert expr.equals(expected)
    assert expr.equals(expr2)
    assert expr.equals(expr3)


def test_window_analysis_combine_preserves_existing_window():
    t = ibis.table(
        [("one", "string"), ("two", "double"), ("three", "int32")],
        name="my_data",
    )
    w = ibis.cumulative_window(order_by=t.one)
    mut = t.group_by(t.three).mutate(four=t.two.sum().over(w))
    assert mut.op().values["four"].start is None


def test_window_analysis_auto_windowize_bug():
    # GH #544
    t = ibis.table(
        name="airlines", schema={"arrdelay": "int32", "dest": "string", "year": "int32"}
    )

    def metric(x):
        return x.arrdelay.mean().name("avg_delay")

    annual_delay = (
        t.filter(t.dest.isin(["JFK", "SFO"]))
        .group_by(["dest", "year"])
        .aggregate(metric)
    )
    what = annual_delay.group_by("dest")
    enriched = what.mutate(grand_avg=annual_delay.avg_delay.mean())

    expr = (
        annual_delay.avg_delay.mean()
        .name("grand_avg")
        .over(ibis.window(group_by=annual_delay.dest))
    )
    expected = annual_delay.select(annual_delay, expr)

    assert enriched.equals(expected)


def test_windowization_wraps_reduction_inside_a_nested_value_expression(t):
    win = ibis.window(
        following=0,
        group_by=[t.g],
        order_by=[t.a],
    )
    expr = (t.f == 0).notany().over(win)
    expected = ops.Not(
        ops.WindowFunction(
            func=ops.Any(t.f == 0),
            how="rows",
            start=None,
            end=0,
            group_by=[t.g],
            order_by=[t.a],
        )
    )
    assert expr.op() == expected


def test_group_by_with_window_function_preserves_range():
    t = ibis.table(dict(one="string", two="double", three="int32"), name="my_data")
    w = ibis.cumulative_window(order_by=t.one)
    expr = t.group_by(t.three).mutate(four=t.two.sum().over(w))

    expected = ops.Project(
        parent=t,
        values={
            "one": t.one,
            "two": t.two,
            "three": t.three,
            "four": ops.WindowFunction(
                func=ops.Sum(t.two),
                how="rows",
                start=None,
                end=0,
                group_by=[t.three],
                order_by=[t.one],
            ),
        },
    )
    assert expr.op() == expected
