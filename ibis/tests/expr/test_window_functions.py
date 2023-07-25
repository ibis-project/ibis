from __future__ import annotations

import ibis
import ibis.expr.operations as ops


def test_mutate_with_analytic_functions(alltypes):
    t = alltypes.limit(1000)

    f = t.f
    g = t.group_by(t.g).order_by(t.f)

    exprs = [
        f.lag(),
        f.lead(),
        f.rank(),
        f.dense_rank(),
        f.percent_rank(),
        f.ntile(buckets=7),
        f.first(),
        f.last(),
        f.first().over(ibis.window(preceding=10)),
        f.first().over(ibis.window(following=10)),
        ibis.row_number(),
        f.cumsum(),
        f.cummean(),
        f.cummin(),
        f.cummax(),
        # boolean cumulative reductions
        (f == 0).cumany(),
        (f == 0).cumall(),
        f.sum(),
        f.mean(),
        f.min(),
        f.max(),
    ]

    exprs = [expr.name("e%d" % i) for i, expr in enumerate(exprs)]
    proj = g.mutate(exprs)
    for field in proj.op().selections[1:]:
        assert isinstance(field, ops.Alias)
        assert isinstance(field.arg, ops.WindowFunction)


def test_value_over_api(alltypes):
    t = alltypes

    w1 = ibis.window(rows=(0, 1), group_by=t.g, order_by=[t.f, t.h])
    w2 = ibis.window(range=(-1, 1), group_by=[t.g, t.a], order_by=[t.f])

    expr = t.f.cumsum().over(rows=(0, 1), group_by=t.g, order_by=[t.f, t.h])
    expected = t.f.cumsum().over(w1)
    assert expr.equals(expected)

    expr = t.f.cumsum().over(range=(-1, 1), group_by=[t.g, t.a], order_by=[t.f])
    expected = t.f.cumsum().over(w2)
    assert expr.equals(expected)
