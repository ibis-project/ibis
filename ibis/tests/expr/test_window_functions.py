from __future__ import annotations

import pytest

import ibis
import ibis.expr.operations as ops
from ibis.common.exceptions import ExpressionError


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

    values = list(proj.op().values.values())
    for field in values[len(t.schema()) :]:
        assert isinstance(field, ops.WindowFunction)


def test_value_over_api(alltypes):
    t = alltypes

    w1 = ibis.window(rows=(0, 1), group_by=t.g, order_by=[t.f, t.h])
    w2 = ibis.window(range=(-1, 1), group_by=[t.g, t.a], order_by=[t.f])

    expr = t.f.sum().over(rows=(0, 1), group_by=t.g, order_by=[t.f, t.h])
    expected = t.f.sum().over(w1)
    assert expr.equals(expected)

    expr = t.f.sum().over(range=(-1, 1), group_by=[t.g, t.a], order_by=[t.f])
    expected = t.f.sum().over(w2)
    assert expr.equals(expected)


def test_conflicting_window_boundaries(alltypes):
    t = alltypes

    with pytest.raises(ExpressionError, match="Unable to merge windows"):
        t.f.cumsum().over(rows=(0, 1))


def test_rank_followed_by_over_call_merge_frames(alltypes):
    t = alltypes
    expr1 = t.f.percent_rank().over(ibis.window(group_by=t.f.notnull()))
    expr2 = ibis.percent_rank().over(group_by=t.f.notnull(), order_by=t.f)
    assert expr1.equals(expr2)
