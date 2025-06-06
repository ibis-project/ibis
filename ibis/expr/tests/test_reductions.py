from __future__ import annotations

import pytest
from pytest import param

import ibis
import ibis.expr.operations as ops
from ibis import _
from ibis.common.annotations import ValidationError
from ibis.common.deferred import Deferred
from ibis.common.exceptions import IbisTypeError


@pytest.mark.parametrize(
    ("fn", "operation"),
    [
        param(
            lambda t, where: t.int_col.nunique(where=where),
            ops.CountDistinct,
            id="nunique",
        ),
        param(lambda t, where: t.bool_col.any(where=where), ops.Any, id="any"),
        param(lambda t, where: t.bool_col.all(where=where), ops.All, id="all"),
        param(lambda t, where: t.int_col.sum(where=where), ops.Sum, id="sum"),
        param(lambda t, where: t.int_col.mean(where=where), ops.Mean, id="mean"),
        param(lambda t, where: t.int_col.min(where=where), ops.Min, id="min"),
        param(lambda t, where: t.int_col.max(where=where), ops.Max, id="max"),
        param(
            lambda t, where: t.int_col.argmin(t.string_col, where=where),
            ops.ArgMin,
            id="argmin",
        ),
        param(
            lambda t, where: t.int_col.argmax(t.string_col, where=where),
            ops.ArgMax,
            id="argmax",
        ),
        param(
            lambda t, where: t.int_col.std(how="pop", where=where),
            ops.StandardDev,
            id="std",
        ),
        param(lambda t, where: t.int_col.var(where=where), ops.Variance, id="var"),
        param(
            lambda t, where: t.int_col.approx_nunique(where=where),
            ops.ApproxCountDistinct,
            id="approx_nunique",
        ),
        param(
            lambda t, where: t.int_col.arbitrary(where=where),
            ops.Arbitrary,
            id="arbitrary",
        ),
        param(lambda t, where: t.int_col.first(where=where), ops.First, id="first"),
        param(lambda t, where: t.int_col.last(where=where), ops.Last, id="last"),
        param(
            lambda t, where: t.int_col.bit_and(where=where), ops.BitAnd, id="bit_and"
        ),
        param(lambda t, where: t.int_col.bit_or(where=where), ops.BitOr, id="bit_or"),
        param(
            lambda t, where: t.int_col.bit_xor(where=where), ops.BitXor, id="bit_xor"
        ),
        param(
            lambda t, where: t.int_col.collect(where=where),
            ops.ArrayCollect,
            id="collect",
        ),
        param(
            lambda t, where: t.int_col.approx_quantile(0.5, where=where),
            ops.ApproxQuantile,
            id="approx_quantile",
        ),
        param(
            lambda t, where: t.int_col.approx_quantile([0.25, 0.5, 0.75], where=where),
            ops.ApproxMultiQuantile,
            id="approx_multi_quantile",
        ),
    ],
)
@pytest.mark.parametrize(
    "cond",
    [
        pytest.param(lambda _: None, id="no_cond"),
        pytest.param(lambda t: t.string_col.isin(["1", "7"]), id="is_in"),
        pytest.param(lambda _: _.string_col.isin(["1", "7"]), id="is_in_deferred"),
    ],
)
def test_reduction_methods(fn, operation, cond):
    t = ibis.table(
        name="t",
        schema={
            "string_col": "string",
            "int_col": "int64",
            "bool_col": "boolean",
        },
    )
    where = cond(t)
    expr = fn(t, where)
    node = expr.op()
    assert isinstance(node, operation)
    if where is None:
        assert node.where is None
    elif isinstance(where, Deferred):
        resolved = where.resolve(t).op()
        assert node.where == resolved
    else:
        assert node.where == where.op()


@pytest.mark.parametrize("func_name", ["argmin", "argmax"])
def test_argminmax_deferred(func_name):
    t = ibis.table({"a": "int", "b": "int"}, name="t")
    func = getattr(t.a, func_name)
    assert func(_.b).equals(func(t.b))


@pytest.mark.parametrize("func_name", ["cov", "corr"])
def test_cov_corr_deferred(func_name):
    t = ibis.table({"a": "int", "b": "int"}, name="t")
    func = getattr(t.a, func_name)
    assert func(_.b).equals(func(t.b))


@pytest.mark.parametrize("method", ["collect", "first", "last", "group_concat"])
def test_ordered_aggregations(method):
    t = ibis.table({"a": "string", "b": "int", "c": "int"}, name="t")
    func = getattr(t.a, method)

    q1 = func(order_by="b")
    q2 = func(order_by=("b",))
    q3 = func(order_by=_.b)
    q4 = func(order_by=t.b)
    assert q1.equals(q2)
    assert q1.equals(q3)
    assert q1.equals(q4)

    q5 = func(order_by=("b", "c"))
    q6 = func(order_by=(_.b, _.c))
    assert q5.equals(q6)

    q7 = func(order_by=_.b.desc())
    q8 = func(order_by=t.b.desc())
    assert q7.equals(q8)

    with pytest.raises(IbisTypeError):
        func(order_by="oops")


@pytest.mark.parametrize("method", ["collect", "first", "last", "group_concat"])
def test_ordered_aggregations_no_order(method):
    t = ibis.table({"a": "string", "b": "int", "c": "int"}, name="t")
    func = getattr(t.a, method)

    q1 = func()
    q2 = func(order_by=None)
    q3 = func(order_by=())
    assert q1.equals(q2)
    assert q1.equals(q3)


def test_collect_distinct():
    t = ibis.table({"a": "string", "b": "int", "c": "int"}, name="t")
    # Fine
    t.a.collect(distinct=True)
    t.a.collect(distinct=True, order_by=t.a.desc())
    (t.a + 1).collect(distinct=True, order_by=(t.a + 1).desc())

    with pytest.raises(ValidationError, match="only order by the collected column"):
        t.b.collect(distinct=True, order_by=t.a)
    with pytest.raises(ValidationError, match="only order by the collected column"):
        t.b.collect(
            distinct=True,
            order_by=(
                t.a,
                t.b,
            ),
        )
