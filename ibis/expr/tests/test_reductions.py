from __future__ import annotations

import pytest
from pytest import param

import ibis
import ibis.expr.operations as ops
from ibis import _
from ibis.common.deferred import Deferred


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
    ],
)
@pytest.mark.parametrize(
    "cond",
    [
        pytest.param(lambda t: None, id="no_cond"),
        pytest.param(
            lambda t: t.string_col.isin(["1", "7"]),
            id="is_in",
        ),
        pytest.param(
            lambda t: _.string_col.isin(["1", "7"]),
            id="is_in_deferred",
        ),
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
