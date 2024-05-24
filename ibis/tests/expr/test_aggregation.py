from __future__ import annotations

import pytest

import ibis
from ibis import _


@pytest.fixture
def table():
    return ibis.table(
        {"ints": "int", "floats": "float", "bools": "bool", "strings": "string"}
    )


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(lambda t, **kws: t.strings.arbitrary(**kws), id="arbitrary"),
        pytest.param(lambda t, **kws: t.strings.collect(**kws), id="collect"),
        pytest.param(lambda t, **kws: t.strings.group_concat(**kws), id="group_concat"),
        pytest.param(
            lambda t, **kws: t.strings.approx_nunique(**kws), id="approx_nunique"
        ),
        pytest.param(
            lambda t, **kws: t.strings.approx_median(**kws), id="approx_median"
        ),
        pytest.param(lambda t, **kws: t.strings.mode(**kws), id="mode"),
        pytest.param(lambda t, **kws: t.strings.max(**kws), id="max"),
        pytest.param(lambda t, **kws: t.strings.min(**kws), id="min"),
        pytest.param(lambda t, **kws: t.strings.argmax(t.ints, **kws), id="argmax"),
        pytest.param(lambda t, **kws: t.strings.argmin(t.ints, **kws), id="argmin"),
        pytest.param(lambda t, **kws: t.strings.median(**kws), id="median"),
        pytest.param(lambda t, **kws: t.strings.quantile(0.25, **kws), id="quantile"),
        pytest.param(
            lambda t, **kws: t.strings.quantile([0.25, 0.75], **kws),
            id="multi-quantile",
        ),
        pytest.param(lambda t, **kws: t.strings.nunique(**kws), id="nunique"),
        pytest.param(lambda t, **kws: t.strings.count(**kws), id="count"),
        pytest.param(lambda t, **kws: t.strings.first(**kws), id="first"),
        pytest.param(lambda t, **kws: t.strings.last(**kws), id="last"),
        pytest.param(lambda t, **kws: t.ints.std(**kws), id="std"),
        pytest.param(lambda t, **kws: t.ints.var(**kws), id="var"),
        pytest.param(lambda t, **kws: t.ints.mean(**kws), id="mean"),
        pytest.param(lambda t, **kws: t.ints.sum(**kws), id="sum"),
        pytest.param(lambda t, **kws: t.ints.corr(t.floats, **kws), id="corr"),
        pytest.param(lambda t, **kws: t.ints.cov(t.floats, **kws), id="cov"),
        pytest.param(lambda t, **kws: t.ints.bit_and(**kws), id="bit_and"),
        pytest.param(lambda t, **kws: t.ints.bit_xor(**kws), id="bit_xor"),
        pytest.param(lambda t, **kws: t.ints.bit_or(**kws), id="bit_or"),
        pytest.param(lambda t, **kws: t.bools.any(**kws), id="any"),
        pytest.param(lambda t, **kws: t.bools.all(**kws), id="all"),
        pytest.param(lambda t, **kws: t.count(**kws), id="table-count"),
        pytest.param(lambda t, **kws: t.nunique(**kws), id="table-nunique"),
    ],
)
def test_aggregation_where(table, func):
    # No where
    op = func(table).op()
    assert op.where is None

    # Literal where
    op = func(table, where=False).op()
    assert op.where.equals(ibis.literal(False).op())

    # Various ways to spell the same column expression
    r1 = func(table, where=table.bools)
    r2 = func(table, where=_.bools)
    r3 = func(table, where=lambda t: t.bools)
    r4 = func(table, where="bools")
    assert r1.equals(r2)
    assert r1.equals(r3)
    assert r1.equals(r4)
    assert r1.op().where.equals(table.bools.op())
