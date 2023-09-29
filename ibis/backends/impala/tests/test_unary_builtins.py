from __future__ import annotations

import pytest
from pytest import param

import ibis
import ibis.expr.types as ir
from ibis.backends.impala.tests.conftest import translate
from ibis.common.annotations import ValidationError


@pytest.fixture(scope="module")
def table(mockcon):
    return mockcon.table("functional_alltypes")


@pytest.mark.parametrize(
    "method",
    [
        param(lambda x: x.abs(), id="abs"),
        param(lambda x: x.ceil(), id="ceil"),
        param(lambda x: x.floor(), id="floor"),
        param(lambda x: x.exp(), id="exp"),
        param(lambda x: x.sqrt(), id="sqrt"),
        param(lambda x: x.log(), id="log"),
        param(lambda x: x.approx_median(), id="approx_median"),
        param(lambda x: x.approx_nunique(), id="approx_nunique"),
        param(lambda x: x.ln(), id="ln"),
        param(lambda x: x.log2(), id="log2"),
        param(lambda x: x.log10(), id="log10"),
        param(
            lambda x: pytest.warns(FutureWarning, lambda y: y.nullifzero(), x),
            id="nullifzero",
        ),
        param(
            lambda x: pytest.warns(FutureWarning, lambda y: y.zeroifnull(), x),
            id="zeroifnull",
        ),
    ],
)
@pytest.mark.parametrize("cname", ["double_col", "int_col"])
def test_numeric_unary_builtins(method, cname, table, snapshot):
    col = table[cname]
    expr = method(col)

    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize(
    "expr_fn",
    [
        pytest.param(lambda t: t.double_col.log(5), id="log_with_base"),
        pytest.param(lambda t: t.double_col.round(), id="round_no_args"),
        pytest.param(lambda t: t.double_col.round(0), id="round_zero"),
        pytest.param(lambda t: t.double_col.round(2), id="round_two"),
        pytest.param(lambda t: t.double_col.round(t.tinyint_col), id="round_expr"),
        pytest.param(lambda t: t.tinyint_col.sign(), id="sign_tinyint"),
        pytest.param(lambda t: t.float_col.sign(), id="sign_float"),
        pytest.param(lambda t: t.double_col.sign(), id="sign_double"),
    ],
)
def test_numeric(expr_fn, table, snapshot):
    expr = expr_fn(table)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


def test_hash(table, snapshot):
    expr = table.int_col.hash()
    assert isinstance(expr, ir.IntegerColumn)
    assert isinstance(table.int_col.sum().hash(), ir.IntegerScalar)

    expr = table.int_col.hash()
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize(
    "expr_fn",
    [
        pytest.param(lambda t: t.double_col.sum(where=t.bigint_col < 70), id="sum"),
        pytest.param(lambda t: t.double_col.count(where=t.bigint_col < 70), id="count"),
        pytest.param(lambda t: t.double_col.mean(where=t.bigint_col < 70), id="avg"),
        pytest.param(lambda t: t.double_col.max(where=t.bigint_col < 70), id="max"),
        pytest.param(lambda t: t.double_col.min(where=t.bigint_col < 70), id="min"),
        pytest.param(
            lambda t: t.double_col.std(where=t.bigint_col < 70), id="stddev_samp"
        ),
        pytest.param(
            lambda t: t.double_col.std(where=t.bigint_col < 70, how="pop"),
            id="stddev_pop",
        ),
        pytest.param(
            lambda t: t.double_col.var(where=t.bigint_col < 70), id="var_samp"
        ),
        pytest.param(
            lambda t: t.double_col.var(where=t.bigint_col < 70, how="pop"), id="var_pop"
        ),
    ],
)
def test_reduction_where(table, expr_fn, snapshot):
    expr = expr_fn(table)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize("method_name", ["sum", "count", "mean", "max", "min"])
def test_reduction_invalid_where(table, method_name):
    condbad_literal = ibis.literal("T")
    reduction = getattr(table.double_col, method_name)

    with pytest.raises(ValidationError):
        reduction(where=condbad_literal)
