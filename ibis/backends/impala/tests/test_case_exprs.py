from __future__ import annotations

import pytest

import ibis
import ibis.expr.types as ir
from ibis.backends.impala.compiler import ImpalaCompiler
from ibis.backends.impala.tests.conftest import translate


@pytest.fixture(scope="module")
def table(mockcon):
    return mockcon.table("alltypes")


@pytest.fixture
def simple_case(table):
    return table.g.case().when("foo", "bar").when("baz", "qux").else_("default").end()


@pytest.fixture
def search_case(table):
    t = table
    return ibis.case().when(t.f > 0, t.d * 2).when(t.c < 0, t.a * 2).end()


@pytest.fixture
def tpch_lineitem(mockcon):
    return mockcon.table("tpch_lineitem")


def test_isnull_1_0(table, snapshot):
    expr = table.g.isnull().ifelse(1, 0)

    result = translate(expr)
    snapshot.assert_match(result, "out1.sql")

    # inside some other function
    result = translate(expr.sum())
    snapshot.assert_match(result, "out2.sql")


def test_simple_case(simple_case, snapshot):
    expr = simple_case
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


def test_search_case(search_case, snapshot):
    expr = search_case
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


def test_ifelse_use_if(table, snapshot):
    expr = ibis.ifelse(table.f > 0, table.e, table.a)
    assert isinstance(expr, ir.FloatingValue)

    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize(
    "expr_fn",
    [
        pytest.param(lambda f: f.nullif(f), id="nullif_input"),
        pytest.param(lambda f: (f == 0).nullif(f == 0), id="nullif_boolean"),
        pytest.param(lambda f: (f != 0).nullif(f == 0), id="nullif_negate_boolean"),
    ],
)
def test_nullif_ifnull(tpch_lineitem, expr_fn, snapshot):
    expr = expr_fn(tpch_lineitem.l_quantity)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize(
    "expr_fn",
    [
        pytest.param(lambda t: t.l_quantity.fillna(0), id="fillna_l_quantity"),
        pytest.param(
            lambda t: t.l_extendedprice.fillna(0), id="fillna_l_extendedprice"
        ),
        pytest.param(
            lambda t: t.l_extendedprice.fillna(0.0), id="fillna_l_extendedprice_double"
        ),
    ],
)
def test_decimal_fillna_cast_arg(tpch_lineitem, expr_fn, snapshot):
    expr = expr_fn(tpch_lineitem)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


def test_identical_to(mockcon, snapshot):
    t = mockcon.table("functional_alltypes")
    expr = t.tinyint_col.identical_to(t.double_col).name("tmp")
    result = ImpalaCompiler.to_sql(expr)
    snapshot.assert_match(result, "out.sql")


def test_identical_to_special_case(snapshot):
    expr = ibis.NA.cast("int64").identical_to(ibis.NA.cast("int64")).name("tmp")
    result = ImpalaCompiler.to_sql(expr)
    snapshot.assert_match(result, "out.sql")
