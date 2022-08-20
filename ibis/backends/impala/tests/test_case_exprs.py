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
    return (
        table.g.case()
        .when('foo', 'bar')
        .when('baz', 'qux')
        .else_('default')
        .end()
    )


@pytest.fixture
def search_case(table):
    t = table
    return ibis.case().when(t.f > 0, t.d * 2).when(t.c < 0, t.a * 2).end()


@pytest.fixture
def tpch_lineitem(con):
    return con.table('tpch_lineitem')


def test_isnull_1_0(table):
    expr = table.g.isnull().ifelse(1, 0)

    result = translate(expr)
    expected = 'if(`g` IS NULL, 1, 0)'
    assert result == expected

    # inside some other function
    result = translate(expr.sum())
    expected = 'sum(if(`g` IS NULL, 1, 0))'
    assert result == expected


def test_simple_case(simple_case):
    expr = simple_case
    result = translate(expr)
    expected = """CASE `g`
  WHEN 'foo' THEN 'bar'
  WHEN 'baz' THEN 'qux'
  ELSE 'default'
END"""
    assert result == expected


def test_search_case(search_case):
    expr = search_case
    result = translate(expr)
    expected = """CASE
  WHEN `f` > 0 THEN `d` * 2
  WHEN `c` < 0 THEN `a` * 2
  ELSE CAST(NULL AS bigint)
END"""
    assert result == expected


def test_where_use_if(table):
    expr = ibis.where(table.f > 0, table.e, table.a)
    assert isinstance(expr, ir.FloatingValue)

    result = translate(expr)
    expected = "if(`f` > 0, `e`, `a`)"
    assert result == expected


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(
            lambda f: f.nullif(f),
            'nullif(`l_quantity`, `l_quantity`)',
            id="nullif_input",
        ),
        pytest.param(
            lambda f: (f == 0).nullif(f == 0),
            'nullif(`l_quantity` = 0, `l_quantity` = 0)',
            id="nullif_boolean",
        ),
        pytest.param(
            lambda f: (f != 0).nullif(f == 0),
            'nullif(`l_quantity` != 0, `l_quantity` = 0)',
            id="nullif_negate_boolean",
        ),
    ],
)
def test_nullif_ifnull(tpch_lineitem, expr_fn, expected):
    expr = expr_fn(tpch_lineitem.l_quantity)
    result = translate(expr)
    assert result == expected


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(
            lambda t: t.l_quantity.fillna(0),
            'isnull(`l_quantity`, CAST(0 AS decimal(12, 2)))',
            id="fillna_l_quantity",
        ),
        pytest.param(
            lambda t: t.l_extendedprice.fillna(0),
            'isnull(`l_extendedprice`, CAST(0 AS decimal(12, 2)))',
            id="fillna_l_extendedprice",
        ),
        pytest.param(
            lambda t: t.l_extendedprice.fillna(0.0),
            'isnull(`l_extendedprice`, 0.0)',
            id="fillna_l_extendedprice_double",
        ),
    ],
)
def test_decimal_fillna_cast_arg(tpch_lineitem, expr_fn, expected):
    expr = expr_fn(tpch_lineitem)
    result = translate(expr)
    assert result == expected


def test_identical_to(con):
    t = con.table('functional_alltypes')
    expr = t.tinyint_col.identical_to(t.double_col)
    result = ImpalaCompiler.to_sql(expr)
    expected = """\
SELECT `tinyint_col` IS NOT DISTINCT FROM `double_col` AS `tmp`
FROM ibis_testing.`functional_alltypes`"""
    assert result == expected


def test_identical_to_special_case():
    expr = ibis.NA.cast('int64').identical_to(ibis.NA.cast('int64'))
    result = ImpalaCompiler.to_sql(expr)
    assert result == 'SELECT TRUE AS `tmp`'
