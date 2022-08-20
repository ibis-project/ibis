import pytest

import ibis.expr.types as ir
from ibis import literal as L
from ibis.backends.impala.tests.conftest import translate


@pytest.fixture(scope="module")
def table(mockcon):
    return mockcon.table("functional_alltypes")


IBIS_TO_SQL_NAMES = {
    "log": "ln",
    "approx_median": "appx_median",
    "approx_nunique": "ndv",
}


@pytest.mark.parametrize(
    "ibis_name",
    [
        'abs',
        'ceil',
        'floor',
        'exp',
        'sqrt',
        'log',
        'approx_median',
        'approx_nunique',
        'ln',
        'log2',
        'log10',
        'nullifzero',
        'zeroifnull',
    ],
)
@pytest.mark.parametrize("cname", ["double_col", "int_col"])
def test_numeric_unary_builtins(ibis_name, cname, table):
    sql_name = IBIS_TO_SQL_NAMES.get(ibis_name, ibis_name)

    method = getattr(table[cname], ibis_name)
    expr = method()
    expected = f'{sql_name}(`{cname}`)'

    result = translate(expr)
    assert result == expected


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(
            lambda t: t.double_col.log(5),
            'log(5, `double_col`)',
            id="log_with_base",
        ),
        pytest.param(
            lambda t: t.double_col.round(),
            'round(`double_col`)',
            id="round_no_args",
        ),
        pytest.param(
            lambda t: t.double_col.round(0),
            'round(`double_col`, 0)',
            id="round_zero",
        ),
        pytest.param(
            lambda t: t.double_col.round(2),
            'round(`double_col`, 2)',
            id="round_two",
        ),
        pytest.param(
            lambda t: t.double_col.round(t.tinyint_col),
            'round(`double_col`, `tinyint_col`)',
            id="round_expr",
        ),
        pytest.param(
            lambda t: t.tinyint_col.sign(),
            'CAST(sign(`tinyint_col`) AS tinyint)',
            id="sign_tinyint",
        ),
        pytest.param(
            lambda t: t.float_col.sign(),
            'sign(`float_col`)',
            id="sign_float",
        ),
        pytest.param(
            lambda t: t.double_col.sign(),
            'CAST(sign(`double_col`) AS double)',
            id="sign_double",
        ),
    ],
)
def test_numeric(expr_fn, expected, table):
    expr = expr_fn(table)
    result = translate(expr)
    assert result == expected


def test_hash(table):
    expr = table.int_col.hash()
    assert isinstance(expr, ir.IntegerColumn)
    assert isinstance(table.int_col.sum().hash(), ir.IntegerScalar)

    expr = table.int_col.hash()
    expected = 'fnv_hash(`int_col`)'
    assert translate(expr) == expected


@pytest.mark.parametrize(
    ("expr_fn", "func_name"),
    [
        pytest.param(
            lambda t: t.double_col.sum(where=t.bigint_col < 70),
            'sum',
            id='sum',
        ),
        pytest.param(
            lambda t: t.double_col.count(where=t.bigint_col < 70),
            'count',
            id='count',
        ),
        pytest.param(
            lambda t: t.double_col.mean(where=t.bigint_col < 70),
            'avg',
            id='avg',
        ),
        pytest.param(
            lambda t: t.double_col.max(where=t.bigint_col < 70),
            'max',
            id='max',
        ),
        pytest.param(
            lambda t: t.double_col.min(where=t.bigint_col < 70),
            'min',
            id='min',
        ),
        pytest.param(
            lambda t: t.double_col.std(where=t.bigint_col < 70),
            'stddev_samp',
            id='stddev_samp',
        ),
        pytest.param(
            lambda t: t.double_col.std(where=t.bigint_col < 70, how='pop'),
            'stddev_pop',
            id='stddev_pop',
        ),
        pytest.param(
            lambda t: t.double_col.var(where=t.bigint_col < 70),
            'var_samp',
            id='var_samp',
        ),
        pytest.param(
            lambda t: t.double_col.var(where=t.bigint_col < 70, how='pop'),
            'var_pop',
            id='var_pop',
        ),
    ],
)
def test_reduction_where(table, expr_fn, func_name):
    expr = expr_fn(table)
    result = translate(expr)
    expected = f'{func_name}(if(`bigint_col` < 70, `double_col`, NULL))'
    assert result == expected


@pytest.mark.parametrize("method_name", ["sum", "count", "mean", "max", "min"])
def test_reduction_invalid_where(table, method_name):
    condbad_literal = L('T')
    reduction = getattr(table.double_col, method_name)

    with pytest.raises(TypeError):
        reduction(where=condbad_literal)
