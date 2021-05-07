import operator

import pytest

import ibis.expr.api as api
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir


def test_type_metadata(lineitem):
    col = lineitem.l_extendedprice
    assert isinstance(col, ir.DecimalColumn)

    assert col.type() == dt.Decimal(12, 2)


def test_cast_scalar_to_decimal():
    val = api.literal('1.2345')

    casted = val.cast('decimal(15,5)')
    assert isinstance(casted, ir.DecimalScalar)
    assert casted.type() == dt.Decimal(15, 5)


def test_decimal_sum_type(lineitem):
    col = lineitem.l_extendedprice
    result = col.sum()
    assert isinstance(result, ir.DecimalScalar)
    assert result.type() == dt.Decimal(38, col.type().scale)


@pytest.mark.parametrize('func', ['mean', 'max', 'min'])
def test_decimal_aggregate_function_type(lineitem, func):
    col = lineitem.l_extendedprice
    method = operator.methodcaller(func)
    result = method(col)
    assert isinstance(result, ir.DecimalScalar)
    assert result.type() == col.type()


def test_where(lineitem):
    table = lineitem

    q = table.l_quantity
    expr = api.where(table.l_discount > 0, q * table.l_discount, api.null())

    assert isinstance(expr, ir.DecimalColumn)

    expr = api.where(
        table.l_discount > 0, (q * table.l_discount).sum(), api.null()
    )
    assert isinstance(expr, ir.DecimalColumn)

    expr = api.where(
        table.l_discount.sum() > 0, (q * table.l_discount).sum(), api.null()
    )
    assert isinstance(expr, ir.DecimalScalar)


def test_fillna(lineitem):
    expr = lineitem.l_extendedprice.fillna(0)
    assert isinstance(expr, ir.DecimalColumn)

    expr = lineitem.l_extendedprice.fillna(lineitem.l_quantity)
    assert isinstance(expr, ir.DecimalColumn)


def test_precision_scale(lineitem):
    col = lineitem.l_extendedprice

    p = col.precision()
    s = col.scale()

    assert isinstance(p, ir.IntegerValue)
    assert isinstance(p.op(), ops.DecimalPrecision)

    assert isinstance(s, ir.IntegerValue)
    assert isinstance(s.op(), ops.DecimalScale)


@pytest.mark.parametrize(
    ('precision', 'scale'),
    [
        (-1, 3),  # negative precision
        (0, 1),  # zero precision
        (12, 38),  # precision less than scale
        (33, -1),  # negative scale
    ],
)
def test_invalid_precision_scale_combo(precision, scale):
    with pytest.raises(ValueError):
        dt.Decimal(precision, scale)


@pytest.mark.parametrize(
    ('precision', 'scale'),
    [(38.1, 3), (38, 3.1)],  # non integral precision  # non integral scale
)
def test_invalid_precision_scale_type(precision, scale):
    with pytest.raises(TypeError):
        dt.Decimal(precision, scale)


def test_decimal_str(lineitem):
    col = lineitem.l_extendedprice
    t = col.type()
    assert str(t) == 'decimal({:d}, {:d})'.format(t.precision, t.scale)


def test_decimal_repr(lineitem):
    col = lineitem.l_extendedprice
    t = col.type()
    expected = 'Decimal(precision={:d}, scale={:d}, nullable=True)'.format(
        t.precision, t.scale
    )
    assert repr(t) == expected
