from __future__ import annotations

import operator

import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis.common.annotations import ValidationError


def test_type_metadata(lineitem):
    col = lineitem.l_extendedprice
    assert isinstance(col, ir.DecimalColumn)

    assert col.type() == dt.Decimal(12, 2)


def test_cast_scalar_to_decimal():
    val = ibis.literal("1.2345")

    casted = val.cast("decimal(15,5)")
    assert isinstance(casted, ir.DecimalScalar)
    assert casted.type() == dt.Decimal(15, 5)


def test_decimal_sum_type(lineitem):
    col = lineitem.l_extendedprice
    result = col.sum()
    assert isinstance(result, ir.DecimalScalar)
    assert result.type() == dt.Decimal(38, 2)


def test_promote_decimal_type_mul(lineitem):
    col_1 = lineitem.l_extendedprice
    col_2 = lineitem.l_discount
    result = col_1 * col_2
    assert result.type().precision == 24
    assert result.type().scale == 4


def test_promote_decimal_type_add(lineitem):
    col_1 = lineitem.l_extendedprice
    col_2 = lineitem.l_discount
    result = col_1 + col_2
    assert result.type().precision == 13
    assert result.type().scale == 2


def test_promote_decimal_type_mod(lineitem):
    col_1 = lineitem.l_extendedprice
    col_2 = lineitem.l_discount
    result = col_1 % col_2
    assert result.type().precision == 12
    assert result.type().scale == 2


def test_promote_decimal_type_max():
    t = ibis.table([("a", "decimal(31, 3)"), ("b", "decimal(31, 3)")], "t")
    result = t.a * t.b
    assert result.type().precision == 31
    assert result.type().scale == 6


@pytest.mark.parametrize(
    "precision, scale, expected",
    [
        (None, None, (None, None)),
        (38, 2, (38, 2)),
        (16, 2, (38, 2)),
        (39, 3, (39, 3)),
    ],
)
def test_decimal_sum_type_precision(precision, scale, expected):
    t = ibis.table([("l_extendedprice", dt.Decimal(precision, scale))], name="t")
    col = t.l_extendedprice
    result = col.sum()
    assert isinstance(result, ir.DecimalScalar)
    assert result.type() == dt.Decimal(*expected)


@pytest.mark.parametrize("func", ["mean", "max", "min"])
def test_decimal_aggregate_function_type(lineitem, func):
    col = lineitem.l_extendedprice
    method = operator.methodcaller(func)
    result = method(col)
    assert isinstance(result, ir.DecimalScalar)
    assert result.type() == col.type()


def test_ifelse(lineitem):
    table = lineitem

    q = table.l_quantity
    expr = ibis.ifelse(table.l_discount > 0, q * table.l_discount, ibis.null())

    assert isinstance(expr, ir.DecimalColumn)

    expr = ibis.ifelse(table.l_discount > 0, (q * table.l_discount).sum(), ibis.null())
    assert isinstance(expr, ir.DecimalColumn)

    expr = ibis.ifelse(
        table.l_discount.sum() > 0, (q * table.l_discount).sum(), ibis.null()
    )
    assert isinstance(expr, ir.DecimalScalar)


def test_fillna(lineitem):
    expr = lineitem.l_extendedprice.fillna(0)
    assert isinstance(expr, ir.DecimalColumn)

    expr = lineitem.l_extendedprice.fillna(lineitem.l_quantity)
    assert isinstance(expr, ir.DecimalColumn)


@pytest.mark.parametrize(
    ("precision", "scale"),
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
    ("precision", "scale"),
    [(38.1, 3), (38, 3.1)],  # non integral precision  # non integral scale
)
def test_invalid_precision_scale_type(precision, scale):
    with pytest.raises(ValidationError):
        dt.Decimal(precision, scale)


def test_decimal_str(lineitem):
    col = lineitem.l_extendedprice
    t = col.type()
    assert str(t) == f"decimal({t.precision:d}, {t.scale:d})"


def test_decimal_repr(lineitem):
    col = lineitem.l_extendedprice
    t = col.type()
    expected = f"Decimal(precision={t.precision:d}, scale={t.scale:d}, nullable=True)"
    assert repr(t) == expected
