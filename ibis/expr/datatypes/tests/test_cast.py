from __future__ import annotations

import hypothesis as h
import pytest

import ibis.expr.datatypes as dt
import ibis.tests.strategies as ibst


@h.given(ibst.signed_integer_dtypes(), ibst.signed_integer_dtypes())
def test_signed_integer_castable_to_signed_integer(from_, to):
    if from_.nbytes > to.nbytes:
        assert not from_.castable(to)
    else:
        assert from_.castable(to)


@h.given(ibst.unsigned_integer_dtypes(), ibst.unsigned_integer_dtypes())
def test_unsigned_integer_castable_to_unsigned_integer(from_, to):
    if from_.nbytes > to.nbytes:
        assert not from_.castable(to)
    else:
        assert from_.castable(to)


# TODO(kszucs): unsigned to signed implicit cast is currently allowed to not
# break the integral promotion rule logic in rules.py
# @h.given(ibst.signed_integer_dtypes(), ibst.unsigned_integer_dtypes())
# def test_signed_integer_not_castable_to_unsigned_integer(from_, to):
#     assert not from_.castable(to)


@h.given(ibst.integer_dtypes(), ibst.floating_dtypes())
def test_integer_castable_to_floating(from_, to):
    assert from_.castable(to)


# TODO(kszucs): we could be more pedantic here considering the precision
@h.given(ibst.integer_dtypes(), ibst.decimal_dtypes())
def test_integer_castable_to_decimal(from_, to):
    assert from_.castable(to)


@h.given(ibst.integer_dtypes(), ibst.boolean_dtype())
def test_integer_castable_to_boolean(from_, to):
    assert from_.castable(to, value=0)
    assert from_.castable(to, value=1)
    assert not from_.castable(to, value=-1)
    assert not from_.castable(to)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (dt.string, dt.uuid),
        (dt.uuid, dt.string),
        (dt.null, dt.date),
        (dt.int8, dt.int64),
        (dt.int8, dt.Decimal(12, 2)),
        # (dt.int16, dt.uint64),
        (dt.int32, dt.int32),
        (dt.int32, dt.int64),
        (dt.uint32, dt.uint64),
        (dt.uint32, dt.int64),
        (dt.uint32, dt.Decimal(12, 2)),
        (dt.uint32, dt.float32),
        (dt.uint32, dt.float64),
        (dt.Interval("s"), dt.Interval("s")),
    ],
)
def test_implicitly_castable_primitives(source, target):
    assert source.castable(target)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (dt.string, dt.null),
        (dt.int32, dt.int16),
        (dt.int32, dt.uint16),
        (dt.uint64, dt.int16),
        (dt.uint64, dt.uint16),
        # (dt.uint64, dt.int64), TODO: https://github.com/ibis-project/ibis/issues/7331
        (dt.Decimal(12, 2), dt.int32),
        (dt.timestamp, dt.boolean),
        (dt.Interval("s"), dt.Interval("ns")),
    ],
)
def test_implicitly_uncastable_primitives(source, target):
    assert not source.castable(target)


@pytest.mark.parametrize("value", [0, 1])
def test_implicitly_castable_int_to_bool(value):
    assert dt.int8.castable(dt.boolean, value=value)


@pytest.mark.parametrize(
    ("source", "target", "value"),
    [(dt.int8, dt.boolean, 3), (dt.int8, dt.boolean, -1)],
)
def test_implicitly_uncastable_values(source, target, value):
    assert not source.castable(target, value=value)


def test_struct_different_fields():
    x = dt.Struct({"x": dt.int32})
    x2 = dt.Struct({"x": dt.int64})
    y = dt.Struct({"y": dt.int32})
    xy = dt.Struct({"x": dt.int32, "y": dt.int32})

    # Can upcast int32 to int64, but not other way
    assert x.castable(x2)
    assert not x2.castable(x)
    # Can remove a field, but not add one
    assert xy.castable(x)
    assert not x.castable(xy)

    # Missing fields entirely from each other
    assert not x.castable(y)
    assert not y.castable(x)


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        # Fixed precision
        ((12, 2), (12, 3), True),
        ((12, 3), (12, 2), False),
        # Fixed scale
        ((12, 2), (13, 2), True),
        ((13, 2), (12, 2), False),
        # Equal
        ((12, 2), (12, 2), True),
        # Not equal
        ((12, 2), (13, 3), True),
        ((13, 2), (12, 3), False),
        ((13, 2), (12, 1), False),
    ],
)
def test_castable_decimal_to_decimal(source, target, expected):
    left = dt.Decimal(*source)
    right = dt.Decimal(*target)
    assert left.castable(right) is expected
