from __future__ import annotations

import pytest

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.patterns import (
    CoercedTo,
    GenericCoercedTo,
    Pattern,
    ValidationError,
)

# TODO(kszucs): actually we should only allow datatype classes not instances


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        (1, dt.int8),
        (1.0, dt.double),
        (True, dt.boolean),
        ("foo", dt.string),
        (b"foo", dt.binary),
        ((1, 2), dt.Array(dt.int8)),
    ],
)
def test_literal_coercion_type_inference(value, dtype):
    assert ops.Literal.__coerce__(value) == ops.Literal(value, dtype)
    assert ops.Literal.__coerce__(value, dtype) == ops.Literal(value, dtype)


def test_coerced_to_literal():
    p = CoercedTo(ops.Literal)
    one = ops.Literal(1, dt.int8)
    assert p.validate(ops.Literal(1, dt.int8), {}) == one
    assert p.validate(1, {}) == one
    assert p.validate(False, {}) == ops.Literal(False, dt.boolean)

    p = GenericCoercedTo(ops.Literal[dt.Int8])
    assert p.validate(ops.Literal(1, dt.int8), {}) == one

    p = Pattern.from_typehint(ops.Literal[dt.Int8])
    assert p == GenericCoercedTo(ops.Literal[dt.Int8])

    one = ops.Literal(1, dt.int16)
    with pytest.raises(ValidationError):
        p.validate(ops.Literal(1, dt.int16), {})


def test_coerced_to_value():
    one = ops.Literal(1, dt.int8)

    p = Pattern.from_typehint(ops.Value)
    assert p.validate(1, {}) == one

    p = Pattern.from_typehint(ops.Value[dt.Int8, ds.Any])
    assert p.validate(1, {}) == one

    p = Pattern.from_typehint(ops.Value[dt.Int8, ds.Scalar])
    assert p.validate(1, {}) == one

    p = Pattern.from_typehint(ops.Value[dt.Int8, ds.Columnar])
    with pytest.raises(ValidationError):
        p.validate(1, {})

    # dt.Integer is not instantiable so it will be only used for checking
    # that the produced literal has any integer datatype
    p = Pattern.from_typehint(ops.Value[dt.Integer, ds.Any])
    assert p.validate(1, {}) == one

    # same applies here, the coercion itself will use only the inferred datatype
    # but then the result is checked against the given typehint
    p = Pattern.from_typehint(ops.Value[dt.Int8 | dt.Int16, ds.Any])
    assert p.validate(1, {}) == one
    assert p.validate(128, {}) == ops.Literal(128, dt.int16)

    p1 = Pattern.from_typehint(ops.Value[dt.Int8, ds.Any])
    p2 = Pattern.from_typehint(ops.Value[dt.Int16, ds.Scalar])
    assert p1.validate(1, {}) == one
    # this is actually supported by creating an explicit dtype
    # in Value.__coerce__ based on the `T` keyword argument
    assert p2.validate(1, {}) == ops.Literal(1, dt.int16)
    assert p2.validate(128, {}) == ops.Literal(128, dt.int16)

    p = p1 | p2
    assert p.validate(1, {}) == one


@pytest.mark.pandas
def test_coerced_to_interval_value():
    import pandas as pd

    p = Pattern.from_typehint(ops.Value[dt.Interval, ds.Any])

    value = pd.Timedelta("1s")
    result = p.match(value, {})
    assert result.value == 1
    assert result.dtype == dt.Interval("s")

    value = pd.Timedelta("1h 1m 1s")
    result = p.match(value, {})
    assert result.value == 3661
    assert result.dtype == dt.Interval("s")
