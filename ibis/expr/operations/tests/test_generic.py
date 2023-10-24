from __future__ import annotations

from functools import partial
from typing import Union

import pytest

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.annotations import ValidationError
from ibis.common.patterns import NoMatch, Pattern

one = ops.Literal(1, dt.int8)


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        (1, dt.Int8),
        (1.0, dt.Float64),
        (True, dt.Boolean),
        ("foo", dt.String),
        (b"foo", dt.Binary),
        ((1, 2), dt.Array[dt.Int8]),
    ],
)
def test_literal_coercion_type_inference(value, dtype):
    assert ops.Literal.__coerce__(value) == ops.Literal(value, dtype)
    assert ops.Literal.__coerce__(value, dtype) == ops.Literal(value, dtype)


@pytest.mark.parametrize(
    ("typehint", "value", "expected"),
    [
        (ops.Literal, 1, one),
        (ops.Literal, one, one),
        (ops.Literal, False, ops.Literal(False, dt.boolean)),
        (ops.Literal[dt.Int8], 1, one),
        (ops.Literal[dt.Int16], 1, ops.Literal(1, dt.int16)),
        (ops.Literal[dt.Int8], ops.Literal(1, dt.int16), NoMatch),
        (ops.Literal[dt.Integer], 1, ops.Literal(1, dt.int8)),
        (ops.Literal[dt.Floating], 1, ops.Literal(1, dt.float64)),
        (ops.Literal[dt.Float32], 1.0, ops.Literal(1.0, dt.float32)),
    ],
)
def test_coerced_to_literal(typehint, value, expected):
    pat = Pattern.from_typehint(typehint)
    assert pat.match(value, {}) == expected


@pytest.mark.parametrize(
    ("typehint", "value", "expected"),
    [
        (ops.Value, 1, one),
        (ops.Value[dt.Int8], 1, one),
        (ops.Value[dt.Int8, ds.Any], 1, one),
        (ops.Value[dt.Int8, ds.Scalar], 1, one),
        (ops.Value[dt.Int8, ds.Columnar], 1, NoMatch),
        # dt.Integer is not instantiable so it will be only used for checking
        # that the produced literal has any integer datatype
        (ops.Value[dt.Integer], 1, one),
        # same applies here, the coercion itself will use only the inferred datatype
        # but then the result is checked against the given typehint
        (ops.Value[dt.Int8 | dt.Int16], 1, one),
        (Union[ops.Value[dt.Int8], ops.Value[dt.Int16]], 1, one),
        (ops.Value[dt.Int8 | dt.Int16], 128, ops.Literal(128, dt.int16)),
        (
            Union[ops.Value[dt.Int8], ops.Value[dt.Int16]],
            128,
            ops.Literal(128, dt.int16),
        ),
        (ops.Value[dt.Int8 | dt.Int16], 128, ops.Literal(128, dt.int16)),
        (
            Union[ops.Value[dt.Int8], ops.Value[dt.Int16]],
            128,
            ops.Literal(128, dt.int16),
        ),
        (ops.Value[dt.Int8], 128, NoMatch),
        # this is actually supported by creating an explicit dtype
        # in Value.__coerce__ based on the `T` keyword argument
        (ops.Value[dt.Int16, ds.Scalar], 1, ops.Literal(1, dt.int16)),
        (ops.Value[dt.Int16, ds.Scalar], 128, ops.Literal(128, dt.int16)),
        # equivalent with ops.Value[dt.Int8 | dt.Int16]
        (Union[ops.Value[dt.Int8], ops.Value[dt.Int16]], 1, one),
        # when expecting floating point values given an integer value it will
        # be coerced to float64
        (ops.Value[dt.Floating], 1, ops.Literal(1, dt.float64)),
    ],
)
def test_coerced_to_value(typehint, value, expected):
    pat = Pattern.from_typehint(typehint)
    assert pat.match(value, {}) == expected


@pytest.mark.pandas
def test_coerced_to_interval_value():
    import pandas as pd

    expected = ops.Literal(1, dt.Interval("s"))
    pat = Pattern.from_typehint(ops.Value[dt.Interval])
    assert pat.match(pd.Timedelta("1s"), {}) == expected

    expected = ops.Literal(3661, dt.Interval("s"))
    assert pat.match(pd.Timedelta("1h 1m 1s"), {}) == expected


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (partial(ops.Literal, 1), "missing_a_required_argument"),
        (partial(ops.Literal, 1, dt.int8, "foo"), "too_many_positional_arguments"),
        (partial(ops.Literal, 1, dt.int8, name="foo"), "got_an_unexpected_keyword"),
        (
            partial(ops.Literal, 1, dt.int8, dtype=dt.int16),
            "multiple_values_for_argument",
        ),
        (partial(ops.Literal, 1, 4), "invalid_dtype"),
    ],
)
def test_error_message_when_constructing_literal(call, error, snapshot):
    with pytest.raises(ValidationError) as exc:
        call()
    snapshot.assert_match(str(exc.value), f"{error}.txt")
