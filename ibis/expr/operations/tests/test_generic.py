from __future__ import annotations

from functools import partial
from typing import Union

import pytest

import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.common.annotations import ValidationError
from ibis.common.patterns import NoMatch, match


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
    one = ops.Literal(1, dt.int8)

    assert match(ops.Literal, 1) == one
    assert match(ops.Literal, one) == one
    assert match(ops.Literal, False) == ops.Literal(False, dt.boolean)

    assert match(ops.Literal[dt.Int8], 1) == one
    assert match(ops.Literal[dt.Int16], 1) == ops.Literal(1, dt.int16)

    assert match(ops.Literal[dt.Int8], ops.Literal(1, dt.int16)) is NoMatch


def test_coerced_to_value():
    one = ops.Literal(1, dt.int8)

    assert match(ops.Value, 1) == one
    assert match(ops.Value[dt.Int8], 1) == one
    assert match(ops.Value[dt.Int8, ds.Any], 1) == one
    assert match(ops.Value[dt.Int8, ds.Scalar], 1) == one
    assert match(ops.Value[dt.Int8, ds.Columnar], 1) is NoMatch

    # dt.Integer is not instantiable so it will be only used for checking
    # that the produced literal has any integer datatype
    assert match(ops.Value[dt.Integer], 1) == one

    # same applies here, the coercion itself will use only the inferred datatype
    # but then the result is checked against the given typehint
    assert match(ops.Value[dt.Int8 | dt.Int16], 1) == one
    assert match(ops.Value[dt.Int8 | dt.Int16], 128) == ops.Literal(128, dt.int16)
    assert match(ops.Value[dt.Int8], 128) is NoMatch

    # this is actually supported by creating an explicit dtype
    # in Value.__coerce__ based on the `T` keyword argument
    assert match(ops.Value[dt.Int16, ds.Scalar], 1) == ops.Literal(1, dt.int16)
    assert match(ops.Value[dt.Int16, ds.Scalar], 128) == ops.Literal(128, dt.int16)

    # equivalent with ops.Value[dt.Int8 | dt.Int16]
    assert match(Union[ops.Value[dt.Int8], ops.Value[dt.Int16]], 1) == one


@pytest.mark.pandas
def test_coerced_to_interval_value():
    import pandas as pd

    expected = ops.Literal(1, dt.Interval("s"))
    assert match(ops.Value[dt.Interval], pd.Timedelta("1s")) == expected

    expected = ops.Literal(3661, dt.Interval("s"))
    assert match(ops.Value[dt.Interval], pd.Timedelta("1h 1m 1s")) == expected


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
