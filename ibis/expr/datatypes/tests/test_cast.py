from __future__ import annotations

import pytest

import ibis.expr.datatypes as dt


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (dt.string, dt.uuid),
        (dt.uuid, dt.string),
        (dt.null, dt.date),
        (dt.int8, dt.int64),
        (dt.int8, dt.Decimal(12, 2)),
        (dt.int16, dt.uint64),
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
    assert dt.castable(source, target)


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
    assert not dt.castable(source, target)


@pytest.mark.parametrize(
    ("source", "target", "value"),
    [(dt.int8, dt.boolean, 0), (dt.int8, dt.boolean, 1)],
)
def test_implicitly_castable_values(source, target, value):
    assert dt.castable(source, target, value=value)


@pytest.mark.parametrize(
    ("source", "target", "value"),
    [(dt.int8, dt.boolean, 3), (dt.int8, dt.boolean, -1)],
)
def test_implicitly_uncastable_values(source, target, value):
    assert not dt.castable(source, target, value=value)
