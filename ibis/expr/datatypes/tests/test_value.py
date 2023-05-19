import datetime
import decimal
import enum
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
import pytz

import ibis.expr.datatypes as dt


class Foo(enum.Enum):
    a = 1
    b = 2


@pytest.mark.parametrize(
    ('value', 'expected_dtype'),
    [
        (None, dt.null),
        (False, dt.boolean),
        (True, dt.boolean),
        ('foo', dt.string),
        (b'fooblob', dt.binary),
        (datetime.date.today(), dt.date),
        (datetime.datetime.now(), dt.timestamp),
        (datetime.timedelta(days=3), dt.Interval(unit='D')),
        (pd.Timedelta('5 hours'), dt.Interval(unit='h')),
        (pd.Timedelta('7 minutes'), dt.Interval(unit='m')),
        (datetime.timedelta(seconds=9), dt.Interval(unit='s')),
        (pd.Timedelta('11 milliseconds'), dt.Interval(unit='ms')),
        (datetime.timedelta(microseconds=15), dt.Interval(unit='us')),
        (pd.Timedelta('17 nanoseconds'), dt.Interval(unit='ns')),
        # numeric types
        (5, dt.int8),
        (5, dt.int8),
        (127, dt.int8),
        (128, dt.int16),
        (32767, dt.int16),
        (32768, dt.int32),
        (2147483647, dt.int32),
        (2147483648, dt.int64),
        (-5, dt.int8),
        (-128, dt.int8),
        (-129, dt.int16),
        (-32769, dt.int32),
        (-2147483649, dt.int64),
        (1.5, dt.double),
        (decimal.Decimal(1.5), dt.decimal),
        # parametric types
        (list('abc'), dt.Array(dt.string)),
        (set('abc'), dt.Array(dt.string)),
        ({1, 5, 5, 6}, dt.Array(dt.int8)),
        (frozenset(list('abc')), dt.Array(dt.string)),
        ([1, 2, 3], dt.Array(dt.int8)),
        ([1, 128], dt.Array(dt.int16)),
        ([1, 128, 32768], dt.Array(dt.int32)),
        ([1, 128, 32768, 2147483648], dt.Array(dt.int64)),
        ({'a': 1, 'b': 2, 'c': 3}, dt.Map(dt.string, dt.int8)),
        ({1: 2, 3: 4, 5: 6}, dt.Map(dt.int8, dt.int8)),
        (
            {'a': [1.0, 2.0], 'b': [], 'c': [3.0]},
            dt.Map(dt.string, dt.Array(dt.double)),
        ),
        (
            OrderedDict(
                [
                    ('a', 1),
                    ('b', list('abc')),
                    ('c', OrderedDict([('foo', [1.0, 2.0])])),
                ]
            ),
            dt.Struct.from_tuples(
                [
                    ('a', dt.int8),
                    ('b', dt.Array(dt.string)),
                    (
                        'c',
                        dt.Struct.from_tuples([('foo', dt.Array(dt.double))]),
                    ),
                ]
            ),
        ),
        (Foo.a, dt.Enum()),
        # numpy types
        (np.int8(5), dt.int8),
        (np.int16(-1), dt.int16),
        (np.int32(2), dt.int32),
        (np.int64(-5), dt.int64),
        (np.uint8(5), dt.uint8),
        (np.uint16(50), dt.uint16),
        (np.uint32(500), dt.uint32),
        (np.uint64(5000), dt.uint64),
        (np.float32(5.5), dt.float32),
        (np.float64(5.55), dt.float64),
        (np.bool_(True), dt.boolean),
        (np.bool_(False), dt.boolean),
        # pandas types
        (
            pd.Timestamp('2015-01-01 12:00:00', tz='US/Eastern'),
            dt.Timestamp('US/Eastern'),
        ),
    ],
)
def test_infer_dtype(value, expected_dtype):
    assert dt.infer(value) == expected_dtype


def test_infer_mixed_type_fails():
    data = [1, 'a']
    with pytest.raises(TypeError):
        dt.infer(data)


def test_infer_timestamp_with_tz():
    now_raw = datetime.datetime.utcnow()
    now_utc = pytz.utc.localize(now_raw)
    assert now_utc.tzinfo == pytz.UTC
    assert dt.infer(now_utc).timezone == str(pytz.UTC)
