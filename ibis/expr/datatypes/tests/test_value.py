import decimal
import enum
from collections import OrderedDict
from datetime import date, datetime, timedelta

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
        (date.today(), dt.date),
        (datetime.now(), dt.timestamp),
        (timedelta(days=3), dt.Interval(unit='D')),
        (timedelta(seconds=9), dt.Interval(unit='s')),
        (timedelta(microseconds=15), dt.Interval(unit='us')),
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
        ({1, 5, 6}, dt.Array(dt.int8)),
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
    ],
)
def test_infer_dtype(value, expected_dtype):
    assert dt.infer(value) == expected_dtype


def test_infer_mixed_type_fails():
    data = [1, 'a']
    with pytest.raises(TypeError):
        dt.infer(data)


def test_infer_timestamp_with_tz():
    now_raw = datetime.utcnow()
    now_utc = pytz.utc.localize(now_raw)
    assert now_utc.tzinfo == pytz.UTC
    assert dt.infer(now_utc).timezone == str(pytz.UTC)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2019-01-01", datetime(2019, 1, 1)),
        ("2019-01-01 00:00:00", datetime(2019, 1, 1)),
        ("2019-01-01 01:02:03.000004", datetime(2019, 1, 1, 1, 2, 3, 4)),
        (
            "2019-01-01 01:02:03.000004+00:00",
            datetime(2019, 1, 1, 1, 2, 3, 4, tzinfo=pytz.utc),
        ),
        (
            "2019-01-01 01:02:03.000004+01:00",
            datetime(2019, 1, 1, 1, 2, 3, 4, tzinfo=pytz.FixedOffset(60)),
        ),
        (
            "2019-01-01 01:02:03.000004-01:00",
            datetime(2019, 1, 1, 1, 2, 3, 4, tzinfo=pytz.FixedOffset(-60)),
        ),
        (
            "2019-01-01 01:02:03.000004+01",
            datetime(2019, 1, 1, 1, 2, 3, 4, tzinfo=pytz.FixedOffset(60)),
        ),
        (datetime(2019, 1, 1), datetime(2019, 1, 1)),
        (datetime(2019, 1, 1, 1, 2, 3, 4), datetime(2019, 1, 1, 1, 2, 3, 4)),
    ],
)
def test_normalize_timestamp(value, expected):
    normalized = dt.normalize(dt.timestamp, value)
    assert normalized == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2019-01-01", date(2019, 1, 1)),
        ("2019-01-01 00:00:00", date(2019, 1, 1)),
        ("2019-01-01 01:02:03.000004", date(2019, 1, 1)),
        (datetime(2019, 1, 1), date(2019, 1, 1)),
        (datetime(2019, 1, 1, 1, 2, 3, 4), date(2019, 1, 1)),
    ],
)
def test_normalize_date(value, expected):
    normalized = dt.normalize(dt.date, value)
    assert normalized == expected
