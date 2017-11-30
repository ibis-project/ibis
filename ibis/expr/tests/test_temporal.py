# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import datetime

from ibis.common import IbisError, IbisTypeError
from ibis.compat import PY2
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.expr.temporal as T
import ibis.expr.api as api


@pytest.mark.parametrize(
    ('offset', 'unit', 'expected'),
    [
        (T.day(14), 'w', T.week(2)),
        (T.hour(72), 'd', T.day(3)),
        (T.minute(240), 'h', T.hour(4)),
        (T.second(360), 'm', T.minute(6)),
        (T.second(3 * 86400), 'd', T.day(3)),
        (T.millisecond(5000), 's', T.second(5)),
        (T.microsecond(5000000), 's', T.second(5)),
        (T.nanosecond(5000000000), 's', T.second(5)),
    ]
)
def test_upconvert(offset, unit, expected):
    result = offset.to_unit(unit)
    assert result.equals(expected)


def test_multiply():
    offset = T.day(2)

    assert (offset * 2).equals(T.day(4))
    assert (offset * (-2)).equals(T.day(-4))
    assert (3 * offset).equals(T.day(6))
    assert ((-3) * offset).equals(T.day(-6))


def test_repr():
    assert repr(T.day()) == '<Timedelta: 1 day>'
    assert repr(T.day(2)) == '<Timedelta: 2 days>'
    assert repr(T.year()) == '<Timedelta: 1 year>'
    assert repr(T.month(2)) == '<Timedelta: 2 months>'
    assert repr(T.second(40)) == '<Timedelta: 40 seconds>'


@pytest.mark.parametrize(
    ('delta', 'target'),
    [
        (T.day(), 'w'),
        (T.hour(), 'd'),
        (T.minute(), 'h'),
        (T.second(), 'm'),
        (T.second(), 'd'),
        (T.millisecond(), 's'),
        (T.microsecond(), 's'),
        (T.nanosecond(), 's'),
    ]
)
def test_cannot_upconvert(delta, target):
    with pytest.raises(IbisError):
        delta.to_unit(target)


@pytest.mark.parametrize(
    ('case', 'expected'),
    [
        (T.second(2).to_unit('s'), T.second(2)),
        (T.second(2).to_unit('ms'), T.millisecond(2 * 1000)),
        (T.second(2).to_unit('us'), T.microsecond(2 * 1000000)),
        (T.second(2).to_unit('ns'), T.nanosecond(2 * 1000000000)),

        (T.millisecond(2).to_unit('ms'), T.millisecond(2)),
        (T.millisecond(2).to_unit('us'), T.microsecond(2 * 1000)),
        (T.millisecond(2).to_unit('ns'), T.nanosecond(2 * 1000000)),

        (T.microsecond(2).to_unit('us'), T.microsecond(2)),
        (T.microsecond(2).to_unit('ns'), T.nanosecond(2 * 1000)),

        (T.nanosecond(2).to_unit('ns'), T.nanosecond(2)),
    ]
)
def test_downconvert_second_parts(case, expected):
    assert case.equals(expected)


@pytest.mark.parametrize(
    ('case', 'expected'),
    [
        (T.hour(2).to_unit('h'), T.hour(2)),
        (T.hour(2).to_unit('m'), T.minute(2 * 60)),
        (T.hour(2).to_unit('s'), T.second(2 * 3600)),
        (T.hour(2).to_unit('ms'), T.millisecond(2 * 3600000)),
        (T.hour(2).to_unit('us'), T.microsecond(2 * 3600000000)),
        (T.hour(2).to_unit('ns'), T.nanosecond(2 * 3600000000000))
    ]
)
def test_downconvert_hours(case, expected):
    assert case.equals(expected)


@pytest.mark.parametrize(
    ('case', 'expected'),
    [
        (T.week(2).to_unit('d'), T.day(2 * 7)),
        (T.week(2).to_unit('h'), T.hour(2 * 7 * 24)),

        (T.day(2).to_unit('d'), T.day(2)),
        (T.day(2).to_unit('h'), T.hour(2 * 24)),
        (T.day(2).to_unit('m'), T.minute(2 * 1440)),
        (T.day(2).to_unit('s'), T.second(2 * 86400)),
        (T.day(2).to_unit('ms'), T.millisecond(2 * 86400000)),
        (T.day(2).to_unit('us'), T.microsecond(2 * 86400000000)),
        (T.day(2).to_unit('ns'), T.nanosecond(2 * 86400000000000)),
    ]
)
def test_downconvert_day(case, expected):
    assert case.equals(expected)


@pytest.mark.parametrize(
    ('case', 'expected'),
    [
        (T.day() + T.minute(), T.minute(1441)),
        (T.second() + T.millisecond(10), T.millisecond(1010)),
        (T.hour() + T.minute(5) + T.second(10), T.second(3910)),
    ]
)
def test_combine_with_different_kinds(case, expected):
    assert case.equals(expected)


@pytest.mark.parametrize(
    ('case', 'expected'),
    [
        (T.timedelta(weeks=2), T.week(2)),
        (T.timedelta(days=3), T.day(3)),
        (T.timedelta(hours=4), T.hour(4)),
        (T.timedelta(minutes=5), T.minute(5)),
        (T.timedelta(seconds=6), T.second(6)),
        (T.timedelta(milliseconds=7), T.millisecond(7)),
        (T.timedelta(microseconds=8), T.microsecond(8)),
        (T.timedelta(nanoseconds=9), T.nanosecond(9)),
    ]
)
def test_timedelta_generic_api(case, expected):
    assert case.equals(expected)


# def test_offset_timestamp_expr(table):
#     c = table.i
#     x = T.timedelta(days=1)

#     expr = x + c
#     assert isinstance(expr, ir.TimestampColumn)
#     assert isinstance(expr.op(), ops.TimestampDelta)

#     # test radd
#     expr = c + x
#     assert isinstance(expr, ir.TimestampColumn)
#     assert isinstance(expr.op(), ops.TimestampDelta)


@pytest.mark.xfail(raises=AssertionError, reason='NYI')
def test_compound_offset():
    # These are not yet allowed (e.g. 1 month + 1 hour)
    assert False


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_offset_months():
    assert False


@pytest.mark.parametrize('literal', [
    api.interval(3600),
    api.interval(datetime.timedelta(days=3)),
    api.interval(years=1),
    api.interval(months=2),
    api.interval(weeks=3),
    api.interval(days=-1),
    api.interval(hours=-3),
    api.interval(minutes=5),
    api.interval(seconds=-8)
])
def test_interval(literal):
    assert isinstance(literal, ir.IntervalScalar)


# TODO: parametrize tests
def test_interval_repr():
    assert repr(api.interval(weeks=3)) == "Literal[interval('w')]\n  3"
    assert repr(api.interval(months=3)) == "Literal[interval('M')]\n  3"
    assert repr(api.interval(seconds=-10)) == "Literal[interval('s')]\n  -10"


def test_timestamp_arithmetics():
    ts1 = api.timestamp(datetime.datetime.now())
    ts2 = api.timestamp(datetime.datetime.today())

    i1 = api.interval(minutes=30)

    # TODO: raise for unsupported operations too
    for expr in [ts2 - ts1, ts1 - ts2]:
        assert isinstance(expr, ir.IntervalScalar)
        assert isinstance(expr.op(), ops.TimestampSubtract)
        assert expr.type() == dt.Interval('s')

    for expr in [ts1 - i1, ts2 - i1]:
        assert isinstance(expr, ir.TimestampScalar)
        assert isinstance(expr.op(), ops.TimestampSubtract)

    for expr in [ts1 + i1, ts2 + i1]:
        assert isinstance(expr, ir.TimestampScalar)
        assert isinstance(expr.op(), ops.TimestampAdd)


def test_date_arithmetics():
    d1 = api.date('2015-01-02')
    d2 = api.date('2017-01-01')

    i1 = api.interval(weeks=3)

    # TODO: raise for unsupported operations too
    for expr in [d1 - d2, d2 - d1]:
        assert isinstance(expr, ir.IntervalScalar)
        assert isinstance(expr.op(), ops.DateSubtract)
        assert expr.type() == dt.Interval('d')

    for expr in [d1 - i1, d2 - i1]:
        assert isinstance(expr, ir.DateScalar)
        assert isinstance(expr.op(), ops.DateSubtract)

    for expr in [d1 + i1, d2 + i1]:
        assert isinstance(expr, ir.DateScalar)
        assert isinstance(expr.op(), ops.DateAdd)


@pytest.mark.skipif(PY2, reason='time support is not enabled on python 2')
def test_time_arithmetics():
    t1 = api.time('18:00')
    t2 = api.time('19:12')

    i1 = api.interval(minutes=3)

    # TODO: raise for unsupported operations too
    for expr in [t1 - t2, t2 - t1]:
        assert isinstance(expr, ir.IntervalScalar)
        assert isinstance(expr.op(), ops.TimeSubtract)
        assert expr.type() == dt.Interval('s')

    for expr in [t1 - i1, t2 - i1]:
        assert isinstance(expr, ir.TimeScalar)
        assert isinstance(expr.op(), ops.TimeSubtract)

    for expr in [t1 + i1, t2 + i1]:
        assert isinstance(expr, ir.TimeScalar)
        assert isinstance(expr.op(), ops.TimeAdd)


def test_invalid_date_arithmetics():
    d1 = api.date('2015-01-02')
    i1 = api.interval(seconds=300)
    i2 = api.interval(minutes=15)
    i3 = api.interval(hours=1)

    for i in [i1, i2, i3]:
        with pytest.raises(IbisTypeError):
            d1 - i
        with pytest.raises(IbisTypeError):
            d1 + i


def test_interval_properties():
    i = api.interval(seconds=300)

    i.nanoseconds
    i.microseconds
    i.milliseconds
    i.seconds
    i.minutes
    i.hours
    i.days
    i.weeks


# TODO:
# assert for ibis type errors to check rules
# assert for d1 - d2 == -(d2 - d1)
# assert for ts1 - ts2 == -(ts2 - ts1)
