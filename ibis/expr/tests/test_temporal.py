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
import ibis.expr.api as api


@pytest.mark.parametrize(('interval', 'unit', 'expected'), [
    (api.day(14), 'w', api.week(2)),
    (api.minute(240), 'h', api.hour(4)),
    (api.second(360), 'm', api.minute(6)),
    (api.second(3 * 86400), 'd', api.day(3)),
    (api.millisecond(5000), 's', api.second(5)),
    (api.microsecond(5000000), 's', api.second(5)),
    (api.nanosecond(5000000000), 's', api.second(5)),
])
def test_upconvert(interval, unit, expected):
    result = interval.to_unit(unit)
    assert result.type().unit == expected.type().unit


@pytest.mark.parametrize(('delta', 'target'), [
    (api.day(), 'w'),
    (api.hour(), 'd'),
    (api.minute(), 'h'),
    (api.second(), 'm'),
    (api.second(), 'd'),
    (api.millisecond(), 's'),
    (api.microsecond(), 's'),
    (api.nanosecond(), 's'),
])
def test_cannot_upconvert(delta, target):
    assert delta.to_unit(target).type().unit == target


# def test_multiply():
#     offset = api.day(2)

#     assert (offset * 2).equals(api.day(4))
#     assert (offset * (-2)).equals(api.day(-4))
#     assert (3 * offset).equals(api.day(6))
#     assert ((-3) * offset).equals(api.day(-6))


# def test_repr():
#     assert repr(T.day()) == '<Timedelta: 1 day>'
#     assert repr(T.day(2)) == '<Timedelta: 2 days>'
#     assert repr(T.year()) == '<Timedelta: 1 year>'
#     assert repr(T.month(2)) == '<Timedelta: 2 months>'
#     assert repr(T.second(40)) == '<Timedelta: 40 seconds>'

# @pytest.mark.parametrize(
#     ('case', 'expected'),
#     [
#         (api.second(2).to_unit('s'), api.second(2)),
#         (api.second(2).to_unit('ms'), api.millisecond(2 * 1000)),
#         (api.second(2).to_unit('us'), api.microsecond(2 * 1000000)),
#         (api.second(2).to_unit('ns'), api.nanosecond(2 * 1000000000)),

#         (api.millisecond(2).to_unit('ms'), api.millisecond(2)),
#         (api.millisecond(2).to_unit('us'), api.microsecond(2 * 1000)),
#         (api.millisecond(2).to_unit('ns'), api.nanosecond(2 * 1000000)),

#         (api.microsecond(2).to_unit('us'), api.microsecond(2)),
#         (api.microsecond(2).to_unit('ns'), api.nanosecond(2 * 1000)),

#         (api.nanosecond(2).to_unit('ns'), api.nanosecond(2)),
#     ]
# )
# def test_downconvert_second_parts(case, expected):
#     assert case.equals(expected)


# @pytest.mark.parametrize(
#     ('case', 'expected'),
#     [
#         (api.hour(2).to_unit('h'), api.hour(2)),
#         (api.hour(2).to_unit('m'), api.minute(2 * 60)),
#         (api.hour(2).to_unit('s'), api.second(2 * 3600)),
#         (api.hour(2).to_unit('ms'), api.millisecond(2 * 3600000)),
#         (api.hour(2).to_unit('us'), api.microsecond(2 * 3600000000)),
#         (api.hour(2).to_unit('ns'), api.nanosecond(2 * 3600000000000))
#     ]
# )
# def test_downconvert_hours(case, expected):
#     assert case.equals(expected)


# @pytest.mark.parametrize(
#     ('case', 'expected'),
#     [
#         (api.week(2).to_unit('d'), api.day(2 * 7)),
#         (api.week(2).to_unit('h'), api.hour(2 * 7 * 24)),

#         (api.day(2).to_unit('d'), api.day(2)),
#         (api.day(2).to_unit('h'), api.hour(2 * 24)),
#         (api.day(2).to_unit('m'), api.minute(2 * 1440)),
#         (api.day(2).to_unit('s'), api.second(2 * 86400)),
#         (api.day(2).to_unit('ms'), api.millisecond(2 * 86400000)),
#         (api.day(2).to_unit('us'), api.microsecond(2 * 86400000000)),
#         (api.day(2).to_unit('ns'), api.nanosecond(2 * 86400000000000)),
#     ]
# )
# def test_downconvert_day(case, expected):
#     assert case.equals(expected)


@pytest.mark.parametrize(('a', 'b', 'unit'), [
    (api.day(), api.day(3), 'd'),
    (api.second(), api.hour(10), 's'),
    (api.hour(3), api.day(2), 'h')
])
def test_combine_with_different_kinds(a, b, unit):
    assert (a + b).type().unit == unit


# @pytest.mark.parametrize(('case', 'expected'), [
#     (api.interval(weeks=2), api.week(2)),
#     (api.interval(days=3), api.day(3)),
#     (api.interval(hours=4), api.hour(4)),
#     (api.interval(minutes=5), api.minute(5)),
#     (api.interval(seconds=6), api.second(6)),
#     (api.interval(milliseconds=7), api.millisecond(7)),
#     (api.interval(microseconds=8), api.microsecond(8)),
#     (api.interval(nanoseconds=9), api.nanosecond(9)),
# ])
# def test_timedelta_generic_api(case, expected):
#     assert case.equals(expected)


# def test_offset_timestamp_expr(table):
#     c = table.i
#     x = api.timedelta(days=1)

#     expr = x + c
#     assert isinstance(expr, ir.TimestampColumn)
#     assert isinstance(expr.op(), ops.TimestampDelta)

#     # test radd
#     expr = c + x
#     assert isinstance(expr, ir.TimestampColumn)
#     assert isinstance(expr.op(), ops.TimestampDelta)


# @pytest.mark.xfail(raises=AssertionError, reason='NYI')
# def test_compound_offset():
#     # These are not yet allowed (e.g. 1 month + 1 hour)
#     assert False


# @pytest.mark.xfail(raises=AssertionError, reason='NYT')
# def test_offset_months():
#     assert False


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



@pytest.mark.parametrize(('expr', 'expected'), [
    (api.interval(weeks=3), "Literal[interval<int8>(unit='w')]\n  3"),
    (api.interval(months=3), "Literal[interval<int8>(unit='M')]\n  3"),
    (api.interval(seconds=-10), "Literal[interval<int8>(unit='s')]\n  -10")
])
def test_interval_repr(expr, expected):
    assert repr(expr) == expected


def test_timestamp_arithmetics():
    ts1 = api.timestamp(datetime.datetime.now())
    ts2 = api.timestamp(datetime.datetime.today())

    i1 = api.interval(minutes=30)

    # TODO: raise for unsupported operations too
    for expr in [ts2 - ts1, ts1 - ts2]:
        assert isinstance(expr, ir.IntervalScalar)
        assert isinstance(expr.op(), ops.TimestampSubtract)
        assert expr.type() == dt.Interval(dt.int32, 's')

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
        assert expr.type() == dt.Interval(dt.int32, 'd')

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
        assert expr.type() == dt.Interval(dt.int32, 's')

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
    i = api.interval(seconds=3600)

    i.nanoseconds
    i.microseconds
    i.milliseconds
    i.seconds
    i.minutes
    i.hours
    i.days
    i.weeks

