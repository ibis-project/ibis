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

from ibis.common import IbisError
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.expr.temporal as T

from ibis.expr.tests.mocks import MockConnection
from ibis.compat import unittest


class TestFixedOffsets(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('alltypes')

    def test_upconvert(self):
        cases = [
            (T.day(14), 'w', T.week(2)),
            (T.hour(72), 'd', T.day(3)),
            (T.minute(240), 'h', T.hour(4)),
            (T.second(360), 'm', T.minute(6)),
            (T.second(3 * 86400), 'd', T.day(3)),
            (T.millisecond(5000), 's', T.second(5)),
            (T.microsecond(5000000), 's', T.second(5)),
            (T.nanosecond(5000000000), 's', T.second(5)),
        ]

        for offset, unit, expected in cases:
            result = offset.to_unit(unit)
            assert result.equals(expected)

    def test_multiply(self):
        offset = T.day(2)

        assert (offset * 2).equals(T.day(4))
        assert (offset * (-2)).equals(T.day(-4))
        assert (3 * offset).equals(T.day(6))
        assert ((-3) * offset).equals(T.day(-6))

    def test_repr(self):
        assert repr(T.day()) == '<Timedelta: 1 day>'
        assert repr(T.day(2)) == '<Timedelta: 2 days>'
        assert repr(T.year()) == '<Timedelta: 1 year>'
        assert repr(T.month(2)) == '<Timedelta: 2 months>'
        assert repr(T.second(40)) == '<Timedelta: 40 seconds>'

    def test_cannot_upconvert(self):
        cases = [
            (T.day(), 'w'),
            (T.hour(), 'd'),
            (T.minute(), 'h'),
            (T.second(), 'm'),
            (T.second(), 'd'),
            (T.millisecond(), 's'),
            (T.microsecond(), 's'),
            (T.nanosecond(), 's'),
        ]

        for delta, target in cases:
            self.assertRaises(IbisError, delta.to_unit, target)

    def test_downconvert_second_parts(self):
        K = 2

        sec = T.second(K)
        milli = T.millisecond(K)
        micro = T.microsecond(K)
        nano = T.nanosecond(K)

        cases = [
            (sec.to_unit('s'), T.second(K)),
            (sec.to_unit('ms'), T.millisecond(K * 1000)),
            (sec.to_unit('us'), T.microsecond(K * 1000000)),
            (sec.to_unit('ns'), T.nanosecond(K * 1000000000)),

            (milli.to_unit('ms'), T.millisecond(K)),
            (milli.to_unit('us'), T.microsecond(K * 1000)),
            (milli.to_unit('ns'), T.nanosecond(K * 1000000)),

            (micro.to_unit('us'), T.microsecond(K)),
            (micro.to_unit('ns'), T.nanosecond(K * 1000)),

            (nano.to_unit('ns'), T.nanosecond(K))
        ]
        self._check_cases(cases)

    def test_downconvert_hours(self):
        K = 2
        offset = T.hour(K)

        cases = [
            (offset.to_unit('h'), T.hour(K)),
            (offset.to_unit('m'), T.minute(K * 60)),
            (offset.to_unit('s'), T.second(K * 3600)),
            (offset.to_unit('ms'), T.millisecond(K * 3600000)),
            (offset.to_unit('us'), T.microsecond(K * 3600000000)),
            (offset.to_unit('ns'), T.nanosecond(K * 3600000000000))
        ]
        self._check_cases(cases)

    def test_downconvert_day(self):
        K = 2

        week = T.week(K)
        day = T.day(K)

        cases = [
            (week.to_unit('d'), T.day(K * 7)),
            (week.to_unit('h'), T.hour(K * 7 * 24)),

            (day.to_unit('d'), T.day(K)),
            (day.to_unit('h'), T.hour(K * 24)),
            (day.to_unit('m'), T.minute(K * 1440)),
            (day.to_unit('s'), T.second(K * 86400)),
            (day.to_unit('ms'), T.millisecond(K * 86400000)),
            (day.to_unit('us'), T.microsecond(K * 86400000000)),
            (day.to_unit('ns'), T.nanosecond(K * 86400000000000))
        ]
        self._check_cases(cases)

    def test_combine_with_different_kinds(self):
        cases = [
            (T.day() + T.minute(), T.minute(1441)),
            (T.second() + T.millisecond(10), T.millisecond(1010)),
            (T.hour() + T.minute(5) + T.second(10), T.second(3910))
        ]
        self._check_cases(cases)

    def test_timedelta_generic_api(self):
        cases = [
            (T.timedelta(weeks=2), T.week(2)),
            (T.timedelta(days=3), T.day(3)),
            (T.timedelta(hours=4), T.hour(4)),
            (T.timedelta(minutes=5), T.minute(5)),
            (T.timedelta(seconds=6), T.second(6)),
            (T.timedelta(milliseconds=7), T.millisecond(7)),
            (T.timedelta(microseconds=8), T.microsecond(8)),
            (T.timedelta(nanoseconds=9), T.nanosecond(9)),
        ]
        self._check_cases(cases)

    def _check_cases(self, cases):
        for x, y in cases:
            assert x.equals(y)

    def test_offset_timestamp_expr(self):
        c = self.table.i
        x = T.timedelta(days=1)

        expr = x + c
        assert isinstance(expr, ir.TimestampArray)
        assert isinstance(expr.op(), ops.TimestampDelta)

        # test radd
        expr = c + x
        assert isinstance(expr, ir.TimestampArray)
        assert isinstance(expr.op(), ops.TimestampDelta)


class TestTimedelta(unittest.TestCase):

    def test_compound_offset(self):
        # These are not yet allowed (e.g. 1 month + 1 hour)
        pass

    def test_offset_months(self):
        pass
