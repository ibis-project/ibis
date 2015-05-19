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

import unittest

from ibis.common import IbisError
import ibis.expr.temporal as T


class TestFixedOffsets(unittest.TestCase):

    def test_upconvert(self):
        pass

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
            (offset.to_unit('ns'), T.nanosecond(K * 3600000000000L))
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
            (day.to_unit('ns'), T.nanosecond(K * 86400000000000L))
        ]
        self._check_cases(cases)

    def test_combine_with_different_kinds(self):
        cases = [


        ]
        self._check_cases(cases)

    def test_timedelta_generic_api(self):
        cases = [


        ]
        self._check_cases(cases)

    def _check_cases(self, cases):
        for x, y in cases:
            assert x.equals(y)


class TestTimedelta(unittest.TestCase):

    def test_compound_offset(self):
        # These are not yet allowed (e.g. 1 month + 1 hour)
        pass
