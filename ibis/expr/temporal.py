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
import ibis.expr.types as ir
import ibis.expr.operations as ops


__all__ = ['timedelta', 'year', 'month', 'week', 'day',
           'hour', 'minute', 'second',
           'millisecond', 'microsecond', 'nanosecond']


class Timedelta(object):
    """
    Represents any kind of date/time/timestamp increment, the precise length
    possibly dependent on the timestamp being modified.
    """
    def __init__(self, n):
        self.n = int(n)

    def __mul__(self, times):
        pass

    __rmul__ = __mul__

    def __add__(self, expr):
        pass

    def __sub__(self, expr):
        pass

    def equals(self, other):
        if type(self) != type(other):
            return False

        return self.n == other.n


class TimeIncrement(Timedelta):

    @property
    def unit(self):
        return self._unit

    def to_unit(self, target_unit):
        """

        """
        target_unit = target_unit.lower()
        klass = _timedelta_units[target_unit]
        increments = CONVERTER.convert(self.n, self.unit, target_unit)
        return klass(increments)


class Nanosecond(TimeIncrement):
    _unit = 'ns'


class Microsecond(TimeIncrement):
    _unit = 'us'


class Millisecond(TimeIncrement):
    _unit = 'ms'


class Second(TimeIncrement):
    _unit = 's'


class Minute(TimeIncrement):
    _unit = 'm'


class Hour(TimeIncrement):
    _unit = 'h'


class Day(TimeIncrement):
    _unit = 'd'


class Week(TimeIncrement):
    _unit = 'w'


class Month(Timedelta):
    _unit = 'M'


class Year(Timedelta):
    _unit = 'Y'


_timedelta_units = {
    'Y': Year,
    'M': Month,
    'w': Week,
    'd': Day,
    'h': Hour,
    'm': Minute,
    's': Second,
    'ms': Millisecond,
    'us': Microsecond,
    'ns': Nanosecond
}



class UnitConverter(object):

    def __init__(self, ordering, conv_factors):
        self.ordering = ordering
        self.conv_factors = conv_factors
        self.ranks = dict((name, i) for i, name in enumerate(ordering))

    def convert(self, n, from_unit, to_unit):
        i = self.ranks[from_unit]
        j = self.ranks[to_unit]

        if i == j:
            return n

        factors = self.conv_factors[i + 1 : j + 1]
        factor = 1
        for x in factors:
            factor *= x

        if j < i:
            if not n % factor:
                raise IbisError('{} is not a multiple of {}'.format(n, factor))
            return n / factor
        else:
            return n * factor

    def anglicize(self, n, unit):
        raise NotImplementedError


_ordering = ['w', 'd', 'h', 'm', 's', 'ms', 'us', 'ns']
_factors = [1, 7, 24, 60, 60, 1000, 1000, 1000]
_name = ['week', 'day', 'hour', 'minute', 'second',
         'millisecond', 'microsecond', 'nanosecond']


CONVERTER = UnitConverter(_ordering, _factors)


def _conversion_factor(source, target):
    pass


def _delta_factory(name, unit):
    klass = _timedelta_units[unit]
    def factory(n=1):
        return klass(n)

    factory.__name__ = name

    return factory

nanosecond = _delta_factory('nanosecond', 'ns')
microsecond = _delta_factory('microsecond', 'us')
millisecond = _delta_factory('millisecond', 'ms')
second = _delta_factory('second', 's')
minute = _delta_factory('minute', 'm')
hour = _delta_factory('hour', 'h')
day = _delta_factory('day', 'd')
week = _delta_factory('week', 'w')
month = _delta_factory('month', 'M')
year = _delta_factory('year', 'Y')


def timedelta(days=None,
              weeks=None,
              hours=None,
              minutes=None,
              seconds=None,
              milliseconds=None,
              microseconds=None,
              nanoseconds=None):
    """

    """
    pass



class TimestampDelta(ops.ValueNode):

    def __init__(self, arg, offset):
        self.arg = ops.as_value_expr(arg)
        pass
