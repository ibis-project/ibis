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


__all__ = ['timedelta', 'year', 'month', 'week', 'day',
           'hour', 'minute', 'second',
           'millisecond', 'microsecond']


class Timedelta(object):
    """
    Represents any kind of date/time/timestamp increment, the precise length
    possibly dependent on the timestamp being modified.
    """
    def __init__(self, n):
        self.n = int(n)

    @property
    def unit(self):
        raise NotImplementedError

    @property
    def unit_name(self):
        return type(self).__name__.lower()

    def __repr__(self):
        if self.n == 1:
            pretty_unit = self.unit_name
        else:
            pretty_unit = '{0}s'.format(self.unit_name)

        return '<Timedelta: {0} {1}>'.format(self.n, pretty_unit)

    def replace(self, n):
        return type(self)(n)

    def __mul__(self, times):
        return self.replace(self.n * times)

    __rmul__ = __mul__

    def __add__(self, arg):
        from ibis.expr.operations import TimestampDelta

        if isinstance(arg, ir.TimestampValue):
            op = TimestampDelta(arg, self)
            return op.to_expr()
        elif isinstance(arg, Timedelta):
            return self.combine(arg)
        else:
            raise TypeError(arg)

    __radd__ = __add__

    def __sub__(self, arg):
        if isinstance(arg, ir.Expr):
            raise TypeError(arg)
        elif isinstance(arg, Timedelta):
            return self.combine(arg.replace(-arg.n))
        else:
            raise NotImplementedError

    def __rsub__(self, arg):
        return self.replace(-self.n).__add__(arg)

    def combine(self, other):
        if type(self) != type(other):
            raise TypeError(type(other))

        klass = type(self)
        return klass(self.n + other.n)

    def equals(self, other):
        if type(self) != type(other):
            return False

        return self.n == other.n


class TimeIncrement(Timedelta):

    @property
    def unit(self):
        return self._unit

    def combine(self, other):
        if not isinstance(other, TimeIncrement):
            raise TypeError('Must be a fixed size timedelta, was {0!r}'
                            .format(type(other)))

        a, b = _to_common_units([self, other])
        return type(a)(a.n + b.n)

    def to_unit(self, target_unit):
        """

        """
        target_unit = target_unit.lower()
        if self.unit == target_unit:
            return self

        klass = _timedelta_units[target_unit]
        increments = CONVERTER.convert(self.n, self.unit, target_unit)
        return klass(increments)


def _to_common_units(args):
    common_unit = CONVERTER.get_common_unit([x.unit for x in args])
    return [x.to_unit(common_unit) for x in args]


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

    def __init__(self, ordering, conv_factors, names):
        self.ordering = ordering
        self.conv_factors = conv_factors
        self.names = names

        self.ranks = dict((name, i) for i, name in enumerate(ordering))
        self.rank_to_unit = dict((v, k) for k, v in self.ranks.items())

    def get_common_unit(self, units):
        min_rank = max(self.ranks[x] for x in units)
        return self.rank_to_unit[min_rank]

    def convert(self, n, from_unit, to_unit):
        i = self.ranks[from_unit]
        j = self.ranks[to_unit]

        if i == j:
            return n

        factors = self.conv_factors[min(i, j) + 1: max(i, j) + 1]
        factor = 1
        for x in factors:
            factor *= x

        if j < i:
            if n % factor:
                raise IbisError('{0} is not a multiple of {1}'.format(n,
                                                                      factor))
            return n / factor
        else:
            return n * factor

    def anglicize(self, n, unit):
        raise NotImplementedError


_ordering = ['w', 'd', 'h', 'm', 's', 'ms', 'us', 'ns']
_factors = [1, 7, 24, 60, 60, 1000, 1000, 1000]
_names = ['week', 'day', 'hour', 'minute', 'second',
          'millisecond', 'microsecond', 'nanosecond']


CONVERTER = UnitConverter(_ordering, _factors, _names)


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


def timedelta(days=None, hours=None, minutes=None, seconds=None,
              milliseconds=None, microseconds=None, nanoseconds=None,
              weeks=None):
    """
    Generic API for creating a fixed size timedelta

    Parameters
    ----------
    days : int, default None
    weeks : int, default None
    hours : int, default None
    minutes : int, default None
    seconds : int, default None
    milliseconds : int, default None
    microseconds : int, default None
    nanoseconds : int, default None

    Notes
    -----
    For potentially non-fixed-length timedeltas (like year, month, etc.), use
    the corresponding named API (e.g. ibis.month).

    Returns
    -------
    delta : TimeIncrement (Timedelta)
    """
    out = {
        'result': None
    }

    def _apply(klass, n):
        if not n:
            return
        offset = klass(n)
        delta = out['result']
        out['result'] = delta + offset if delta else offset

    _apply(Week, weeks)
    _apply(Day, days)
    _apply(Hour, hours)
    _apply(Minute, minutes)
    _apply(Second, seconds)
    _apply(Millisecond, milliseconds)
    _apply(Microsecond, microseconds)
    _apply(Nanosecond, nanoseconds)

    result = out['result']
    if not result:
        raise IbisError('Must pass some offset parameter')

    return result
