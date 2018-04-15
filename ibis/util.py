from __future__ import print_function

import collections
import functools
import operator
import types

import six

import ibis.compat as compat
from ibis.config import options


def guid():
    try:
        from ibis.comms import uuid4_hex
        return uuid4_hex()
    except ImportError:
        from uuid import uuid4
        guid = uuid4()
        return guid.hex if not compat.PY2 else guid.get_hex()


def indent(text, spaces):
    prefix = ' ' * spaces
    return ''.join(prefix + line for line in text.splitlines(True))


def any_of(values, t):
    for x in values:
        if isinstance(x, t):
            return True
    return False


def all_of(values, t):
    for x in values:
        if not isinstance(x, t):
            return False
    return True


def promote_list(val):
    if not isinstance(val, list):
        val = [val]
    return val


class IbisSet(object):

    def __init__(self, keys=None):
        self.keys = keys or []

    @classmethod
    def from_list(cls, keys):
        return IbisSet(keys)

    def __contains__(self, obj):
        for other in self.keys:
            if obj.equals(other):
                return True
        return False

    def add(self, obj):
        self.keys.append(obj)


class IbisMap(object):

    def __init__(self):
        self.keys = []
        self.values = []

    def __contains__(self, obj):
        for other in self.keys:
            if obj.equals(other):
                return True
        return False

    def set(self, key, value):
        self.keys.append(key)
        self.values.append(value)

    def get(self, key):
        for k, v in zip(self.keys, self.values):
            if key.equals(k):
                return v
        raise KeyError(key)


def is_function(v):
    return isinstance(v, (types.FunctionType, types.LambdaType))


def adjoin(space, *lists):
    """
    Glues together two sets of strings using the amount of space requested.
    The idea is to prettify.

    Brought over from from pandas
    """
    out_lines = []
    newLists = []
    lengths = [max(map(len, x)) + space for x in lists[:-1]]

    # not the last one
    lengths.append(max(map(len, lists[-1])))

    maxLen = max(map(len, lists))
    for i, lst in enumerate(lists):
        nl = [x.ljust(lengths[i]) for x in lst]
        nl.extend([' ' * lengths[i]] * (maxLen - len(lst)))
        newLists.append(nl)
    toJoin = zip(*newLists)
    for lines in toJoin:
        out_lines.append(_join_unicode(lines))
    return _join_unicode(out_lines, sep='\n')


def _join_unicode(lines, sep=''):
    try:
        return sep.join(lines)
    except UnicodeDecodeError:
        sep = compat.unicode_type(sep)
        return sep.join([x.decode('utf-8') if isinstance(x, str) else x
                         for x in lines])


def deprecate(f, message):
    def g(*args, **kwargs):
        print(message)
        return f(*args, **kwargs)
    return g


def log(msg):
    if options.verbose:
        (options.verbose_log or print)(msg)


class cache_readonly(object):

    def __init__(self, func=None, allow_setting=False):
        if func is not None:
            self.func = func
            self.name = func.__name__
            self.__doc__ = func.__doc__
        self.allow_setting = allow_setting

    def __call__(self, func):
        self.func = func
        self.name = func.__name__
        return self

    def __get__(self, obj, typ):
        # Get the cache or set a default one if needed

        cache = getattr(obj, '_cache', None)
        if cache is None:
            try:
                cache = obj._cache = {}
            except (AttributeError):
                return

        if self.name in cache:
            val = cache[self.name]
        else:
            val = self.func(obj)
            cache[self.name] = val
        return val

    def __set__(self, obj, value):
        if not self.allow_setting:
            raise Exception("cannot set values for [%s]" % self.name)

        # Get the cache or set a default one if needed
        cache = getattr(obj, '_cache', None)
        if cache is None:
            try:
                cache = obj._cache = {}
            except (AttributeError):
                return

        cache[self.name] = value


def approx_equal(a, b, eps):
    assert abs(a - b) < eps


def implements(f):
    def decorator(g):
        g.__doc__ = f.__doc__
        return g
    return decorator


def safe_index(elements, value):
    """Find the location of `value` in `elements`, return -1 if `value` is
    not found instead of raising ``ValueError``.

    Parameters
    ----------
    elements : Sequence
    value : object

    Returns
    -------
    location : object

    Examples
    --------
    >>> sequence = [1, 2, 3]
    >>> safe_index(sequence, 2)
    1
    >>> safe_index(sequence, 4)
    -1
    """
    try:
        return elements.index(value)
    except ValueError:
        return -1


def is_sequence(o):
    """Is `o` a non-string sequence?

    Parameters
    ----------
    o : object

    Returns
    -------
    is_seq : bool
    """
    return (not isinstance(o, six.string_types) and
            isinstance(o, collections.Iterable))


def convert_unit(value, unit, to):
    """Convert `value`--which is assumed to be in units of `unit`--to units of
    `to`.

    Parameters
    ----------
    value : Union[numbers.Real, ibis.expr.types.NumericValue]

    Returns
    -------
    result : Union[numbers.Integral, ibis.expr.types.NumericValue]

    Examples
    --------
    >>> one_second = 1000
    >>> x = convert_unit(one_second, 'ms', 's')
    >>> x
    1
    >>> one_second = 1
    >>> x = convert_unit(one_second, 's', 'ms')
    >>> x
    1000
    >>> x = convert_unit(one_second, 's', 's')
    >>> x
    1
    >>> x = convert_unit(one_second, 's', 'M')
    Traceback (most recent call last):
        ...
    ValueError: Cannot convert to or from variable length interval
    """
    # Don't do anything from and to units are equivalent

    if unit == to:
        return value

    units = ('W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns')
    factors = (7, 24, 60, 60, 1000, 1000, 1000)

    monthly_units = ('Y', 'Q', 'M')
    monthly_factors = (4, 3)

    try:
        i, j = units.index(unit), units.index(to)
    except ValueError:
        try:
            i, j = monthly_units.index(unit), monthly_units.index(to)
            factors = monthly_factors
        except ValueError:
            raise ValueError('Cannot convert to or from '
                             'non-fixed-length interval')

    factor = functools.reduce(operator.mul, factors[min(i, j):max(i, j)], 1)
    assert factor > 1

    if i < j:
        return value * factor

    assert i > j
    return value // factor
